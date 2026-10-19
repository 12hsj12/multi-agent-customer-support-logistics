"""Decision provider adapters used by intent arbitration.

The Mapika provider loads the checked-in GGUF model in process. The HTTP
adapter remains available for a separately hosted OpenJev-compatible service.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)


class DecisionStatus(str, Enum):
    DECIDED = "decided"
    ABSTAINED = "abstained"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class DecisionResult:
    """A normalized provider result; confidence is an optional raw signal."""

    label: Optional[str]
    confidence: Optional[float]
    source: str
    latency_ms: float
    status: DecisionStatus
    error_code: Optional[str] = None


class DecisionProvider(Protocol):
    async def decide(
        self,
        message: str,
        candidates: Mapping[str, str],
        history: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> DecisionResult:
        """Choose one supplied candidate, or return a non-decided status."""


class OpenJevProvider:
    """HTTP adapter for a locally hosted OpenJev-compatible decider server."""

    source = "openjev"

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 8.0,
        api_key: Optional[str] = None,
    ):
        if not base_url or not base_url.strip():
            raise ValueError("OpenJev base URL must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("OpenJev timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key.strip() if api_key and api_key.strip() else None

    def _payload(
        self,
        message: str,
        candidates: Mapping[str, str],
        history: Optional[Sequence[Mapping[str, str]]],
    ) -> dict:
        state = {"message": message}
        if history:
            state["history"] = [
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        return {
            "state": state,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": (
                        "根据用户本轮主要诉求选择最匹配的一个意图。只有确实不属于任何候选意图时选择 other。"
                    ),
                    "criteria": dict(candidates),
                }
            },
        }

    async def _post_json(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/systemone",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise ValueError("response_not_object")
        return body

    async def decide(
        self,
        message: str,
        candidates: Mapping[str, str],
        history: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> DecisionResult:
        started = time.monotonic()
        payload = self._payload(message, candidates, history)
        try:
            body = await self._post_json(payload)
        except httpx.TimeoutException:
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.TIMEOUT,
                error_code="timeout",
            )
        except (httpx.HTTPError, ValueError):
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ERROR,
                error_code="invalid_or_unavailable_response",
            )
        except Exception:
            # Keep provider failures from breaking the original intent path.
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ERROR,
                error_code="provider_error",
            )

        answers = body.get("answers")
        if not isinstance(answers, dict):
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ABSTAINED,
                error_code="missing_answers",
            )
        answer = answers.get("intent", {})
        if not isinstance(answer, dict):
            answer = {}
        label = answer.get("choice")
        if not isinstance(label, str) or label not in candidates:
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ABSTAINED,
                error_code="missing_or_unknown_choice",
            )

        raw_confidence = answer.get("x_p_max", answer.get("confidence"))
        confidence: Optional[float] = None
        if isinstance(raw_confidence, (int, float)) and math.isfinite(raw_confidence):
            candidate_confidence = float(raw_confidence)
            if 0.0 <= candidate_confidence <= 1.0:
                confidence = candidate_confidence
        return DecisionResult(
            label=label,
            confidence=confidence,
            source=self.source,
            latency_ms=(time.monotonic() - started) * 1000,
            status=DecisionStatus.DECIDED,
        )


@dataclass(frozen=True)
class _LoadedGgufModel:
    model: Any
    inference_lock: threading.Lock


_GGUF_MODELS: Dict[Tuple[str, int, int, int], _LoadedGgufModel] = {}
_GGUF_MODELS_LOCK = threading.Lock()


def _load_gguf_model(
    model_path: Path,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
) -> _LoadedGgufModel:
    """Load each model/configuration once per process and serialize inference."""
    resolved_path = str(model_path.resolve())
    key = (resolved_path, n_ctx, n_threads, n_gpu_layers)
    with _GGUF_MODELS_LOCK:
        cached = _GGUF_MODELS.get(key)
        if cached is not None:
            return cached

        if not model_path.is_file():
            raise FileNotFoundError("model_file_missing")
        model_dir = model_path.parent
        for required_file in ("decider_config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (model_dir / required_file).is_file():
                raise FileNotFoundError("model_metadata_missing")

        # Keep heavyweight optional imports out of baseline startup. This helper
        # is the exact GGUF path used by the G4 benchmark and loads tokenizer
        # metadata from the model directory.
        from models.decide_gguf import GGUFDecider

        model = GGUFDecider(
            str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
        loaded = _LoadedGgufModel(model=model, inference_lock=threading.Lock())
        _GGUF_MODELS[key] = loaded
        return loaded


class MapikaGgufProvider:
    """In-process adapter for the Mapika Decider GGUF model."""

    source = "mapika-decider-2b-v11-q4-gguf"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
        model: Any = None,
    ):
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._injected_model = model
        self._injected_lock = threading.Lock()

    def _decide_sync(
        self,
        message: str,
        candidates: Mapping[str, str],
        history: Optional[Sequence[Mapping[str, str]]],
    ) -> Optional[Tuple[str, Optional[float]]]:
        state = f"用户本轮消息：{message}"
        if history:
            recent = history[-3:]
            state += "\n最近对话：" + "\n".join(
                f"{str(item.get('role', 'user'))}: {str(item.get('content', ''))}"
                for item in recent
            )

        option_to_label = {
            f"{label}｜{description}": label for label, description in candidates.items()
        }
        question = {
            "question": "根据用户本轮主要诉求，选择最匹配的一个客服意图。确实不属于任何候选时选择 other。",
            "options": list(option_to_label),
        }

        if self._injected_model is not None:
            loaded = _LoadedGgufModel(self._injected_model, self._injected_lock)
        else:
            loaded = _load_gguf_model(
                self.model_path,
                self.n_ctx,
                self.n_threads,
                self.n_gpu_layers,
            )
        with loaded.inference_lock:
            answers = loaded.model.decide(state, [question])
        answer = answers[0] if isinstance(answers, list) and answers else {}
        choice = answer.get("choice") if isinstance(answer, dict) else None
        label = option_to_label.get(choice) if isinstance(choice, str) else None
        if not isinstance(label, str) or label not in candidates:
            return None
        raw_confidence = answer.get("confidence") if isinstance(answer, dict) else None
        confidence = None
        if isinstance(raw_confidence, (int, float)) and math.isfinite(raw_confidence):
            value = float(raw_confidence)
            if 0.0 <= value <= 1.0:
                confidence = value
        return label, confidence

    async def decide(
        self,
        message: str,
        candidates: Mapping[str, str],
        history: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> DecisionResult:
        started = time.monotonic()
        if len(candidates) < 2:
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ABSTAINED,
                error_code="insufficient_candidates",
            )

        try:
            decision = await asyncio.to_thread(self._decide_sync, message, candidates, history)
        except FileNotFoundError as exc:
            error_code = str(exc) if str(exc) in {"model_file_missing", "model_metadata_missing"} else "model_file_missing"
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ERROR,
                error_code=error_code,
            )
        except ImportError:
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ERROR,
                error_code="backend_dependency_missing",
            )
        except Exception as exc:
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ERROR,
                error_code=type(exc).__name__,
            )

        if decision is None:
            return DecisionResult(
                label=None,
                confidence=None,
                source=self.source,
                latency_ms=(time.monotonic() - started) * 1000,
                status=DecisionStatus.ABSTAINED,
                error_code="missing_or_unknown_choice",
            )

        label, confidence = decision
        return DecisionResult(
            label=label,
            confidence=confidence,
            source=self.source,
            latency_ms=(time.monotonic() - started) * 1000,
            status=DecisionStatus.DECIDED,
        )


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def create_local_decision_provider() -> Optional[DecisionProvider]:
    """Build the configured local provider without eagerly loading a model."""
    provider_name = os.getenv("ECHOMIND_LOCAL_DECISION_PROVIDER", "mapika_gguf").strip().lower()
    if provider_name in {"mapika", "mapika_gguf"}:
        default_model = Path(__file__).resolve().parents[1] / "models" / "decider-2b-v11-Q4_K_M.gguf"
        model_path = os.getenv("ECHOMIND_DECIDER_MODEL_PATH", str(default_model)).strip()
        if not model_path:
            logger.warning("本地决策模型路径为空")
            return None
        return MapikaGgufProvider(
            model_path=model_path,
            n_ctx=_env_positive_int("ECHOMIND_DECIDER_N_CTX", 2048),
            n_threads=_env_positive_int("ECHOMIND_DECIDER_N_THREADS", 6),
            n_gpu_layers=0,
        )

    if provider_name == "openjev_http":
        base_url = os.getenv("ECHOMIND_OPENJEV_BASE_URL", "").strip()
        if not base_url:
            logger.warning("OpenJev HTTP 决策后端未配置 ECHOMIND_OPENJEV_BASE_URL")
            return None
        try:
            timeout_seconds = float(os.getenv("ECHOMIND_OPENJEV_TIMEOUT_SECONDS", "8"))
            return OpenJevProvider(
                base_url,
                timeout_seconds=timeout_seconds,
                api_key=os.getenv("ECHOMIND_OPENJEV_API_KEY"),
            )
        except ValueError:
            logger.warning("OpenJev HTTP 决策配置无效，local 决策将回退 baseline")
            return None

    logger.warning("未知本地决策提供方，回退 baseline: %s", provider_name)
    return None
