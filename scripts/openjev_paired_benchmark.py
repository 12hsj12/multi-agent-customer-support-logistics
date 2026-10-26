"""Paired baseline/OpenJev run over the frozen G4 intent dataset.

Each run writes a new report under data/eval/g5_results and refuses to replace
an existing report. The raw API key and endpoint values are never serialized.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.getenv("G5_DATASET_PATH", str(ROOT / "data/eval/g4_intents_v1.json")))
FINE_LOGISTICS = {
    "logistics_lost", "logistics_damaged", "logistics_dispute",
    "logistics_delay", "logistics_address",
}


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "core/decision_provider.py",
        "core/intent_recognizer.py",
        "scripts/openjev_paired_benchmark.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 2)


def _metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"total": 0, "correct": 0, "accuracy": None, "macro_f1": None, "per_class": {}}
    labels = sorted({row["expected"] for row in rows})
    correct = sum(row["expected"] == row["predicted"] for row in rows)
    per_class: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        tp = sum(row["expected"] == label and row["predicted"] == label for row in rows)
        fp = sum(row["expected"] != label and row["predicted"] == label for row in rows)
        fn = sum(row["expected"] == label and row["predicted"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(row["expected"] == label for row in rows),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4),
        "macro_f1": round(statistics.mean(row["f1"] for row in per_class.values()), 4),
        "per_class": per_class,
    }


class CountingProvider:
    """Count app-to-OpenJev decisions without recording customer text."""

    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    async def decide(self, **kwargs):
        self.calls += 1
        return await self.provider.decide(**kwargs)


class _LlmFailureCounter(logging.Handler):
    """Count recognizer parse/API failures without retaining message text."""

    def __init__(self, active_arm: List[str]):
        super().__init__(level=logging.WARNING)
        self.active_arm = active_arm
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "core.intent_recognizer" and record.getMessage().startswith("LLM 识别失败:"):
            self.counts[self.active_arm[0]] += 1


async def _run() -> Dict[str, Any]:
    dataset_bytes = DATASET.read_bytes()
    dataset = json.loads(dataset_bytes)
    cases = dataset["cases"]
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-"):
        raise SystemExit("ANTHROPIC_API_KEY is not configured for the baseline recognizer")

    from core.decision_provider import OpenJevApiProvider
    from core.intent_recognizer import IntentRecognizer

    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL") or None
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    openjev_model = os.getenv("OPENJEV_MODEL", "deepseek-flash")
    baseline = IntentRecognizer(
        api_key=api_key,
        base_url=anthropic_base_url,
        model=model_name,
        decision_backend="baseline",
    )
    counter = CountingProvider(OpenJevApiProvider(
        base_url=os.getenv("ECHOMIND_OPENJEV_BASE_URL", "http://openjev:3001"),
        model=openjev_model,
        timeout_seconds=float(os.getenv("ECHOMIND_OPENJEV_TIMEOUT_SECONDS", "8")),
    ))
    enhanced = IntentRecognizer(
        api_key=api_key,
        base_url=anthropic_base_url,
        model=model_name,
        decision_backend="local",
        decision_provider=counter,
    )

    active_arm = ["baseline"]
    failure_counter = _LlmFailureCounter(active_arm)
    logging.getLogger("core.intent_recognizer").addHandler(failure_counter)

    latencies: Dict[str, List[float]] = {"baseline": [], "openjev": []}
    decision_latencies: List[float] = []
    openjev_statuses: Counter[str] = Counter()
    per_case: List[Dict[str, Any]] = []

    try:
        for case in cases:
            active_arm[0] = "baseline"
            base_result = await baseline.recognize(case["message"])
            active_arm[0] = "openjev_enhanced"
            enhanced_result = await enhanced.recognize(case["message"])
            base_label = base_result.intent.value
            enhanced_label = enhanced_result.intent.value
            latencies["baseline"].append(float(base_result.latency_ms))
            latencies["openjev"].append(float(enhanced_result.latency_ms))
            openjev_statuses[str(enhanced_result.decision_status)] += 1
            if enhanced_result.decision_latency_ms > 0:
                decision_latencies.append(float(enhanced_result.decision_latency_ms))
            per_case.append({
                "id": case["id"],
                "expected_intent": case["expected_intent"],
                "baseline_intent": base_label,
                "openjev_intent": enhanced_label,
                "baseline_latency_ms": round(float(base_result.latency_ms), 2),
                "openjev_latency_ms": round(float(enhanced_result.latency_ms), 2),
                "decision_source": enhanced_result.decision_source,
                "decision_status": enhanced_result.decision_status,
                "decision_latency_ms": round(float(enhanced_result.decision_latency_ms), 2),
            })
    finally:
        logging.getLogger("core.intent_recognizer").removeHandler(failure_counter)

    # Build explicit scopes again: the common-label set collapses the five new
    # logistics labels to the legacy logistics category before scoring.
    common_labels = [
        (case, row) for case, row in zip(cases, per_case) if case.get("score_intent", True)
    ]
    common_baseline_metrics = _metrics([
        {"expected": case["common_expected_intent"],
         "predicted": "logistics" if row["baseline_intent"] in FINE_LOGISTICS else row["baseline_intent"]}
        for case, row in common_labels
    ])
    common_openjev_metrics = _metrics([
        {"expected": case["common_expected_intent"],
         "predicted": "logistics" if row["openjev_intent"] in FINE_LOGISTICS else row["openjev_intent"]}
        for case, row in common_labels
    ])
    fine_cases = [
        (case, row) for case, row in zip(cases, per_case)
        if case.get("score_intent", True) and case["expected_intent"] in FINE_LOGISTICS
    ]
    fine_baseline_metrics = _metrics([
        {"expected": case["expected_intent"], "predicted": row["baseline_intent"]}
        for case, row in fine_cases
    ])
    fine_openjev_metrics = _metrics([
        {"expected": case["expected_intent"], "predicted": row["openjev_intent"]}
        for case, row in fine_cases
    ])

    applied = sum(row["decision_source"] == "openjev" for row in per_case)
    report = {
        "schema_version": "openjev-paired-benchmark-v1",
        "dataset_id": dataset.get("dataset_id"),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "source_revision": os.getenv("G5_SOURCE_REVISION", "working-tree"),
        "source_sha256": _source_fingerprint(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(cases),
        "class_distribution": dict(sorted(Counter(case["expected_intent"] for case in cases).items())),
        "configuration": {
            "baseline_backend": "three-way fusion",
            "enhanced_backend": "OpenJev /api/evaluate mode=oneshot",
            "llm_model": model_name,
            "openjev_model": openjev_model,
            "openjev_timeout_seconds": float(os.getenv("ECHOMIND_OPENJEV_TIMEOUT_SECONDS", "8")),
            "decision_trigger_confidence": float(os.getenv("ECHOMIND_DECISION_TRIGGER_CONFIDENCE", "0.65")),
            "llm_priority_confidence": float(os.getenv("ECHOMIND_LLM_PRIORITY_CONFIDENCE", "0.85")),
        },
        "metrics": {
            "common_labels": {
                "baseline": common_baseline_metrics,
                "openjev": common_openjev_metrics,
            },
            "fine_logistics_labels": {
                "baseline": fine_baseline_metrics,
                "openjev": fine_openjev_metrics,
            },
            "latency_ms": {
                "baseline_p50": _percentile(latencies["baseline"], 0.50),
                "baseline_p95": _percentile(latencies["baseline"], 0.95),
                "openjev_p50": _percentile(latencies["openjev"], 0.50),
                "openjev_p95": _percentile(latencies["openjev"], 0.95),
                "openjev_decision_p50": _percentile(decision_latencies, 0.50),
                "openjev_decision_p95": _percentile(decision_latencies, 0.95),
            },
            "openjev_invocations": counter.calls,
            "openjev_applied_decisions": applied,
            "openjev_fallback_or_abstain": sum(value for key, value in openjev_statuses.items() if key == "fallback"),
            "openjev_status_counts": dict(sorted(openjev_statuses.items())),
            "intent_llm_calls_per_arm": len(cases),
            "intent_llm_failure_events": dict(sorted(failure_counter.counts.items())),
            "valid_model_comparison": applied > 0,
        },
        "cases": per_case,
    }
    return report


def main() -> None:
    report = asyncio.run(_run())
    output_dir = ROOT / "data/eval/g5_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(os.getenv("G5_OUTPUT_PATH", str(output_dir / f"openjev_paired_{timestamp}.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "sample_count": report["sample_count"],
        "openjev_invocations": report["metrics"]["openjev_invocations"],
        "openjev_applied_decisions": report["metrics"]["openjev_applied_decisions"],
        "valid_model_comparison": report["metrics"]["valid_model_comparison"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
