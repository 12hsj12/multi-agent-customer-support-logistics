"""Exercise the real OpenJev service and baseline fallback in Compose E2E."""
from __future__ import annotations

import asyncio
import json
import os

from core.decision_provider import OpenJevApiProvider
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


def _recognizer(message: str) -> IntentRecognizer:
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._decision_backend = "local"
    recognizer._decision_provider = OpenJevApiProvider(
        os.getenv("ECHOMIND_OPENJEV_BASE_URL", "http://openjev:3001"),
        model=os.getenv("ECHOMIND_OPENJEV_MODEL", "e2e-mock"),
        timeout_seconds=float(os.getenv("ECHOMIND_OPENJEV_TIMEOUT_SECONDS", "10")),
    )
    recognizer._local_trigger_confidence = 0.65
    recognizer._llm_priority_confidence = 0.85
    recognizer._embedding_enabled = False
    recognizer.threshold = 0.5
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0
    recognizer._cache_key = lambda current, history=None: current

    async def llm_recognize(current, history):
        return {
            "intent": IntentCategory.BILLING,
            "confidence": 0.60,
            "reasoning": "E2E synthetic baseline vote",
        }

    recognizer._llm_recognize = llm_recognize
    recognizer._pattern_recognize = lambda current: {
        "intent": IntentCategory.BILLING,
        "confidence": 0.60,
    }
    recognizer._extract_entities = lambda current: {}
    recognizer._urgency = lambda current, intent: UrgencyLevel.LOW
    return recognizer


async def _run():
    recognizer = _recognizer("合成 E2E：请协助处理重复扣款")
    assert recognizer._decision_provider.model == "e2e-mock"
    applied = await recognizer.recognize(
        "合成 E2E：请协助处理重复扣款"
    )
    assert applied.decision_source == "openjev", applied
    assert applied.decision_status == "applied", applied
    assert applied.intent is IntentCategory.BILLING, applied

    fallback = await _recognizer(
        "合成 E2E：__mock_unknown_candidate__ 请协助处理重复扣款"
    ).recognize("合成 E2E：__mock_unknown_candidate__ 请协助处理重复扣款")
    assert fallback.decision_source == "baseline", fallback
    assert fallback.decision_status == "fallback", fallback
    assert fallback.intent is IntentCategory.BILLING, fallback

    return {
        "openjev": {
            "decision_source": applied.decision_source,
            "decision_status": applied.decision_status,
        },
        "fallback": {
            "decision_source": fallback.decision_source,
            "decision_status": fallback.decision_status,
        },
        "model_endpoint": "deterministic local OpenAI-compatible test stub",
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, sort_keys=True))
