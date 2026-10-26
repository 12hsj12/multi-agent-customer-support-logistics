"""Single real-model OpenJev oneshot smoke call; prints no prompt or secrets."""
from __future__ import annotations

import asyncio
import json
import os

from core.decision_provider import DecisionStatus, OpenJevApiProvider


async def _run() -> dict:
    model = os.getenv("ECHOMIND_OPENJEV_MODEL", "deepseek-flash")
    provider = OpenJevApiProvider(
        os.getenv("ECHOMIND_OPENJEV_BASE_URL", "http://openjev:3001"),
        model=model,
        timeout_seconds=float(os.getenv("ECHOMIND_OPENJEV_TIMEOUT_SECONDS", "30")),
    )
    result = await provider.decide(
        message="合成验证样例：本月账单被重复扣款，请判断主要客服意图。",
        candidates={
            "billing": "账单、费用、重复扣款或退款问题",
            "logistics": "快递、包裹、物流状态问题",
            "other": "不属于其它候选的客服问题",
        },
    )
    report = {
        "decision_source": result.source,
        "status": result.status.value,
        "model": model,
        "selected_candidate": result.label,
        "latency_ms": round(result.latency_ms, 2),
        "error_code": result.error_code,
    }
    if result.status is not DecisionStatus.DECIDED:
        raise SystemExit(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, sort_keys=True))
