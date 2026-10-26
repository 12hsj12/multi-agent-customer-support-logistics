"""Exercise the complete Dockerized FastAPI /chat path with real OpenJev."""
from __future__ import annotations

import asyncio
import json
import os

import httpx


async def _run() -> dict:
    base_url = os.getenv("ECHOMIND_CHAT_E2E_BASE_URL", "http://echomind:8000")
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/chat",
            json={
                "user_id": "synthetic-openjev-e2e",
                "message": "物流签收记录和实际收货情况不符，我没拿到包裹；同时我想申请退款。",
            },
        )
        response.raise_for_status()
        body = response.json()

    result = {
        "decision_source": body.get("intent_decision_source"),
        "decision_status": body.get("intent_decision_status"),
        "intent": body.get("intent"),
        "agent_type": body.get("agent_type"),
        "request_id_present": bool(body.get("request_id")),
        "latency_ms": body.get("latency_ms"),
        "decision_latency_ms": body.get("intent_decision_latency_ms"),
    }
    if result["decision_source"] != "openjev" or result["decision_status"] != "applied":
        raise SystemExit(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["request_id_present"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, sort_keys=True))
