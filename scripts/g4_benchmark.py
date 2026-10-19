"""Run the frozen G4 intent/routing benchmark without touching the legacy baseline."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(os.getenv("G4_DATASET_PATH", str(PROJECT_ROOT / "data/eval/g4_intents_v1.json")))
VARIANT = os.getenv("G4_VARIANT", "logistics")
OUTPUT_PATH = Path(os.getenv(
    "G4_OUTPUT_PATH",
    str(PROJECT_ROOT / "data/eval/g4_results" / f"reproduction_{VARIANT}.json"),
))
FINE_LOGISTICS = {
    "logistics_lost",
    "logistics_damaged",
    "logistics_dispute",
    "logistics_delay",
    "logistics_address",
}


def _percentile(values: Iterable[float], pct: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1))
    return round(ordered[index], 2)


def _metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": None,
            "macro_f1": None,
            "macro_f1_label_scope": "fixed labels present in this variant's gold rows",
            "per_class": {},
        }
    # Keep the label set fixed to the benchmark's gold classes. A prediction
    # outside that set remains an accuracy error and a miss for its gold class,
    # but must not create a new zero-support class that changes macro-F1's
    # denominator from one variant to another.
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
        support = sum(row["expected"] == label for row in rows)
        per_class[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4),
        "macro_f1": round(statistics.mean(item["f1"] for item in per_class.values()), 4),
        "macro_f1_label_scope": "fixed labels present in this variant's gold rows",
        "per_class": per_class,
    }


def _make_model_provider():
    from core.decision_provider import MapikaGgufProvider

    model_path = os.getenv(
        "ECHOMIND_DECIDER_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "decider-2b-v11-Q4_K_M.gguf"),
    )
    return MapikaGgufProvider(
        model_path=model_path,
        n_ctx=int(os.getenv("ECHOMIND_DECIDER_N_CTX", "2048")),
        n_threads=int(os.getenv("ECHOMIND_DECIDER_N_THREADS", "6")),
        n_gpu_layers=0,
    )


async def _run() -> Dict[str, Any]:
    if VARIANT not in {"baseline", "logistics", "local"}:
        raise SystemExit("G4_VARIANT must be baseline, logistics, or local")
    data_bytes = DATASET_PATH.read_bytes()
    dataset = json.loads(data_bytes)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.intent_recognizer import IntentRecognizer

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not configured in the container")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or None
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    recognizer_parameters = inspect.signature(IntentRecognizer).parameters
    kwargs: Dict[str, Any] = {}
    if "decision_backend" in recognizer_parameters:
        kwargs["decision_backend"] = "local" if VARIANT == "local" else "baseline"
    if VARIANT == "local":
        if "decision_provider" not in recognizer_parameters:
            raise SystemExit("The mounted source does not expose the G3 DecisionProvider injection point")
        kwargs["decision_provider"] = _make_model_provider()

    recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model_name, **kwargs)
    router = AgentOrchestrator(api_key=api_key, base_url=base_url, model=model_name)

    scored_common: List[Dict[str, str]] = []
    scored_fine: List[Dict[str, str]] = []
    route_rows: List[Dict[str, Any]] = []
    case_results: List[Dict[str, Any]] = []
    inference_latencies: List[float] = []
    decision_latencies: List[float] = []
    llm_failure_count = 0
    status_counts: Dict[str, int] = {}

    for case in dataset["cases"]:
        result = await recognizer.recognize(case["message"])
        predicted = result.intent.value
        predicted_common = "logistics" if predicted in FINE_LOGISTICS else predicted
        inference_latencies.append(float(result.latency_ms))
        llm_failed = "LLM" in str(getattr(result, "reasoning", "")) and "失败" in str(getattr(result, "reasoning", ""))
        llm_failure_count += int(llm_failed)
        status = str(getattr(result, "decision_status", "not_available"))
        status_counts[status] = status_counts.get(status, 0) + 1
        decision_latency = float(getattr(result, "decision_latency_ms", 0.0) or 0.0)
        if decision_latency > 0:
            decision_latencies.append(decision_latency)

        if case.get("score_intent", True):
            scored_common.append({"expected": case["common_expected_intent"], "predicted": predicted_common})
            if case["expected_intent"] in FINE_LOGISTICS:
                scored_fine.append({"expected": case["expected_intent"], "predicted": predicted})

        route_request = Request(
            message=case["message"],
            user_id="g4-benchmark",
            conv_id=f"g4-{case['id']}",
            intent=result.intent,
            intent_group=result.intent_group,
            urgency=result.urgency,
            entities=result.entities,
            intent_confidence=result.confidence,
        )
        route = router._route_decision(route_request)
        primary_agent = getattr(route, "primary_agent", route)
        supporting_agents = list(getattr(route, "supporting_agents", []) or [])
        actual_agents = [agent.value for agent in [primary_agent, *supporting_agents]]
        expected_agents = case.get("expected_agents_by_variant", {}).get(VARIANT)
        if expected_agents:
            route_rows.append({
                "id": case["id"],
                "expected_agents": expected_agents,
                "actual_agents": actual_agents,
                "primary_correct": primary_agent.value == expected_agents[0],
                "required_agents_present": all(agent in actual_agents for agent in expected_agents),
            })

        case_results.append({
            "id": case["id"],
            "expected_intent": case["expected_intent"],
            "common_expected_intent": case["common_expected_intent"],
            "predicted_intent": predicted,
            "predicted_common_intent": predicted_common,
            "score_intent": bool(case.get("score_intent", True)),
            "confidence": round(float(result.confidence), 4),
            "intent_latency_ms": round(float(result.latency_ms), 2),
            "decision_source": str(getattr(result, "decision_source", "baseline")),
            "decision_status": status,
            "decision_latency_ms": round(decision_latency, 2),
            "llm_failed": llm_failed,
            "route_primary": primary_agent.value,
            "route_agents": actual_agents,
            "route_reason": str(getattr(route, "reason", "")),
        })

    logistics_tool_report: Dict[str, Any] = {"applicable": False}
    try:
        from agents.tools import query_logistics
        from core.intent_recognizer import IntentCategory

        previous_days = os.environ.get("ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD")
        previous_followups = os.environ.get("ECHOMIND_LOGISTICS_MIN_FOLLOWUPS")
        os.environ["ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD"] = "7"
        os.environ["ECHOMIND_LOGISTICS_MIN_FOLLOWUPS"] = "2"
        tool_rows = []
        try:
            for case in dataset.get("logistics_tool_cases", []):
                request = Request(
                    message=f"G4 synthetic tool case {case['id']}",
                    user_id="g4-benchmark",
                    conv_id=f"g4-tool-{case['id']}",
                    intent=IntentCategory(case["intent"]),
                    history=case.get("history", []),
                    entities={},
                )
                args = {"tracking_no": case["tracking_no"]} if case.get("tracking_no") else {}
                output = query_logistics(request, args)
                tool_rows.append({
                    "id": case["id"],
                    "expected_success": case["expected_success"],
                    "actual_success": bool(output.get("success")),
                    "expected_escalation_candidate": case["expected_escalation_candidate"],
                    "actual_escalation_candidate": output.get("escalation_candidate"),
                    "code": output.get("code"),
                    "source": output.get("source"),
                })
        finally:
            if previous_days is None:
                os.environ.pop("ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD", None)
            else:
                os.environ["ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD"] = previous_days
            if previous_followups is None:
                os.environ.pop("ECHOMIND_LOGISTICS_MIN_FOLLOWUPS", None)
            else:
                os.environ["ECHOMIND_LOGISTICS_MIN_FOLLOWUPS"] = previous_followups

        candidate_rows = [row for row in tool_rows if row["expected_escalation_candidate"] is not None]
        tp = sum(row["expected_escalation_candidate"] is True and row["actual_escalation_candidate"] is True for row in candidate_rows)
        fp = sum(row["expected_escalation_candidate"] is False and row["actual_escalation_candidate"] is True for row in candidate_rows)
        fn = sum(row["expected_escalation_candidate"] is True and row["actual_escalation_candidate"] is False for row in candidate_rows)
        tool_successes = sum(row["expected_success"] == row["actual_success"] for row in tool_rows)
        logistics_tool_report = {
            "applicable": True,
            "tool_cases": len(tool_rows),
            "tool_success_expectation_accuracy": round(tool_successes / len(tool_rows), 4) if tool_rows else None,
            "escalation_candidate_cases": len(candidate_rows),
            "escalation_candidate_accuracy": round(sum(row["expected_escalation_candidate"] == row["actual_escalation_candidate"] for row in candidate_rows) / len(candidate_rows), 4) if candidate_rows else None,
            "escalation_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "escalation_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_positive_count": fp,
            "cases": tool_rows,
        }
    except (ImportError, AttributeError, ValueError):
        pass

    report = {
        "schema_version": "g4-benchmark-report-v1",
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "variant": VARIANT,
        "source_revision": os.getenv("G4_SOURCE_REVISION", "unspecified"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "llm_model": model_name,
            "decision_backend": "mapika_gguf_provider" if VARIANT == "local" else "baseline",
            "local_model": "Mapika/decider-2b-GGUF:decider-2b-v11-Q4_K_M.gguf" if VARIANT == "local" else None,
            "decider_ai_version": None,
            "llama_cpp_python_version": "0.3.35" if VARIANT == "local" else None,
            "local_provider_execution": "production MapikaGgufProvider; in-process GGUFDecider.decide with the bundled tokenizer.json and Mapika prompt format" if VARIANT == "local" else None,
            "sampling_note": "LLM temperature is fixed in source at 0.1; provider has no seed parameter; dataset and request order are fixed",
            "api_key_present": bool(api_key),
        },
        "source_hashes": {
            relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "core/intent_recognizer.py",
                "core/decision_provider.py",
                "agents/agent_orchestrator.py",
                "agents/tools.py",
                "scripts/g4_benchmark.py",
            )
            if (PROJECT_ROOT / relative).is_file()
        },
        "sample_distribution": {
            label: sum(case.get("score_intent", True) and case["common_expected_intent"] == label for case in dataset["cases"])
            for label in sorted({case["common_expected_intent"] for case in dataset["cases"] if case.get("score_intent", True)})
        },
        "metrics": {
            "common_old_label_space": _metrics(scored_common),
            "fine_logistics_label_space": {
                "supported_by_variant": VARIANT != "baseline",
                "scope": "five new logistics labels only",
                **_metrics(scored_fine if VARIANT != "baseline" else []),
            },
            "routing": {
                "total": len(route_rows),
                "primary_accuracy": round(sum(row["primary_correct"] for row in route_rows) / len(route_rows), 4) if route_rows else None,
                "required_agents_coverage": round(sum(row["required_agents_present"] for row in route_rows) / len(route_rows), 4) if route_rows else None,
                "cases": route_rows,
            },
            "latency_ms": {
                "intent_p50": _percentile(inference_latencies, 0.50),
                "intent_p95": _percentile(inference_latencies, 0.95),
                "local_decision_p50": _percentile(decision_latencies, 0.50),
                "local_decision_p95": _percentile(decision_latencies, 0.95),
            },
            "decision_status_counts": status_counts,
            "llm_failure_count": llm_failure_count,
            "local_provider_call_count": sum(count for status, count in status_counts.items() if status in {"applied", "fallback"}),
            "local_provider_fallback_rate": (
                round(status_counts.get("fallback", 0) / sum(count for status, count in status_counts.items() if status in {"applied", "fallback"}), 4)
                if sum(count for status, count in status_counts.items() if status in {"applied", "fallback"}) else None
            ),
        },
        "logistics_tool_and_escalation": logistics_tool_report,
        "cases": case_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = asyncio.run(_run())
    print(json.dumps({
        "variant": result["variant"],
        "dataset_sha256": result["dataset_sha256"],
        "report": str(OUTPUT_PATH),
        "metrics": {
            "common": result["metrics"]["common_old_label_space"],
            "fine_logistics": result["metrics"]["fine_logistics_label_space"],
            "routing": {
                key: value for key, value in result["metrics"]["routing"].items() if key != "cases"
            },
            "latency_ms": result["metrics"]["latency_ms"],
            "decision_status_counts": result["metrics"]["decision_status_counts"],
            "llm_failure_count": result["metrics"]["llm_failure_count"],
            "logistics_tool_and_escalation": {
                key: value for key, value in result["logistics_tool_and_escalation"].items() if key != "cases"
            },
        },
    }, ensure_ascii=False))
