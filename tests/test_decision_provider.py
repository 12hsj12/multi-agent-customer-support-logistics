import asyncio

import httpx

from core import decision_provider as decision_provider_module
from core.decision_provider import (
    DecisionResult,
    DecisionStatus,
    MapikaGgufProvider,
    OpenJevProvider,
    create_local_decision_provider,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


class StubOpenJevProvider(OpenJevProvider):
    def __init__(self, body=None, error=None):
        super().__init__("http://local-decider:8000")
        self.body = body
        self.error = error
        self.payload = None

    async def _post_json(self, payload):
        self.payload = payload
        if self.error:
            raise self.error
        return self.body


class FakeDecisionProvider:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def decide(self, **kwargs):
        self.calls += 1
        return self.result


def test_openjev_provider_sends_system_one_request_and_parses_choice():
    provider = StubOpenJevProvider(
        body={"answers": {"intent": {"choice": "billing", "x_p_max": 0.82}}}
    )

    result = asyncio.run(provider.decide(
        message="重复扣款了",
        candidates={"billing": "账单、扣款", "other": "其它"},
        history=[{"role": "user", "content": "上轮内容"}],
    ))

    assert result.status is DecisionStatus.DECIDED
    assert result.label == "billing"
    assert result.confidence == 0.82
    assert provider.payload["state"]["message"] == "重复扣款了"
    assert provider.payload["state"]["history"][0]["content"] == "上轮内容"
    assert provider.payload["questions"]["intent"]["criteria"]["other"] == "其它"


def test_openjev_http_adapter_uses_local_system_one_endpoint_and_optional_key():
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answers": {"intent": {"choice": "greeting"}}}

    class Client:
        def __init__(self, timeout):
            assert timeout == 2.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json, headers):
            calls.append((url, json, headers))
            return Response()

    original_client = decision_provider_module.httpx.AsyncClient
    decision_provider_module.httpx.AsyncClient = Client
    try:
        provider = OpenJevProvider("http://127.0.0.1:8002/", timeout_seconds=2.0, api_key="local")
        result = asyncio.run(provider.decide("你好", {"greeting": "问候"}))
    finally:
        decision_provider_module.httpx.AsyncClient = original_client

    assert result.status is DecisionStatus.DECIDED
    assert calls[0][0] == "http://127.0.0.1:8002/v1/systemone"
    assert calls[0][2]["Authorization"] == "Bearer local"


def test_openjev_provider_abstains_on_unknown_choice():
    provider = StubOpenJevProvider(body={"answers": {"intent": {"choice": "not_a_candidate"}}})

    result = asyncio.run(provider.decide("你好", {"greeting": "问候"}))

    assert result.status is DecisionStatus.ABSTAINED
    assert result.error_code == "missing_or_unknown_choice"


def test_openjev_provider_maps_timeout_to_status():
    provider = StubOpenJevProvider(error=httpx.TimeoutException("timeout"))

    result = asyncio.run(provider.decide("你好", {"greeting": "问候"}))

    assert result.status is DecisionStatus.TIMEOUT
    assert result.error_code == "timeout"


def test_mapika_gguf_provider_maps_choice_and_confidence_without_blocking_loop():
    class Model:
        def __init__(self):
            self.state = None
            self.questions = None

        def decide(self, state, questions):
            self.state = state
            self.questions = questions
            return [{"choice": questions[0]["options"][1], "confidence": 0.79}]

    model = Model()
    provider = MapikaGgufProvider("unused.gguf", model=model)
    result = asyncio.run(provider.decide(
        message="包裹收到时破损",
        candidates={"billing": "费用问题", "logistics_damaged": "物流破损"},
        history=[
            {"role": "user", "content": "旧消息"},
            {"role": "assistant", "content": "旧回复"},
            {"role": "user", "content": "最近追问"},
            {"role": "assistant", "content": "最近回复"},
        ],
    ))

    assert result.status is DecisionStatus.DECIDED
    assert result.label == "logistics_damaged"
    assert result.confidence == 0.79
    assert "包裹收到时破损" in model.state
    assert "旧消息" not in model.state
    assert "最近追问" in model.state


def test_mapika_gguf_provider_reports_model_failure_for_baseline_fallback():
    class BrokenModel:
        def decide(self, state, questions):
            raise RuntimeError("provider down")

    result = asyncio.run(MapikaGgufProvider("unused.gguf", model=BrokenModel()).decide(
        message="包裹破损",
        candidates={"logistics_damaged": "物流破损", "other": "其它"},
    ))

    assert result.status is DecisionStatus.ERROR
    assert result.error_code == "RuntimeError"


def test_local_provider_factory_is_lazy_and_defaults_to_mapika(monkeypatch):
    monkeypatch.setenv("ECHOMIND_LOCAL_DECISION_PROVIDER", "mapika_gguf")
    monkeypatch.setenv("ECHOMIND_DECIDER_MODEL_PATH", "models/decider-2b-v11-Q4_K_M.gguf")

    provider = create_local_decision_provider()

    assert isinstance(provider, MapikaGgufProvider)
    assert str(provider.model_path).endswith("decider-2b-v11-Q4_K_M.gguf")


def _make_recognizer(provider, llm_result, pattern_result):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._decision_backend = "local"
    recognizer._decision_provider = provider
    recognizer._local_trigger_confidence = 0.65
    recognizer._llm_priority_confidence = 0.85
    recognizer._embedding_enabled = False
    recognizer.threshold = 0.5
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0
    recognizer._cache_key = lambda message, history=None: message

    async def llm_recognize(message, history):
        return llm_result

    recognizer._llm_recognize = llm_recognize
    recognizer._pattern_recognize = lambda message: pattern_result
    recognizer._extract_entities = lambda message: {}
    recognizer._urgency = lambda message, intent: UrgencyLevel.LOW
    return recognizer


def test_confident_llm_keeps_priority_and_skips_local_provider():
    provider = FakeDecisionProvider(DecisionResult(
        label=IntentCategory.LOGISTICS_LOST.value,
        confidence=0.9,
        source="openjev",
        latency_ms=10,
        status=DecisionStatus.DECIDED,
    ))
    recognizer = _make_recognizer(
        provider,
        {"intent": IntentCategory.BILLING, "confidence": 0.95, "reasoning": "高置信"},
        {"intent": IntentCategory.LOGISTICS_LOST, "confidence": 0.75},
    )

    result = asyncio.run(recognizer.recognize("这笔账单有问题"))

    assert result.intent is IntentCategory.BILLING
    assert result.decision_source == "baseline"
    assert result.decision_status == "skipped"
    assert provider.calls == 0


def test_baseline_mode_never_calls_local_provider():
    provider = FakeDecisionProvider(DecisionResult(
        label=IntentCategory.LOGISTICS_LOST.value,
        confidence=0.9,
        source="openjev",
        latency_ms=10,
        status=DecisionStatus.DECIDED,
    ))
    recognizer = _make_recognizer(
        provider,
        {"intent": IntentCategory.QUERY, "confidence": 0.6, "reasoning": "低置信"},
        {"intent": IntentCategory.BILLING, "confidence": 0.75},
    )
    recognizer._decision_backend = "baseline"

    result = asyncio.run(recognizer.recognize("账单里的快递费不对"))

    assert result.decision_source == "baseline"
    assert result.decision_status == "disabled"
    assert provider.calls == 0


def test_low_confidence_conflict_uses_local_result():
    provider = FakeDecisionProvider(DecisionResult(
        label=IntentCategory.LOGISTICS_DAMAGED.value,
        confidence=0.81,
        source="openjev",
        latency_ms=120,
        status=DecisionStatus.DECIDED,
    ))
    recognizer = _make_recognizer(
        provider,
        {"intent": IntentCategory.QUERY, "confidence": 0.6, "reasoning": "低置信"},
        {"intent": IntentCategory.BILLING, "confidence": 0.75},
    )

    result = asyncio.run(recognizer.recognize("包裹破损而且账单也不对"))

    assert result.intent is IntentCategory.LOGISTICS_DAMAGED
    assert result.decision_source == "openjev"
    assert result.decision_status == "applied"
    assert result.decision_latency_ms == 120
    assert result.source_scores["openjev"] == 0.81
    assert provider.calls == 1


def test_local_provider_failure_preserves_baseline_result():
    provider = FakeDecisionProvider(DecisionResult(
        label=None,
        confidence=None,
        source="openjev",
        latency_ms=300,
        status=DecisionStatus.TIMEOUT,
        error_code="timeout",
    ))
    recognizer = _make_recognizer(
        provider,
        {"intent": IntentCategory.QUERY, "confidence": 0.6, "reasoning": "低置信"},
        {"intent": IntentCategory.BILLING, "confidence": 0.75},
    )

    result = asyncio.run(recognizer.recognize("账单里的快递费不对"))

    assert result.intent is IntentCategory.QUERY
    assert result.decision_source == "baseline"
    assert result.decision_status == "fallback"
    assert result.decision_latency_ms == 300
    assert provider.calls == 1
