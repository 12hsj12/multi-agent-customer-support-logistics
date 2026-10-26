import asyncio

import httpx

from core import decision_provider as decision_provider_module
from core.decision_provider import (
    DecisionResult,
    DecisionStatus,
    MapikaGgufProvider,
    OfficialJevProvider,
    OpenJevApiProvider,
    create_local_decision_provider,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


class StubOpenJevProvider(OpenJevApiProvider):
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


def test_openjev_provider_sends_oneshot_request_and_parses_choice():
    provider = StubOpenJevProvider(
        body={"answers": {"intent": {"type": "choice", "choice": "billing", "confidence": 0.82}}}
    )

    result = asyncio.run(provider.decide(
        message="重复扣款了",
        candidates={"billing": "账单、扣款", "other": "其它"},
        history=[{"role": "user", "content": "上轮内容"}],
    ))

    assert result.status is DecisionStatus.DECIDED
    assert result.label == "billing"
    assert result.confidence == 0.82
    assert provider.payload["model"] == "deepseek-flash"
    assert provider.payload["mode"] == "oneshot"
    assert provider.payload["state"]["message"] == "重复扣款了"
    assert provider.payload["state"]["history"][0]["content"] == "上轮内容"
    assert provider.payload["questions"]["intent"]["criteria"]["other"] == "其它"


def test_openjev_provider_can_be_configured_for_decider_forwarding():
    provider = OpenJevApiProvider(
        "http://local-decider:3001", model="deepseek-flash", mode="decider",
    )
    payload = provider._payload("账单重复扣款", {"billing": "账单", "other": "其它"}, None)

    assert payload["mode"] == "decider"
    assert payload["model"] == "deepseek-flash"


def test_openjev_provider_rejects_unsupported_mode():
    try:
        OpenJevApiProvider("http://openjev:3001", mode="parallel")
    except ValueError as exc:
        assert "oneshot or decider" in str(exc)
    else:
        raise AssertionError("unsupported OpenJev mode should be rejected")


def test_openjev_http_adapter_posts_api_evaluate_without_authentication():
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answers": {"intent": {"type": "choice", "choice": "greeting"}}}

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
        provider = OpenJevApiProvider("http://127.0.0.1:8002/", timeout_seconds=2.0)
        result = asyncio.run(provider.decide("你好", {"greeting": "问候", "other": "其它"}))
    finally:
        decision_provider_module.httpx.AsyncClient = original_client

    assert result.status is DecisionStatus.DECIDED
    assert calls[0][0] == "http://127.0.0.1:8002/api/evaluate"
    assert calls[0][1]["mode"] == "oneshot"
    assert calls[0][1]["model"] == "deepseek-flash"
    assert calls[0][1]["temperature"] == 0
    assert calls[0][1]["questions"]["intent"]["type"] == "choice"
    assert "Authorization" not in calls[0][2]


def test_official_jev_provider_uses_model_state_questions_and_bearer_key(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "jev-latest",
                "answers": {"intent": {"type": "choice", "choice": "billing"}},
                "usage": {"input_tokens": 12, "output_tokens": 2},
            }

    class Client:
        def __init__(self, timeout):
            assert timeout == 2.5

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json, headers):
            calls.append((url, json, headers))
            return Response()

    monkeypatch.setattr(decision_provider_module.httpx, "AsyncClient", Client)
    provider = OfficialJevProvider(
        "https://api.typesafe.ai/", model="jev-latest", timeout_seconds=2.5,
        api_key="mock-contract-key",
    )
    result = asyncio.run(provider.decide(
        "重复扣款了",
        {"billing": "账单、扣款", "other": "其它"},
    ))

    assert result.status is DecisionStatus.DECIDED
    assert result.source == "official_jev"
    assert result.label == "billing"
    assert calls[0][0] == "https://api.typesafe.ai/v1/systemone"
    assert calls[0][1]["model"] == "jev-latest"
    assert calls[0][1]["state"]["message"] == "重复扣款了"
    assert calls[0][1]["questions"]["intent"]["type"] == "choice"
    assert calls[0][2]["Authorization"] == "Bearer mock-contract-key"


def test_official_jev_factory_stays_disabled_without_api_key(monkeypatch):
    monkeypatch.setenv("ECHOMIND_LOCAL_DECISION_PROVIDER", "official_jev")
    monkeypatch.delenv("ECHOMIND_OFFICIAL_JEV_API_KEY", raising=False)

    assert create_local_decision_provider() is None


def test_openjev_provider_abstains_on_unknown_choice():
    provider = StubOpenJevProvider(body={"answers": {"intent": {
        "type": "choice", "choice": "not_a_candidate",
    }}})

    result = asyncio.run(provider.decide("你好", {"greeting": "问候", "other": "其它"}))

    assert result.status is DecisionStatus.ABSTAINED
    assert result.error_code == "missing_or_unknown_choice"


def test_openjev_provider_maps_timeout_to_status():
    provider = StubOpenJevProvider(error=httpx.TimeoutException("timeout"))

    result = asyncio.run(provider.decide("你好", {"greeting": "问候", "other": "其它"}))

    assert result.status is DecisionStatus.TIMEOUT
    assert result.error_code == "timeout"


def test_openjev_provider_maps_http_failure_without_leaking_response_body():
    request = httpx.Request("POST", "http://openjev:3001/api/evaluate")
    response = httpx.Response(503, request=request, text="contains-no-secret")
    provider = StubOpenJevProvider(
        error=httpx.HTTPStatusError("service unavailable", request=request, response=response)
    )

    result = asyncio.run(provider.decide("你好", {"greeting": "问候", "other": "其它"}))

    assert result.status is DecisionStatus.ERROR
    assert result.error_code == "invalid_or_unavailable_response"


def test_openjev_provider_rejects_wrong_answer_type_and_ignores_invalid_confidence():
    invalid_type = StubOpenJevProvider(body={"answers": {"intent": {
        "type": "noul", "choice": "greeting",
    }}})
    invalid_confidence = StubOpenJevProvider(body={"answers": {"intent": {
        "type": "choice", "choice": "greeting", "confidence": float("nan"),
    }}})

    type_result = asyncio.run(invalid_type.decide(
        "你好", {"greeting": "问候", "other": "其它"},
    ))
    confidence_result = asyncio.run(invalid_confidence.decide(
        "你好", {"greeting": "问候", "other": "其它"},
    ))

    assert type_result.status is DecisionStatus.ABSTAINED
    assert type_result.error_code == "missing_or_invalid_answer_type"
    assert confidence_result.status is DecisionStatus.DECIDED
    assert confidence_result.confidence is None


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


def test_local_provider_factory_selects_openjev_http(monkeypatch):
    monkeypatch.setenv("ECHOMIND_LOCAL_DECISION_PROVIDER", "openjev_http")
    monkeypatch.setenv("ECHOMIND_OPENJEV_BASE_URL", "http://openjev:3001")
    monkeypatch.setenv("ECHOMIND_OPENJEV_MODEL", "deepseek-flash")

    provider = create_local_decision_provider()

    assert isinstance(provider, OpenJevApiProvider)
    assert provider.base_url == "http://openjev:3001"
    assert provider.model == "deepseek-flash"
    assert provider.mode == "oneshot"


def test_local_provider_factory_can_select_openjev_decider_mode(monkeypatch):
    monkeypatch.setenv("ECHOMIND_LOCAL_DECISION_PROVIDER", "openjev_http")
    monkeypatch.setenv("ECHOMIND_OPENJEV_MODE", "decider")
    monkeypatch.setenv("ECHOMIND_OPENJEV_MODEL", "deepseek-flash")

    provider = create_local_decision_provider()

    assert isinstance(provider, OpenJevApiProvider)
    assert provider.mode == "decider"


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
