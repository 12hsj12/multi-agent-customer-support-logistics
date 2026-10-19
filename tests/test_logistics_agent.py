import asyncio
from collections import deque

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    EscalationAgent,
    GeneralAgent,
    LogisticsAgent,
    Request,
)
from agents.tools import mask_tracking_no, query_logistics, redact_entities
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.skill_loader import SkillManager


class FakeClient:
    def __init__(self):
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                raise AssertionError("a missing tracking number must be clarified before calling the model")

        self.messages = Messages()


class ToolUseBlock:
    type = "tool_use"

    def __init__(self, tracking_no):
        self.id = "toolu_logistics"
        self.name = "query_logistics"
        self.input = {"tracking_no": tracking_no}


class TextBlock:
    type = "text"
    text = "本地模拟记录显示物流信息停滞，不能据此认定真实丢件。"


class ToolClient:
    def __init__(self, tracking_no="MOCK-LOST-0002"):
        self.responses = [
            type("Response", (), {"content": [ToolUseBlock(tracking_no)]})(),
            type("Response", (), {"content": [TextBlock()]})(),
        ]

        class Messages:
            async def create(inner, **kwargs):
                return self.responses.pop(0)

        self.messages = Messages()


def make_logistics_request(**kwargs):
    values = {
        "message": "物流疑似丢件，运单号是 MOCK-LOST-0002",
        "user_id": "demo-user",
        "conv_id": "demo-conversation",
        "intent": IntentCategory.LOGISTICS_LOST,
        "intent_group": "query",
        "urgency": UrgencyLevel.LOW,
        "entities": {"tracking_no": ["MOCK-LOST-0002"]},
        "history": [
            {"role": "user", "content": "我的包裹是不是丢件了"},
            {"role": "assistant", "content": "请提供运单号"},
            {"role": "user", "content": "我又来问一下是不是丢件了"},
        ],
    }
    values.update(kwargs)
    return Request(**values)


def test_logistics_intents_and_tracking_number_extraction():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    cases = {
        "包裹疑似丢件": IntentCategory.LOGISTICS_LOST,
        "收到的商品破损": IntentCategory.LOGISTICS_DAMAGED,
        "物流显示已签收但没收到": IntentCategory.LOGISTICS_DISPUTE,
        "物流停滞多天": IntentCategory.LOGISTICS_DELAY,
        "请帮我修改地址": IntentCategory.LOGISTICS_ADDRESS,
        "快递什么时候到": IntentCategory.LOGISTICS,
    }

    for message, expected in cases.items():
        assert recognizer._pattern_recognize(message)["intent"] is expected
        assert recognizer._intent_group(expected) == "query"

    entities = recognizer._extract_entities("快递单号是 MOCK-TRANSIT-0001")
    assert entities["tracking_no"] == ["MOCK-TRANSIT-0001"]
    assert recognizer._extract_entities("包裹没有收到 ABC12345678")["tracking_no"] == []


def test_specialized_logistics_routes_to_logistics_and_broad_query_stays_general():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.LOGISTICS: [object()],
        AgentType.BILLING: [object()],
    }

    specialist = orchestrator._route_decision(make_logistics_request())
    ordinary = orchestrator._route_decision(make_logistics_request(
        intent=IntentCategory.LOGISTICS,
        message="快递什么时候到？",
        entities={},
        history=None,
    ))
    disputed_by_text = orchestrator._route_decision(make_logistics_request(
        intent=IntentCategory.LOGISTICS,
        message="物流延误，快递一直没更新",
        entities={},
        history=None,
    ))
    unknown = orchestrator._route_decision(make_logistics_request(
        intent=IntentCategory.OTHER,
        message="我不确定要咨询什么",
        entities={},
        history=None,
    ))

    assert specialist.primary_agent is AgentType.LOGISTICS
    assert ordinary.primary_agent is AgentType.GENERAL
    assert disputed_by_text.primary_agent is AgentType.LOGISTICS
    assert unknown.primary_agent is AgentType.GENERAL


def test_logistics_skill_loads_only_for_logistics_agent():
    from pathlib import Path

    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    manager = SkillManager(str(skills_dir))
    loaded = manager.load()
    logistics = next(skill for skill in loaded if skill.name == "物流纠纷受理规范")

    assert logistics.agents == ["logistics"]
    assert logistics.matches("我对物流签收有争议", "logistics") is True
    assert logistics.matches("我对物流签收有争议", "general") is False


def test_logistics_agent_asks_for_missing_tracking_number_without_calling_provider():
    client = FakeClient()
    agent = LogisticsAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_logistics_request(entities={})))

    assert response.success is True
    assert "提供运单号" in response.content
    assert response.tools_used == []
    assert client.calls == []


def test_local_logistics_fixture_is_deterministic_and_candidate_rule_is_configurable():
    # Environment restoration keeps this test safe when run outside pytest.
    import os

    keys = ("ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD", "ECHOMIND_LOGISTICS_MIN_FOLLOWUPS")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[keys[0]] = "7"
    os.environ[keys[1]] = "2"
    try:
        request = make_logistics_request()
        result = query_logistics(request, {"tracking_no": "MOCK-LOST-0002"})
        assert result["success"] is True
        assert result["source"] == "local_mock"
        assert result["days_without_update"] == 8
        assert result["escalation_candidate"] is True
        assert "真实" in result["data_notice"]

        not_enough_followups = query_logistics(
            make_logistics_request(history=[{"role": "user", "content": "物流怎么没更新"}]),
            {"tracking_no": "MOCK-LOST-0002"},
        )
        assert not_enough_followups["escalation_candidate"] is False

        missing = query_logistics(request, {"tracking_no": "MOCK-UNKNOWN-9999"})
        assert missing["success"] is False
        assert missing["code"] == "not_found"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_logistics_tool_trace_masks_tracking_number_and_sets_deterministic_escalation():
    import os

    keys = ("ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD", "ECHOMIND_LOGISTICS_MIN_FOLLOWUPS")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[keys[0]] = "7"
    os.environ[keys[1]] = "2"
    try:
        agent = LogisticsAgent(ToolClient(), "test-model")
        response = asyncio.run(agent.handle(make_logistics_request()))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert response.success is True
    assert response.escalate is True
    assert response.tool_traces[0]["escalation_candidate"] is True
    traced_number = response.tool_traces[0]["input"]["tracking_no"]
    assert traced_number.endswith("0002")
    assert traced_number != "MOCK-LOST-0002"


def test_handoff_summary_redacts_tracking_number():
    original = {"tracking_no": ["MOCK-LOST-0002"], "order_id": ["ORD12345"]}
    redacted = redact_entities(original)

    assert original["tracking_no"] == ["MOCK-LOST-0002"]
    assert redacted["tracking_no"] == [mask_tracking_no("MOCK-LOST-0002")]
    assert redacted["order_id"] == ["ORD12345"]


def test_logistics_failure_falls_back_to_general_agent():
    class BrokenLogisticsAgent:
        agent_type = AgentType.LOGISTICS
        stats = AgentStats()

        async def handle(self, req):
            return AgentResponse(
                agent_type=AgentType.LOGISTICS,
                content="模拟物流查询暂不可用",
                success=False,
            )

    class WorkingGeneralAgent:
        agent_type = AgentType.GENERAL
        stats = AgentStats()

        async def handle(self, req):
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="我无法查询模拟记录，可以先协助您核对运单号。",
                success=True,
            )

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [WorkingGeneralAgent()],
        AgentType.LOGISTICS: [BrokenLogisticsAgent()],
    }
    response = asyncio.run(orchestrator._execute(make_logistics_request(), AgentType.LOGISTICS))

    assert response.success is True
    assert response.agent_type is AgentType.GENERAL
    assert "核对运单号" in response.content


def test_evidence_candidate_runs_existing_escalation_agent():
    class CandidateLogisticsAgent:
        agent_type = AgentType.LOGISTICS
        stats = AgentStats()

        async def handle(self, req):
            return AgentResponse(
                agent_type=AgentType.LOGISTICS,
                content="本地模拟记录达到候选升级条件。",
                success=True,
                escalate=True,
                tools_used=["query_logistics"],
                tool_traces=[{
                    "tool_name": "query_logistics",
                    "input": {"tracking_no": mask_tracking_no("MOCK-LOST-0002")},
                    "escalation_candidate": True,
                    "escalation_reason": "mock record and follow-up threshold met",
                }],
            )

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [LogisticsAgent(FakeClient(), "test-model")],
        AgentType.LOGISTICS: [CandidateLogisticsAgent()],
        AgentType.ESCALATION: [EscalationAgent(FakeClient(), "test-model")],
    }
    orchestrator._recent_tool_traces = deque(maxlen=20)

    result = asyncio.run(orchestrator.run(make_logistics_request()))

    assert result.escalated is True
    assert result.agent_type is AgentType.ESCALATION
    assert result.agent_types == [AgentType.LOGISTICS, AgentType.ESCALATION]
    assert result.primary_agent is AgentType.LOGISTICS
    assert result.supporting_agents == [AgentType.ESCALATION]
    assert "local_mock evidence" in result.routing_reason
    assert "MOCK-LOST-0002" not in result.response


def test_chat_endpoint_executes_logistics_tool_and_exposes_request_trace():
    from types import SimpleNamespace

    from httpx import ASGITransport, AsyncClient
    from api import main as api_main
    from core.intent_recognizer import IntentResult
    from core.intent_recognizer import UrgencyLevel

    class FakeRecognizer:
        async def recognize(self, message, history=None):
            return IntentResult(
                intent=IntentCategory.LOGISTICS_DELAY,
                confidence=0.95,
                urgency=UrgencyLevel.LOW,
                intent_group="query",
                entities={"tracking_no": ["MOCK-TRANSIT-0001"]},
                reasoning="test fixture",
                latency_ms=0.1,
                source_scores={"pattern": 0.95},
            )

    class FakeMemory:
        def __init__(self):
            self.saved_messages = []

        async def get_context(self, user_id, conv_id, query=None):
            return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "")

        async def add_message(self, user_id, conv_id, role, content):
            self.saved_messages.append((role.value, content))

        async def update_profile(self, user_id, conv_id):
            return None

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._intent_recognizer = FakeRecognizer()
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(FakeClient(), "test-model")],
        AgentType.LOGISTICS: [LogisticsAgent(ToolClient("MOCK-TRANSIT-0001"), "test-model")],
        AgentType.ESCALATION: [EscalationAgent(FakeClient(), "test-model")],
    }
    orchestrator._recent_tool_traces = deque(maxlen=20)
    memory = FakeMemory()
    original_orchestrator, original_memory = api_main._orchestrator, api_main._memory
    api_main._orchestrator, api_main._memory = orchestrator, memory
    try:
        async def post_chat():
            async with AsyncClient(
                transport=ASGITransport(app=api_main.app),
                base_url="http://testserver",
            ) as client:
                return await client.post("/chat", json={
                    "message": "物流延误，运单号是 MOCK-TRANSIT-0001",
                    "user_id": "g2-demo-user",
                    "conv_id": "g2-demo-conversation",
                })

        http_response = asyncio.run(post_chat())
    finally:
        api_main._orchestrator, api_main._memory = original_orchestrator, original_memory

    assert http_response.status_code == 200
    response = http_response.json()
    trace = orchestrator.get_tool_trace(response["request_id"])
    assert response["intent"] == "logistics_delay"
    assert response["agent_type"] == "logistics"
    assert response["tools_used"] == ["query_logistics"]
    assert response["request_id"]
    assert trace is not None
    assert trace["tool_calls"][0]["input"]["tracking_no"].endswith("0001")
    assert trace["tool_calls"][0]["input"]["tracking_no"] != "MOCK-TRANSIT-0001"
    assert memory.saved_messages[0][0] == "user"
    assert memory.saved_messages[1][0] == "assistant"


def test_chat_endpoint_hands_off_evidence_candidate_and_keeps_trace():
    from types import SimpleNamespace

    from httpx import ASGITransport, AsyncClient
    from api import main as api_main
    from core.intent_recognizer import IntentResult

    class FakeRecognizer:
        async def recognize(self, message, history=None):
            return IntentResult(
                intent=IntentCategory.LOGISTICS_LOST,
                confidence=0.95,
                urgency=UrgencyLevel.LOW,
                intent_group="query",
                entities={"tracking_no": ["MOCK-LOST-0002"]},
                reasoning="test fixture",
                latency_ms=0.1,
                source_scores={"pattern": 0.95},
            )

    class FakeMemory:
        async def get_context(self, user_id, conv_id, query=None):
            return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "")

        async def add_message(self, user_id, conv_id, role, content):
            return None

        async def update_profile(self, user_id, conv_id):
            return None

    class CandidateLogisticsAgent:
        agent_type = AgentType.LOGISTICS
        stats = AgentStats()

        async def handle(self, req):
            return AgentResponse(
                agent_type=AgentType.LOGISTICS,
                content="本地模拟证据满足候选升级条件。",
                success=True,
                escalate=True,
                tools_used=["query_logistics"],
                tool_traces=[{
                    "tool_name": "query_logistics",
                    "input": {"tracking_no": mask_tracking_no("MOCK-LOST-0002")},
                    "escalation_candidate": True,
                    "escalation_reason": "mock record and follow-up threshold met",
                }],
            )

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._intent_recognizer = FakeRecognizer()
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(FakeClient(), "test-model")],
        AgentType.LOGISTICS: [CandidateLogisticsAgent()],
        AgentType.ESCALATION: [EscalationAgent(FakeClient(), "test-model")],
    }
    orchestrator._recent_tool_traces = deque(maxlen=20)
    original_orchestrator, original_memory = api_main._orchestrator, api_main._memory
    api_main._orchestrator, api_main._memory = orchestrator, FakeMemory()
    try:
        async def post_chat_and_get_trace():
            async with AsyncClient(
                transport=ASGITransport(app=api_main.app),
                base_url="http://testserver",
            ) as client:
                response = await client.post("/chat", json={
                    "message": "我的包裹疑似丢件，已经多次来询问",
                    "user_id": "g2-escalation-demo-user",
                    "conv_id": "g2-escalation-demo-conversation",
                })
                trace = await client.get(f"/trace/tool/{response.json()['request_id']}")
                return response, trace

        http_response, trace_response = asyncio.run(post_chat_and_get_trace())
    finally:
        api_main._orchestrator, api_main._memory = original_orchestrator, original_memory

    payload = http_response.json()
    trace_payload = trace_response.json()
    assert http_response.status_code == 200
    assert payload["escalated"] is True
    assert payload["agent_type"] == "escalation"
    assert payload["agent_types"] == ["logistics", "escalation"]
    assert payload["primary_agent"] == "logistics"
    assert "MOCK-LOST-0002" not in payload["response"]
    assert trace_response.status_code == 200
    assert trace_payload["found"] is True
    traced_number = trace_payload["trace"]["tool_calls"][0]["input"]["tracking_no"]
    assert traced_number.endswith("0002")
    assert traced_number != "MOCK-LOST-0002"
