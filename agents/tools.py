"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 技术排障建议
  - 账单字段核验
  - 人工升级摘要
  - 共享知识库 RAG

订单查询、退款执行、账单修改等需要真实业务系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from core.intent_recognizer import IntentCategory

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def mask_tracking_no(value: Any) -> str:
    """在工具轨迹和交接信息中隐藏运单号，只保留末四位。"""
    tracking_no = str(value or "").strip()
    if not tracking_no:
        return ""
    if len(tracking_no) <= 4:
        return "*" * len(tracking_no)
    return "*" * (len(tracking_no) - 4) + tracking_no[-4:]


def redact_entities(entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """复制并脱敏结构化实体，避免交接摘要暴露完整运单号。"""
    redacted = dict(entities or {})
    if "tracking_no" in redacted:
        values = redacted["tracking_no"]
        if isinstance(values, list):
            redacted["tracking_no"] = [mask_tracking_no(value) for value in values]
        elif values:
            redacted["tracking_no"] = mask_tracking_no(values)
    return redacted


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": redact_entities(req.entities),
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent == "order_status":
        fields = ["订单号或下单时间"]
    elif intent == "logistics":
        fields = ["运单号（查询物流轨迹时）或订单号"]
    elif intent in {
        "logistics_lost",
        "logistics_damaged",
        "logistics_dispute",
        "logistics_delay",
        "logistics_address",
    }:
        fields = ["运单号"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": redact_entities(req.entities),
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给人工客服的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工客服继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": redact_entities(req.entities),
        "sensitive_data_required": False,
    }


def _recent_logistics_followups(
    history: Optional[List[Dict[str, str]]],
    intent: Optional[IntentCategory],
) -> int:
    """统计最近五条历史消息中的同类物流用户追问（时间频率代理指标）。"""
    issue_keywords = {
        IntentCategory.LOGISTICS_LOST: ("丢件", "丢了", "丢失", "不见", "找不到", "没收到"),
        IntentCategory.LOGISTICS_DAMAGED: ("破损", "损坏", "错发", "少件", "发错"),
        IntentCategory.LOGISTICS_DISPUTE: ("签收争议", "已签收", "代签", "误签", "没收到"),
        IntentCategory.LOGISTICS_DELAY: ("延误", "停滞", "没更新", "未更新", "超时"),
        IntentCategory.LOGISTICS_ADDRESS: ("地址", "改地址", "修改地址", "更改地址"),
    }
    keywords = issue_keywords.get(intent, ("物流", "快递", "运单", "配送", "签收", "丢件", "破损", "延误", "地址"))
    return sum(
        1
        for item in (history or [])[-5:]
        if str(item.get("role", "")).lower() == "user"
        and any(keyword in str(item.get("content", "")).lower() for keyword in keywords)
    )


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def query_logistics(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """查询仓库内固定的演示运单，不访问任何真实快递平台。"""
    tracking_no = str(args.get("tracking_no") or "").strip()
    if not tracking_no:
        return {"success": False, "code": "missing_tracking_no", "error": "请先提供运单号", "source": "local_mock"}

    fixture_path = Path(__file__).resolve().parents[1] / "data" / "mock_logistics.json"
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        records = fixture.get("records", [])
    except (OSError, json.JSONDecodeError, AttributeError) as ex:
        return {
            "success": False,
            "code": "fixture_unavailable",
            "error": "本地物流模拟数据暂不可用",
            "source": "local_mock",
        }
    if not isinstance(records, list):
        return {
            "success": False,
            "code": "fixture_invalid",
            "error": "本地物流模拟数据格式无效",
            "source": "local_mock",
        }

    record = next(
        (item for item in records if str(item.get("tracking_no", "")).casefold() == tracking_no.casefold()),
        None,
    )
    if record is None:
        return {
            "success": False,
            "code": "not_found",
            "error": "本地模拟数据中没有该运单，请核对运单号；这不代表真实快递状态",
            "source": "local_mock",
            "data_notice": fixture.get("data_notice", "本地演示数据，不代表真实快递平台信息"),
        }

    days_without_update = max(0, int(record.get("days_without_update", 0)))
    lost_days_threshold = _positive_int_env("ECHOMIND_LOGISTICS_LOST_DAYS_THRESHOLD", 7)
    min_followups = _positive_int_env("ECHOMIND_LOGISTICS_MIN_FOLLOWUPS", 2)
    followups = _recent_logistics_followups(req.history, req.intent)
    escalation_candidate = (
        req.intent == IntentCategory.LOGISTICS_LOST
        and days_without_update >= lost_days_threshold
        and followups >= min_followups
    )
    escalation_reason = ""
    if escalation_candidate:
        escalation_reason = (
            f"本地模拟记录停滞 {days_without_update} 天（阈值 {lost_days_threshold} 天）；"
            f"最近 5 条历史消息中有 {followups} 条物流相关用户追问（阈值 {min_followups} 条）"
        )

    return {
        "success": True,
        "source": "local_mock",
        "data_notice": fixture.get("data_notice", "本地演示数据，不代表真实快递平台信息"),
        "status": record.get("status", "未知（模拟）"),
        "last_update": record.get("last_update", ""),
        "days_without_update": days_without_update,
        "events": record.get("events", []),
        "escalation_candidate": escalation_candidate,
        "escalation_reason": escalation_reason,
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查账单核验字段是否齐全；不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string", "description": "支付渠道，例如微信、支付宝、银行卡"}},
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }


def logistics_tools() -> Dict[str, AgentToolSpec]:
    return {
        "query_logistics": make_tool(
            "query_logistics",
            "按运单号查询固定的本地模拟物流记录；不访问真实快递平台，不执行地址修改或理赔。",
            {"tracking_no": {"type": "string", "description": "用户提供的运单号"}},
            query_logistics,
            required=["tracking_no"],
        )
    }
