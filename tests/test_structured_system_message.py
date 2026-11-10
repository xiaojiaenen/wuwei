"""结构化输出注入不得破坏「system 唯一且在首位」。

OpenAI 兼容端点（DeepSeek / vLLM / 内网网关）对 Chat Completions 的 system
位置有硬校验。旧实现 ``messages.insert(0, SystemMessage(...))`` 会把 schema
说明顶到最前，形成相邻两条 system：

    [ schema system, 原有 system, user, ... ]  -> 400 System message must be at the beginning

修法分两层：
- 注入点并入首条 system（顺序与条数都不变，prompt 前缀稳定）；
- 适配器边界合并开头连续的 system，覆盖 RAG / 记忆等其它注入源。
"""

import json

import pytest

from wuwei.core.message import HumanMessage, SystemMessage
from wuwei.llm.adapters.openai import OpenAIAdapter, _coalesce_system_messages
from wuwei.runtime.agent_runner import STRUCTURED_SCHEMA_MARKER, _attach_structured_schema

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def _adapter():
    return OpenAIAdapter(api_key="test-key")


def _roles(payload):
    return [m["role"] for m in payload]


# ── 注入点：并入首条 system，不新增 ─────────────────────────────────


def test_schema_merges_into_existing_system():
    messages = [SystemMessage(content="你是助手"), HumanMessage(content="你好")]
    _attach_structured_schema(messages, SCHEMA)

    assert _roles([m.to_openai() for m in messages]) == ["system", "user"]
    assert messages[0].content.startswith("你是助手")
    assert STRUCTURED_SCHEMA_MARKER in messages[0].content


def test_schema_not_inserted_before_existing_system():
    """回归：旧实现会让前两条都是 system。"""
    messages = [SystemMessage(content="你是助手"), HumanMessage(content="你好")]
    _attach_structured_schema(messages, SCHEMA)

    payload = [m.to_openai() for m in messages]
    assert not (payload[0]["role"] == "system" and payload[1]["role"] == "system")


def test_schema_injection_is_idempotent():
    messages = [SystemMessage(content="你是助手"), HumanMessage(content="你好")]
    for _ in range(5):
        _attach_structured_schema(messages, SCHEMA)

    assert messages[0].content.count(STRUCTURED_SCHEMA_MARKER) == 1
    assert len(messages) == 2


def test_schema_added_when_no_system_present():
    messages = [HumanMessage(content="你好")]
    _attach_structured_schema(messages, SCHEMA)

    assert _roles([m.to_openai() for m in messages]) == ["system", "user"]
    assert STRUCTURED_SCHEMA_MARKER in messages[0].content


def test_schema_on_empty_message_list():
    messages = []
    _attach_structured_schema(messages, SCHEMA)

    assert len(messages) == 1
    assert messages[0].role == "system"


def test_schema_text_is_valid_json_payload():
    messages = [SystemMessage(content="你是助手")]
    _attach_structured_schema(messages, SCHEMA)

    payload = messages[0].content.split(STRUCTURED_SCHEMA_MARKER, 1)[1]
    assert json.loads(payload) == SCHEMA


# ── 适配器边界：合并开头连续的 system ───────────────────────────────


def test_adapter_coalesces_adjacent_system():
    adapter = _adapter()
    request = adapter.build_request(
        [
            SystemMessage(content="schema 说明"),
            SystemMessage(content="你是助手"),
            HumanMessage(content="你好"),
        ]
    )

    assert _roles(request["messages"]) == ["system", "user"]
    assert request["messages"][0]["content"] == "schema 说明\n\n你是助手"


def test_adapter_keeps_single_system_untouched():
    adapter = _adapter()
    request = adapter.build_request(
        [SystemMessage(content="你是助手"), HumanMessage(content="你好")]
    )

    assert _roles(request["messages"]) == ["system", "user"]
    assert request["messages"][0]["content"] == "你是助手"


def test_adapter_merges_middleware_injected_system():
    """RAG / 记忆中间件会插到开头 system 之后，同样会形成相邻两条。"""
    adapter = _adapter()
    request = adapter.build_request(
        [
            SystemMessage(content="你是助手"),
            SystemMessage(content="以下是知识库参考资料：\n- doc"),
            SystemMessage(content="以下是关于用户的已知记忆：\n- 偏好中文"),
            HumanMessage(content="你好"),
        ]
    )

    assert _roles(request["messages"]) == ["system", "user"]
    content = request["messages"][0]["content"]
    assert "你是助手" in content
    assert "知识库" in content
    assert "已知记忆" in content


def test_adapter_keeps_mid_conversation_system():
    """对话中段的 system 是调用方有意为之，不合并、不重排。"""
    adapter = _adapter()
    request = adapter.build_request(
        [
            SystemMessage(content="你是助手"),
            HumanMessage(content="你好"),
            SystemMessage(content="中途切换角色"),
            HumanMessage(content="继续"),
        ]
    )

    assert _roles(request["messages"]) == ["system", "user", "system", "user"]
    assert request["messages"][0]["content"] == "你是助手"


def test_adapter_handles_multimodal_system_blocks():
    payload = _coalesce_system_messages(
        [
            {"role": "system", "content": [{"type": "text", "text": "你是助手"}]},
            {"role": "system", "content": [{"type": "text", "text": "schema 说明"}]},
            {"role": "user", "content": "你好"},
        ]
    )

    assert _roles(payload) == ["system", "user"]
    assert payload[0]["content"] == "你是助手\n\nschema 说明"


def test_adapter_no_system_messages_unchanged():
    adapter = _adapter()
    request = adapter.build_request([HumanMessage(content="你好")])
    assert _roles(request["messages"]) == ["user"]


def test_adapter_preserves_system_name():
    payload = _coalesce_system_messages(
        [
            {"role": "system", "content": "a", "name": "guard"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "hi"},
        ]
    )

    assert payload[0]["name"] == "guard"
    assert payload[0]["content"] == "a\n\nb"


@pytest.mark.parametrize("schema", [{}, {"type": "object"}, SCHEMA])
def test_adapter_output_has_single_system_for_any_schema(schema):
    messages = [SystemMessage(content="你是助手"), HumanMessage(content="你好")]
    _attach_structured_schema(messages, schema)
    request = _adapter().build_request(messages)

    assert _roles(request["messages"]).count("system") == 1
    assert request["messages"][0]["role"] == "system"


# ── 会话历史不被注入污染 ───────────────────────────────────────────


def test_injection_does_not_pollute_session_history():
    """注入只作用于本轮消息视图，会话历史始终只有一条 system。"""
    from wuwei.agent.session import AgentSession

    session = AgentSession(session_id="s1", system_prompt="你是助手")
    session.context.add_user_message("你好")

    for _ in range(3):
        view = [m.model_copy(deep=True) for m in session.context.get_messages()]
        _attach_structured_schema(view, SCHEMA)
        assert _roles([m.to_openai() for m in view]) == ["system", "user"]

    history = session.context.get_messages()
    assert sum(1 for m in history if m.role == "system") == 1
    assert STRUCTURED_SCHEMA_MARKER not in history[0].content


def test_no_duplicate_schema_across_multi_step_run():
    """多步工具循环里每步都注入，但每步视图内 schema 只出现一次。"""
    from wuwei.agent.session import AgentSession

    session = AgentSession(session_id="s1", system_prompt="你是助手")
    session.context.add_user_message("你好")
    for step in range(4):
        session.context.add_ai_message(f"step{step}")
        session.context.add_user_message(f"继续{step}")

    for _ in range(4):
        view = [m.model_copy(deep=True) for m in session.context.get_messages()]
        _attach_structured_schema(view, SCHEMA)
        assert view[0].content.count(STRUCTURED_SCHEMA_MARKER) == 1


# ── max_steps 可编程判定 ───────────────────────────────────────────


def test_max_steps_result_carries_structured_termination():
    from wuwei.llm.types import AgentRunResult

    result = AgentRunResult(
        content="任务未完成",
        status="incomplete",
        termination_reason="max_steps",
        steps_used=3,
        max_steps=3,
    )

    assert result.termination_reason == "max_steps"
    assert result.steps_used == 3
    assert result.max_steps == 3


def test_max_steps_fields_have_defaults():
    from wuwei.llm.types import AgentRunResult

    result = AgentRunResult(content="ok")
    assert result.steps_used == 0
    assert result.max_steps is None
