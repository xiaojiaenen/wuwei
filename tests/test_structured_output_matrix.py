"""结构化输出模式矩阵回归测试（2026-11 审计修复）。

修复内容：
1. ``auto`` 降级链从 native→prompted 扩展为 native→tool→prompted；
2. prompted 注入不再向消息尾部追加 system，而是与 runner 共用
   ``inject_prompted_schema`` 并入首条 system（严格端点 400 根因）。
"""

import json

import pytest

from wuwei.llm.gateway import LLMGateway
from wuwei.llm.output import (
    STRUCTURED_SCHEMA_MARKER,
    STRUCTURED_TOOL_NAME,
    OutputSpec,
    inject_prompted_schema,
)
from wuwei.llm.profile import ModelCapabilities, ModelProfile, default_profile
from wuwei.llm.types import Message

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _compat_profile(native_schema: bool = False, tool_calling: bool = True) -> ModelProfile:
    return ModelProfile(
        name="compat",
        capabilities=ModelCapabilities(
            native_schema=native_schema, tool_calling=tool_calling
        ),
    )


def _roles(messages):
    return [m["role"] if isinstance(m, dict) else m.role for m in messages]


def _contents(messages):
    return [m["content"] if isinstance(m, dict) else m.content for m in messages]


# ── auto 降级链 ────────────────────────────────────────────────────────────


def test_auto_prefers_native_when_supported():
    spec = OutputSpec(schema=SCHEMA)
    assert spec.resolve(default_profile("openai")) == "native"


def test_auto_falls_back_to_tool_without_native_but_with_tool_calling():
    profile = _compat_profile()
    assert OutputSpec(schema=SCHEMA).resolve(profile, has_tools=False) == "tool"


def test_auto_falls_back_to_prompted_when_tools_present():
    """带可执行工具的请求在 auto 下保持 prompted（结构化工具必须独占 tool_choice）。"""
    profile = _compat_profile()
    assert OutputSpec(schema=SCHEMA).resolve(profile, has_tools=True) == "prompted"


def test_auto_falls_back_to_prompted_without_tool_calling():
    profile = _compat_profile(tool_calling=False)
    assert OutputSpec(schema=SCHEMA).resolve(profile, has_tools=False) == "prompted"


def test_explicit_tool_mode_without_tool_calling_raises():
    profile = _compat_profile(tool_calling=False)
    with pytest.raises(ValueError, match="tool output"):
        OutputSpec(schema=SCHEMA, mode="tool").resolve(profile)


# ── gateway.build_request：兼容端点默认路径 ────────────────────────────────


def test_compat_endpoint_auto_uses_tool_mode_and_forces_tool_choice():
    """自定义 base_url 端点（native_schema=False）在 auto 下应选 tool 强制输出。"""
    g = LLMGateway({"provider": "openai", "api_key": "x", "base_url": "https://compat.test/v1"})
    assert g.model_profile.capabilities.native_schema is False
    request = g.build_request(
        [Message(role="user", content="hi")], output_spec={"schema": SCHEMA}
    )
    tools = request["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == STRUCTURED_TOOL_NAME
    assert request["tool_choice"]["function"]["name"] == STRUCTURED_TOOL_NAME
    # 不应有任何 schema 提示词被注入消息
    assert all(STRUCTURED_SCHEMA_MARKER not in (c or "") for c in _contents(request["messages"]))


def test_prompted_mode_merges_into_first_system_not_tail():
    g = LLMGateway({"provider": "openai", "api_key": "x", "base_url": "https://compat.test/v1"})
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hi"),
    ]
    request = g.build_request(
        messages, output_spec={"schema": SCHEMA, "mode": "prompted"}
    )
    assert _roles(request["messages"]) == ["system", "user"]
    head = _contents(request["messages"])[0]
    assert head.count(STRUCTURED_SCHEMA_MARKER) == 1
    injected = json.loads(head.split(STRUCTURED_SCHEMA_MARKER, 1)[1])
    assert injected["properties"]["answer"] == {"type": "string"}


def test_prompted_mode_without_system_prepends_one():
    g = LLMGateway({"provider": "openai", "api_key": "x", "base_url": "https://compat.test/v1"})
    request = g.build_request(
        [Message(role="user", content="hi")],
        output_spec={"schema": SCHEMA, "mode": "prompted"},
    )
    assert _roles(request["messages"]) == ["system", "user"]


def test_prompted_injection_is_idempotent():
    messages = [Message(role="system", content="You are helpful.")]
    inject_prompted_schema(messages, SCHEMA)
    inject_prompted_schema(messages, SCHEMA)
    assert messages[0].content.count(STRUCTURED_SCHEMA_MARKER) == 1


def test_anthropic_prompted_injection_lands_in_single_top_system():
    g = LLMGateway({"provider": "anthropic", "api_key": "x"})
    request = g.build_request(
        [Message(role="system", content="You are helpful."), Message(role="user", content="hi")],
        output_spec={"schema": SCHEMA, "mode": "prompted"},
    )
    system = request["system"]
    text = "".join(
        b["text"] if isinstance(b, dict) else str(b) for b in system
    ) if isinstance(system, list) else str(system)
    assert text.count(STRUCTURED_SCHEMA_MARKER) == 1
    assert "You are helpful." in text


# ── runner 恢复路径与 gateway 主路径注入文案一致 ───────────────────────────


def test_runner_and_gateway_share_marker_and_implementation():
    from wuwei.runtime.agent_runner import (
        STRUCTURED_SCHEMA_MARKER as runner_marker,
        _attach_structured_schema,
    )

    assert runner_marker == STRUCTURED_SCHEMA_MARKER
    messages = [Message(role="system", content="sys")]
    _attach_structured_schema(messages, SCHEMA)
    inject_prompted_schema(messages, SCHEMA)
    assert messages[0].content.count(STRUCTURED_SCHEMA_MARKER) == 1
