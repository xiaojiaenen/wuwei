"""2026-11 审计修复回归测试。

覆盖：
- 检查点 v2（metadata 单写 + messages 从会话副本重灌）与 v1 兼容
- Ledger 全量保真（>100k 截断前归档、占位符报告真实长度）
- SubAgent 工具 schema 与 input_model 契约一致
- build_child_agent 的 UNSET/显式 None 语义与治理继承
- http 插件错误契约（异常不再被吞成成功文本）
- skill 指引按 session 注入 + 并入首条 system
- LLMError.details["provider"] 反映真实供应商
"""

import asyncio
import json

import pytest

from wuwei.graph.checkpoint import MemoryCheckpointer, _decode_state
from wuwei.graph.state import State


# ── 检查点 v2 ──────────────────────────────────────────────────────────────


def _session_like_messages():
    from wuwei.llm.types import Message

    return [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    ]


@pytest.mark.asyncio
async def test_checkpoint_v2_metadata_single_write_and_message_rehydration():
    cp = MemoryCheckpointer()
    from wuwei.memory.context import Context

    context = Context()
    for m in _session_like_messages():
        context.get_messages().append(m)
    session_dict = {
        "session_id": "s1",
        "system_prompt": "sys",
        "max_steps": 10,
        "metadata": {},
        "context": context.to_dict(),
    }
    metadata = {
        "schema_version": 1,
        "kind": "step",
        "session_id": "s1",
        "session": session_dict,
        "run_state": {"run_id": "r1"},
    }
    # runner 形态：State 不携带消息，唯一副本在 metadata.session.context
    state = State(messages=[], metadata={}, step=3)
    cid = await cp.save(state, metadata=metadata)
    loaded = await cp.load(cid)
    assert loaded.metadata.get("session_id") == "s1"
    assert [m.role for m in loaded.messages] == ["system", "user"]
    assert loaded.messages[1].content == "hi"
    assert loaded.step == 3


def test_checkpoint_v1_snapshot_still_loads():
    """v1 快照（metadata 内嵌）必须继续可读——旧检查点兼容。"""
    v1_snapshot = {
        "messages": [{"role": "user", "content": "legacy"}],
        "metadata": {"kind": "step", "run_state": {"run_id": "old"}},
        "step": 2,
    }
    loaded = _decode_state(v1_snapshot, {"kind": "ignored-column"})
    assert loaded.metadata["kind"] == "step"
    assert loaded.messages[0].content == "legacy"
    assert loaded.step == 2


def test_checkpoint_v2_keeps_graph_style_state_metadata():
    """StateGraph 直存路径（自带 state.metadata）语义不变。"""
    snapshot = {
        "schema_version": 2,
        "messages": [{"role": "user", "content": "graph"}],
        "metadata": {"node": "agent"},
        "step": 1,
    }
    loaded = _decode_state(snapshot, {"column": "meta"})
    assert loaded.metadata == {"node": "agent"}


# ── Ledger 全量保真 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ledger_archives_full_output_beyond_executor_truncation():
    from wuwei.middleware.ledger import ContextLedger, LedgerMiddleware
    from wuwei.middleware.base import MiddlewareContext
    from wuwei.tools.executor import ToolExecutor
    from wuwei.tools.registry import ToolRegistry
    from wuwei.tools.tool import Tool, ToolParameters

    registry = ToolRegistry()

    @registry.tool(description="big output")
    def big() -> str:
        return "x" * 150_000

    executor = ToolExecutor(registry)
    ledger = ContextLedger()
    middleware = LedgerMiddleware(ledger)

    from wuwei.llm.types import ToolCall, FunctionCall

    tc = ToolCall(id="call1", function=FunctionCall(name="big", arguments={}))
    msg = await executor.execute_one(tc)
    assert len(msg.content) < 150_000  # 上下文消息保持截断

    # 模拟 runner._commit_tool 的交接契约：从执行器侧信道取原文，
    # 临时挂到 metadata 供 after_tool 消费，钩子跑完立即移除。
    full = executor.drain_full_output(tc.id)
    assert full is not None and len(full) == 150_000
    msg.metadata["_wuwei_full_content"] = full

    ctx = MiddlewareContext(state=State(), config={})
    final = await middleware.after_tool(ctx, msg)
    entry = ledger.list_entries()[-1]
    assert len(entry.content) == 150_000  # 归档的是未截断原文
    assert "原文 150000 字符" in final.content  # 占位符报告真实长度
    assert "_wuwei_full_content" not in (final.metadata or {})  # ledger 消费后移除


# ── SubAgent schema 契约 ──────────────────────────────────────────────────


def test_subagent_tool_schema_declares_input_model_contract():
    from pydantic import BaseModel

    from wuwei.agent.sub_agent import SubAgent

    class TaskInput(BaseModel):
        query: str
        limit: int = 5

    sub = SubAgent(name="searcher", description="search things", input_model=TaskInput)
    schema = sub.to_tool_schema()
    task = schema["function"]["parameters"]["properties"]["task"]
    assert "JSON object serialized to a string" in task["description"]
    assert '"query"' in task["description"] and '"limit"' in task["description"]


def test_subagent_tool_schema_plain_task_when_no_input_model():
    from wuwei.agent.sub_agent import SubAgent

    schema = SubAgent(name="worker", description="work").to_tool_schema()
    task = schema["function"]["parameters"]["properties"]["task"]
    assert task["description"] == "Task description for worker"


# ── build_child_agent 语义 ────────────────────────────────────────────────


def test_build_child_agent_unset_keeps_agent_defaults():
    from wuwei.agent.child import build_child_agent

    worker = build_child_agent(llm=object())
    assert worker.default_max_steps == 10  # Agent 内置默认未被 None 覆盖
    assert worker.default_system_prompt is not None


def test_build_child_agent_explicit_none_clears_defaults():
    from wuwei.agent.child import build_child_agent

    worker = build_child_agent(llm=object(), system_prompt=None, max_steps=None)
    assert worker.default_system_prompt is None
    assert worker.default_max_steps is None


def test_build_child_agent_inherits_and_excludes_middleware():
    from wuwei.agent.child import build_child_agent, inherit_middleware
    from wuwei.middleware import Middleware, MiddlewareStack

    class M(Middleware):
        pass

    a, b = M(), M()
    parent = MiddlewareStack(a, b)
    child = build_child_agent(
        llm=object(), parent_middleware=parent, child_middleware=None, exclude_middleware=a
    )
    names = [type(m) for m in child.middleware.middlewares]
    assert M in names and child.middleware.middlewares.count(b) == 1
    # inherit_middleware 去重
    stack = inherit_middleware(parent, MiddlewareStack(a))
    assert len(stack.middlewares) == 2


# ── http 插件错误契约 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_plugin_raises_instead_of_returning_error_text():
    from wuwei.plugin.builtin.http import setup as http_setup
    from wuwei.tools import ToolRegistry

    registry = ToolRegistry()

    class Ctx:
        tool_registry = registry

    http_setup(Ctx())
    tool = registry.get("http_get")
    # 端口 1 不可达：此前返回 "请求失败: ..."（成功态），现在必须抛出
    with pytest.raises(Exception):
        await tool.invoke({"url": "http://127.0.0.1:1/"})

    from wuwei.tools.executor import ToolExecutor

    executor = ToolExecutor(registry)
    from wuwei.llm.types import ToolCall, FunctionCall

    msg = await executor.execute_one(
        ToolCall(id="c1", function=FunctionCall(name="http_get", arguments={"url": "http://127.0.0.1:1/"}))
    )
    payload = json.loads(msg.content)
    assert payload.get("ok") is False  # 执行器包装为结构化错误
    assert msg.metadata["tool_outcome"]["status"] == "error"


# ── skill 注入 ────────────────────────────────────────────────────────────


def test_skill_prompt_injected_per_session_and_merged_into_first_system():
    from types import SimpleNamespace

    from wuwei.middleware.base import MiddlewareContext
    from wuwei.plugin.builtin.skill import SkillPromptMiddleware
    from wuwei.skill import FileSystemSkillProvider, SkillManager

    class FakeProvider:
        def list_skills(self):
            return [SimpleNamespace(name="demo", description="A demo skill")]

    class FakeManager:
        def list_skills(self):
            return FakeProvider().list_skills()

    middleware = SkillPromptMiddleware(FakeManager())
    from wuwei.llm.types import Message

    messages = [Message(role="system", content="You are helpful."), Message(role="user", content="hi")]

    class FakeState:
        pass

    state = FakeState()
    state.messages = messages

    def make_ctx(session_id):
        return MiddlewareContext(
            state=state, config={"session": SimpleNamespace(session_id=session_id)}
        )

    async def run(sid):
        return await middleware.before_llm(make_ctx(sid))

    ctx1 = asyncio.run(run("s1"))
    assert ctx1.state.messages[0].role == "system"
    assert ctx1.state.messages[0].content.count("A demo skill") == 1
    assert "You are helpful." in ctx1.state.messages[0].content  # 并入而非另插
    assert len(ctx1.state.messages) == 2  # 没有新增消息
    # 同一 session 的后续调用不重复注入
    asyncio.run(run("s1"))
    assert ctx1.state.messages[0].content.count("A demo skill") == 1
    # 新 session 重新注入（修复：此前按实例一次性，跨会话丢失）
    ctx2 = asyncio.run(run("s2"))
    assert "A demo skill" in ctx2.state.messages[0].content


# ── LLMError provider 字段 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_error_details_provider_is_real_provider():
    from wuwei.llm.gateway import LLMGateway

    async def boom(*args, **kwargs):
        raise RuntimeError("network down")

    class FakeAdapter:
        provider_name = "openai"

        async def call(self, request):
            raise RuntimeError("network down")

        def build_request(self, messages, tools=None, stream=False, **kwargs):
            return {}

        async def aclose(self):
            return None

    g = LLMGateway({"provider": "openai", "api_key": "x", "retry": {"max_attempts": 1, "base_delay": 0}})
    g.adapter = FakeAdapter()
    with pytest.raises(Exception) as exc_info:
        await g.generate([_msg()], stream=False)
    details = exc_info.value.details
    assert details["provider"] == "openai"  # 此前恒为 "LLMGateway"


def _msg():
    from wuwei.llm.types import Message

    return Message(role="user", content="hi")
