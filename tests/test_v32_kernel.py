"""v3.2 内核测试：interrupt/resume / EventBus / ActionLog / 检查点"""

import pytest

from wuwei.agent import Agent
from wuwei.core.message import BaseMessage as Message, ToolCall, FunctionCall
from wuwei.graph.checkpoint import MemoryCheckpointer
from wuwei.llm.types import LLMResponse, LLMResponseChunk
from wuwei.middleware.hitl import HitlMiddleware
from wuwei.middleware.stack import MiddlewareStack
from wuwei.runtime import EventBus
from wuwei.runtime.action_log import get_or_create_log
from wuwei.tools import Tool, ToolParameters


# ── 测试基建（与 test_v31_core 相同模式的 fake） ─────────────────────


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def build_request(self, messages, tools=None, stream=False, **kwargs):
        self.calls.append({"messages": messages, "tools": tools or [], "stream": stream})
        return {}

    async def call(self, request):
        return request

    def parse_response(self, raw):
        return self.responses.pop(0)

    def parse_stream_chunk(self, chunk):
        return None


class FakeGateway:
    def __init__(self, responses):
        self.adapter = FakeAdapter(responses)

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        self.adapter.calls.append({"messages": messages, "tools": tools or [], "stream": stream})
        resp = self.adapter.responses.pop(0)
        if not stream:
            return resp

        async def _gen():
            yield LLMResponseChunk(
                content=resp.message.content,
                finish_reason=resp.finish_reason,
                tool_calls_complete=resp.message.tool_calls or None,
                usage=resp.usage,
            )

        return _gen()


def make_llm_response(content="", tool_calls=None):
    return LLMResponse(
        message=Message(role="assistant", content=content, tool_calls=tool_calls or []),
        finish_reason="tool_calls" if tool_calls else "stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="fake",
        latency_ms=0,
    )


def make_tool(name, output, requires_approval=False):
    async def _run(**kwargs):
        return output

    return Tool(
        name=name,
        description=f"test {name}",
        parameters=ToolParameters(),
        handler=_run,
        execution={"requires_approval": requires_approval} if requires_approval else {},
    )


def make_agent(responses, middleware=None, checkpointer=None, event_bus=None):
    return Agent(
        llm=FakeGateway(responses),
        load_builtins=False,
        middleware=middleware,
        checkpointer=checkpointer,
        event_bus=event_bus,
    )


# ── interrupt / resume ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interrupt_and_resume_approved():
    """HITL interrupt 模式：挂起 → 批准 → 恢复执行工具并完成。"""
    tool_call = ToolCall(id="c1", function=FunctionCall(name="deploy", arguments={"env": "prod"}))
    responses = [
        make_llm_response(tool_calls=[tool_call]),
        make_llm_response(content="部署完成"),
        make_llm_response(content="部署完成"),
    ]
    agent = make_agent(
        responses,
        middleware=None,
        checkpointer=MemoryCheckpointer(),
    )
    agent.tool_registry.register(make_tool("deploy", {"ok": True}, requires_approval=True))
    hitl = HitlMiddleware(mode="interrupt", approval_policy="declared")
    agent.middleware.add(hitl)

    result = await agent.run("部署到生产环境")
    assert result.status == "interrupted"
    assert result.interrupt is not None
    assert result.interrupt["tool_name"] == "deploy"
    assert result.interrupt["checkpoint_id"]

    # 批准恢复：执行 deploy，然后模型继续
    result2 = await agent.resume(result.interrupt, approved=True)
    assert result2.status == "completed"
    assert result2.content == "部署完成"

    tool_msgs = [
        m for m in agent._default_session.context.get_messages() if m.role == "tool"
    ]
    assert any('"ok": true' in m.content.lower() for m in tool_msgs)


@pytest.mark.asyncio
async def test_interrupt_and_resume_rejected():
    """拒绝恢复：拒绝结果回喂模型，agent 不执行工具而继续对话。"""
    tool_call = ToolCall(id="c1", function=FunctionCall(name="deploy", arguments={}))
    responses = [
        make_llm_response(tool_calls=[tool_call]),
        make_llm_response(content="好的，已取消部署"),
    ]
    agent = make_agent(responses, checkpointer=MemoryCheckpointer())
    agent.tool_registry.register(make_tool("deploy", {"ok": True}, requires_approval=True))
    agent.middleware.add(HitlMiddleware(mode="interrupt", approval_policy="declared"))

    result = await agent.run("部署")
    assert result.status == "interrupted"

    result2 = await agent.resume(result.interrupt, approved=False)
    assert result2.status == "completed"
    assert result2.content == "好的，已取消部署"

    # 工具从未被执行，会话里只有拒绝回喂
    tool_msgs = [
        m for m in agent._default_session.context.get_messages() if m.role == "tool"
    ]
    assert len(tool_msgs) == 1
    assert "approval_rejected" in tool_msgs[0].content


@pytest.mark.asyncio
async def test_checkpoints_written_per_step():
    """配置 checkpointer 后每步应落 pre_tool/step/final 检查点。"""
    responses = [
        make_llm_response(content="直接回答"),
    ]
    cp = MemoryCheckpointer()
    agent = make_agent(responses, checkpointer=cp)
    await agent.run("hi")

    kinds = [c["metadata"].get("kind") for c in await cp.list_checkpoints()]
    assert "final" in kinds


@pytest.mark.asyncio
async def test_interrupt_without_checkpointer_raises():
    """interrupt without durable storage must fail closed instead of returning an unusable token."""
    tool_call = ToolCall(id="c1", function=FunctionCall(name="deploy", arguments={}))
    responses = [make_llm_response(tool_calls=[tool_call])]
    agent = make_agent(responses)
    agent.tool_registry.register(make_tool("deploy", "x", requires_approval=True))
    agent.middleware.add(HitlMiddleware(mode="interrupt", approval_policy="declared"))

    with pytest.raises(ValueError, match="checkpointer"):
        await agent.run("部署")


# ── EventBus ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_bus_receives_all_events():
    """订阅者应收到 run 全程事件，且不影响主流程。"""
    collected = []
    bus = EventBus()
    unsub = bus.subscribe(lambda e: collected.append(e.type))

    agent = make_agent([make_llm_response(content="ok")], event_bus=bus)
    async for _ in agent.stream_events("hi"):
        pass

    # fire-and-forget：等待事件循环排空
    import asyncio

    for _ in range(10):
        if "run_end" in collected:
            break
        await asyncio.sleep(0.01)
    assert "run_start" in collected
    assert "text_delta" in collected
    assert "run_end" in collected
    unsub()


@pytest.mark.asyncio
async def test_event_bus_subscriber_error_isolated():
    """订阅者抛异常不影响 agent 运行。"""
    bus = EventBus()
    bus.subscribe(lambda e: 1 / 0)
    agent = make_agent([make_llm_response(content="ok")], event_bus=bus)
    result = await agent.run("hi")
    assert result.status == "completed"


# ── Action Log ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_action_log_records_llm_and_tool_steps():
    tool_call = ToolCall(id="c1", function=FunctionCall(name="calc_x", arguments={"a": 1}))
    responses = [
        make_llm_response(tool_calls=[tool_call]),
        make_llm_response(content="done"),
    ]
    agent = make_agent(responses)
    agent.tool_registry.register(make_tool("calc_x", 42))

    await agent.run("计算")
    log = get_or_create_log(agent._default_session)
    kinds = [e.kind for e in log.entries]
    assert "llm" in kinds and "tool" in kinds

    tool_entry = next(e for e in log.entries if e.kind == "tool")
    assert tool_entry.summary.startswith("calc_x")
    assert tool_entry.digest
