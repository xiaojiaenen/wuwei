"""R-9 / P1 / P2 回归：会话锁归属、工具执行策略、终止事件、错误包装。

R-9：会话锁原本在生成器里跨 yield 持有，消费方 break 且不 aclose 时锁永不释放。
P1：超时判定、并发上限、异常终止事件、输出截断。
P2：SDK 异常统一包装为 LLMError；重复调用拦截。
"""

import asyncio
import json

import pytest

from wuwei.agent.session import AgentSession
from wuwei.core.errors import LLMError
from wuwei.core.message import BaseMessage, FunctionCall, HumanMessage, ToolCall
from wuwei.graph.checkpoint import MemoryCheckpointer
from wuwei.llm.adapters.base import BaseAdapter
from wuwei.llm.gateway import LLMGateway
from wuwei.llm.types import LLMResponse, LLMResponseChunk
from wuwei.middleware import MiddlewareStack, RepeatCallGuard
from wuwei.runtime.agent_runner import AgentRunner
from wuwei.tools.executor import ToolExecutor
from wuwei.tools.registry import ToolRegistry
from wuwei.tools.tool import Tool, ToolParameters


class FakeLLM:
    def __init__(self, responses=None, error=None, delay=0.0):
        self.responses = list(responses or [])
        self.error = error
        self.delay = delay
        self.calls = []

    async def generate(self, messages, tools=None, stream=False, **kw):
        self.calls.append([m.model_copy(deep=True) for m in messages])
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        response = self.responses.pop(0)
        if not stream:
            return response

        async def chunks():
            yield LLMResponseChunk(
                content=response.message.content,
                tool_calls_complete=response.message.tool_calls,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )

        return chunks()


def response(text="", calls=None):
    return LLMResponse(
        message=BaseMessage(role="assistant", content=text, tool_calls=calls or []),
        finish_reason="tool_calls" if calls else "stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="fake",
        latency_ms=0,
    )


def call(name, id="c1", **args):
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=args))


def registry(*tools):
    reg = ToolRegistry()
    for item in tools:
        reg.register(item)
    return reg


def simple_tool(name, handler, **execution):
    return Tool(
        name=name,
        description=name,
        handler=handler,
        parameters=ToolParameters(),
        execution=execution,
    )


def runner(llm, tool_registry=None, session=None, **kwargs):
    return AgentRunner(
        llm=llm,
        tools=[],
        tool_executor=ToolExecutor(tool_registry or registry()),
        session=session or AgentSession(session_id="s1"),
        **kwargs,
    )


# ── R-9：锁归属 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abandoned_stream_releases_session_lock():
    session = AgentSession(session_id="s1")
    agen = runner(
        FakeLLM([response("hi"), response("again")]),
        session=session,
        consumer_idle_timeout=0.05,
        event_queue_size=1,
    ).stream_events("go")

    seen = 0
    async for _ in agen:
        seen += 1
        if seen == 3:
            break  # 故意不 aclose，模拟客户端断连

    async def take_lock():
        async with session.lock:
            return True

    assert await asyncio.wait_for(take_lock(), 1.0)
    await agen.aclose()
    assert "_blocked_run" not in session.metadata


@pytest.mark.asyncio
async def test_completed_stream_releases_lock_and_emits_terminal_event():
    session = AgentSession(session_id="s1")
    events = [
        event
        async for event in runner(FakeLLM([response("hi")]), session=session).stream_events("go")
    ]
    assert events[-1].type == "run_end"
    assert events[-1].data["status"] == "completed"
    assert not session.lock.locked()


@pytest.mark.asyncio
async def test_cancelled_consumer_saves_cancelled_checkpoint():
    checkpointer = MemoryCheckpointer()
    session = AgentSession(session_id="s1")
    agen = runner(
        FakeLLM([response("hi")], delay=0.5), session=session, checkpointer=checkpointer
    ).stream_events("go")

    async def consume():
        async for _ in agen:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    kinds = [record["metadata"].get("kind") for record in checkpointer.checkpoints.values()]
    assert "cancelled" in kinds
    assert "_blocked_run" not in session.metadata
    await agen.aclose()


@pytest.mark.asyncio
async def test_tool_not_dispatched_on_abort_is_recorded_as_not_executed():
    session = AgentSession(session_id="s1")
    ag_runner = runner(FakeLLM([]), session=session)
    ag_runner._metadata = {
        "run_id": "r1",
        "tool_states": {"c1": "preparing"},
        "unresolved_calls": [
            {"id": "c1", "function": {"name": "write_file", "arguments": {}}},
        ],
        "approval_decisions": {},
    }
    ag_runner._block_uncertain_tools()

    assert "_blocked_run" not in session.metadata
    assert ag_runner._metadata["tool_states"]["c1"] == "error"
    last = session.context.get_messages()[-1]
    assert last.role == "tool"
    assert json.loads(last.content)["error"]["type"] == "not_executed"


@pytest.mark.asyncio
async def test_dispatched_tool_on_abort_still_requires_reconciliation():
    session = AgentSession(session_id="s1")
    ag_runner = runner(FakeLLM([]), session=session)
    ag_runner._metadata = {
        "run_id": "r1",
        "tool_states": {"c1": "running"},
        "dispatched_calls": ["c1"],
        "unresolved_calls": [
            {"id": "c1", "function": {"name": "pay", "arguments": {}}},
        ],
        "approval_decisions": {},
    }
    ag_runner._block_uncertain_tools()

    assert ag_runner._metadata["tool_states"]["c1"] == "unknown"
    assert session.metadata["_blocked_run"] == "r1"


@pytest.mark.asyncio
async def test_agent_stream_abandon_does_not_block_next_run():
    from wuwei import Agent

    agent = Agent(
        llm=FakeLLM([response("hi"), response("again")]),
        load_builtins=False,
        stream_idle_timeout=0.05,
    )
    events = agent.stream_events("go")
    seen = 0
    async for _ in events:
        seen += 1
        if seen == 2:
            break

    result = await asyncio.wait_for(agent.run("again"), 2)
    assert result.status == "completed"
    assert result.content == "again"
    await events.aclose()


# ── R-14：超时语义 ──────────────────────────────────────────────────


async def _slow():
    await asyncio.sleep(0.2)
    return "late"


async def _raises_builtin_timeout():
    raise TimeoutError("upstream slow")


@pytest.mark.asyncio
async def test_side_effect_timeout_is_unknown():
    executor = ToolExecutor(
        registry(simple_tool("pay", _slow, side_effect=True, timeout_seconds=0.01))
    )
    message = await executor.execute_one(call("pay"))
    outcome = message.metadata["tool_outcome"]
    assert outcome["status"] == "unknown"
    assert outcome["execution_may_continue"] is True


@pytest.mark.asyncio
async def test_idempotent_side_effect_timeout_is_retryable():
    executor = ToolExecutor(
        registry(
            simple_tool("put", _slow, side_effect=True, idempotent=True, timeout_seconds=0.01)
        )
    )
    message = await executor.execute_one(call("put"))
    outcome = message.metadata["tool_outcome"]
    assert outcome["status"] == "error"
    assert json.loads(message.content)["retryable"] is True


@pytest.mark.asyncio
async def test_readonly_timeout_is_plain_failure():
    executor = ToolExecutor(registry(simple_tool("get", _slow, timeout_seconds=0.01)))
    message = await executor.execute_one(call("get"))
    assert message.metadata["tool_outcome"]["status"] == "error"
    assert json.loads(message.content)["error"]["type"] == "ToolTimeout"


@pytest.mark.asyncio
async def test_tool_raised_timeout_is_not_a_framework_timeout():
    executor = ToolExecutor(
        registry(simple_tool("get", _raises_builtin_timeout, timeout_seconds=5))
    )
    message = await executor.execute_one(call("get"))
    payload = json.loads(message.content)
    assert payload["error"]["type"] == "TimeoutError"
    assert payload["status"] == "failed"


# ── R-13：并发上限与默认串行 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrency_is_capped():
    peak = 0
    current = 0

    async def tracked():
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.02)
        current -= 1
        return "ok"

    executor = ToolExecutor(registry(simple_tool("noop", tracked)), max_concurrency=3)
    await executor.execute([call("noop", id=f"c{i}") for i in range(12)], concurrent=True)
    assert peak <= 3


@pytest.mark.asyncio
async def test_side_effect_tools_are_serial_by_default():
    order = []

    async def first():
        order.append("a-start")
        await asyncio.sleep(0.02)
        order.append("a-end")
        return "a"

    async def second():
        order.append("b-start")
        await asyncio.sleep(0.02)
        order.append("b-end")
        return "b"

    executor = ToolExecutor(
        registry(
            simple_tool("a", first, side_effect=True),
            simple_tool("b", second, side_effect=True),
        )
    )
    await executor.execute([call("a", id="c1"), call("b", id="c2")], concurrent=True)
    assert order == ["a-start", "a-end", "b-start", "b-end"]


# ── R-11：异常路径的终止事件 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_run_emits_terminal_events_before_raising():
    ag_runner = runner(FakeLLM(error=RuntimeError("provider down")))
    seen = []
    with pytest.raises(RuntimeError):
        async for event in ag_runner.stream_events("go"):
            seen.append((event.type, event.data.get("status")))
    assert [item[0] for item in seen][-3:] == ["error", "done", "run_end"]
    assert seen[-1][1] == "error"


# ── R-12：输出截断 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_oversized_tool_output_is_truncated():
    async def huge():
        return "x" * 500

    executor = ToolExecutor(registry(simple_tool("echo", huge)), max_output_chars=20)
    message = await executor.execute_one(call("echo"))
    assert message.content.startswith("x" * 20)
    assert "[truncated 480 chars]" in message.content


# ── R-2：SDK 异常包装 ───────────────────────────────────────────────


class ProviderError(Exception):
    status_code = 400


class _FailingAdapter(BaseAdapter):
    provider_name = "openai"
    model = "boom"

    def __init__(self, error=None):
        self.error = error or ProviderError("bad request")

    def build_request(self, messages, tools=None, stream=False, **kwargs):
        return {}

    async def call(self, request):
        raise self.error

    def parse_response(self, raw):
        raise NotImplementedError

    def parse_stream_chunk(self, chunk):
        return None


@pytest.mark.asyncio
async def test_provider_errors_are_wrapped_as_llm_error():
    gateway = LLMGateway(
        {
            "provider": "openai",
            "api_key": "test-key",
            "adapter": _FailingAdapter(),
            "retry": {"max_attempts": 1},
        }
    )
    with pytest.raises(LLMError) as caught:
        await gateway.generate([HumanMessage(content="hi")])
    error = caught.value
    assert error.details["status_code"] == 400
    assert error.details["category"] == "fatal"
    assert isinstance(error.__cause__, ProviderError)


@pytest.mark.asyncio
async def test_contract_errors_are_not_rewrapped():
    gateway = LLMGateway(
        {
            "provider": "openai",
            "api_key": "test-key",
            "adapter": _FailingAdapter(error=ValueError("bad arguments")),
            "retry": {"max_attempts": 1},
        }
    )
    with pytest.raises(ValueError):
        await gateway.generate([HumanMessage(content="hi")])


# ── R-16：重复调用拦截 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeat_call_guard_blocks_the_loop():
    executed = []

    async def probe():
        executed.append(1)
        return "still failing"

    llm = FakeLLM(
        [
            response(calls=[call("probe", id="c1")]),
            response(calls=[call("probe", id="c2")]),
            response(calls=[call("probe", id="c3")]),
            response("done"),
        ]
    )
    ag_runner = runner(llm, registry(simple_tool("probe", probe)))
    ag_runner.middleware = MiddlewareStack(RepeatCallGuard(threshold=2))
    result = await ag_runner.run("go")

    assert result.status == "completed"
    assert len(executed) == 2
    contents = [m.content for m in ag_runner.session.context.get_messages() if m.role == "tool"]
    assert any("repeated_call" in content for content in contents)


# ── R-17：事件类型声明与实际发射一致 ───────────────────────────────


@pytest.mark.asyncio
async def test_every_emitted_event_type_is_declared():
    from typing import get_args

    from wuwei.llm.types import AgentEventType

    tools = registry(simple_tool("probe", lambda: "ok"))
    ag_runner = runner(FakeLLM([response(calls=[call("probe")]), response("done")]), tools)
    declared = set(get_args(AgentEventType))
    async for event in ag_runner.stream_events("go"):
        assert event.type in declared, event.type
