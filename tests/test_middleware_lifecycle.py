import asyncio
import json

import pytest

from wuwei.core.message import AIMessage, BaseMessage, FunctionCall, ToolCall
from wuwei.graph.state import State
from wuwei.llm import AgentEvent, LLMResponse
from wuwei.middleware.base import MiddlewareContext
from wuwei.middleware.context_compression import ContextCompressionMiddleware
from wuwei.observability.tracing import TracingMiddleware
from wuwei.runtime.event_bus import EventBus


class Span:
    def __init__(self, closed):
        self.closed = closed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed.append(True)

    def set_attribute(self, *args):
        pass


class Tracer:
    def __init__(self, closed):
        self.closed = closed

    def start_as_current_span(self, *args):
        return Span(self.closed)


@pytest.mark.asyncio
async def test_trace_resources_json_safe_and_close():
    closed = []
    mw = TracingMiddleware()
    mw._tracer = Tracer(closed)
    ctx = MiddlewareContext(state=State())
    await mw.before_llm(ctx)
    await mw.before_tool(ctx, ToolCall(id="c", function=FunctionCall(name="f")))
    json.dumps(ctx.metadata)
    await mw.aclose(ctx)
    assert len(closed) == 2
    await mw.aclose(ctx)
    assert len(closed) == 2


@pytest.mark.asyncio
async def test_compression_uses_governed_model_and_no_cross_context_summary():
    class LLM:
        async def generate(self, **kw):
            raise AssertionError("ungoverned call")

    prompts = []

    async def model_call(messages, tools=None):
        prompts.append(messages[0].content)
        return LLMResponse(content=f"summary{len(prompts)}", finish_reason="stop")

    mw = ContextCompressionMiddleware(LLM(), trigger_tokens=1, keep_recent_turns=1)

    def context(label):
        return MiddlewareContext(
            state=State(
                messages=[
                    BaseMessage(role="user" if i % 2 == 0 else "assistant", content=label * 20)
                    for i in range(12)
                ]
            ),
            config={"model_call": model_call},
        )

    await mw.before_llm(context("privateA"))
    await mw.before_llm(context("privateB"))
    assert len(prompts) == 2 and "summary1" not in prompts[1]


@pytest.mark.asyncio
async def test_event_order_and_drain():
    bus = EventBus()
    seen = []

    async def handler(e):
        if e.step == 1:
            await asyncio.sleep(0.01)
        seen.append(e.step)

    bus.subscribe(handler)
    bus.publish_nowait(AgentEvent(type="done", session_id="s", step=1))
    bus.publish_nowait(AgentEvent(type="done", session_id="s", step=2))
    await bus.drain()
    assert seen == [1, 2]
    await bus.aclose()
    assert not bus._tasks


@pytest.mark.asyncio
async def test_memory_counter_scoped_and_budget_failure_propagates():
    from wuwei.middleware.budget import BudgetExhausted
    from wuwei.middleware.memory import MemoryMiddleware

    class Store:
        async def add(self, *args, **kwargs):
            pass

    class LLM:
        async def generate(self, **kw):
            raise AssertionError("ungoverned")

    calls = []

    async def model_call(messages, tools=None):
        calls.append(1)
        return LLMResponse(content="[]", finish_reason="stop")

    mw = MemoryMiddleware(LLM(), Store(), extract_every_n=2)

    def context():
        return MiddlewareContext(
            state=State(messages=[BaseMessage(role="user", content="hi")]),
            config={"model_call": model_call},
        )

    a = context()
    b = context()
    await mw.after_llm(a, AIMessage(content="one"))
    await mw.after_llm(b, AIMessage(content="other user"))
    assert calls == []
    await mw.after_llm(a, AIMessage(content="two"))
    assert calls == [1]

    async def exhausted(messages, tools=None):
        raise BudgetExhausted("no budget")

    a.config["model_call"] = exhausted
    await mw.after_llm(a, AIMessage(content="three"))
    with pytest.raises(BudgetExhausted):
        await mw.after_llm(a, AIMessage(content="four"))


def test_nowait_no_loop_does_not_leak_coroutines(recwarn):
    import gc

    bus = EventBus()

    async def handler(event):
        raise AssertionError("must not run")

    bus.subscribe(handler)
    bus.subscribe(lambda event: handler(event))
    bus.publish_nowait(AgentEvent(type="done", session_id="s", step=0))
    gc.collect()
    assert not [w for w in recwarn if "never awaited" in str(w.message)]


@pytest.mark.asyncio
async def test_tracing_error_closes_once():
    closed = []
    mw = TracingMiddleware()
    mw._tracer = Tracer(closed)
    ctx = MiddlewareContext(state=State())
    await mw.before_llm(ctx)
    error = ValueError("failure")
    assert await mw.on_error(ctx, error) is error
    assert closed == [True]
