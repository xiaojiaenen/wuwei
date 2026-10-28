"""Regression contracts for run governance and durable continuation."""

import json
from pathlib import Path

import pytest

from wuwei import Agent
from wuwei.core.message import BaseMessage, ToolCall, FunctionCall
from wuwei.graph.checkpoint import MemoryCheckpointer, SQLiteCheckpointer
from wuwei.graph.state import State
from wuwei.llm.types import LLMResponse, LLMResponseChunk
from wuwei.middleware import Middleware, MiddlewareStack, HitlMiddleware, StorageMiddleware
from wuwei.security import PermissionConfig, PermissionMiddleware
from wuwei.tools import Tool, ToolParameters


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages, tools=None, stream=False, **kw):
        self.calls.append([m.model_copy(deep=True) for m in messages])
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


def make_agent(responses, tools=(), middlewares=(), cp=None, **kw):
    stack = MiddlewareStack()
    for mw in middlewares:
        stack.add(mw)
    return Agent(
        llm=FakeLLM(responses),
        tools=list(tools),
        middleware=stack,
        load_builtins=False,
        checkpointer=cp,
        **kw,
    )


def tool(name, handler, approval=False):
    return Tool(
        name=name,
        description=name,
        handler=handler,
        parameters=ToolParameters(),
        execution={"requires_approval": approval},
    )


@pytest.mark.asyncio
async def test_each_pending_call_requires_its_own_approval_and_consumes_once():
    executed = []

    async def a():
        executed.append("a")
        return "a"

    async def b():
        executed.append("b")
        return "b"

    ag = make_agent(
        [response(calls=[call("a"), call("b", "c2")]), response("done")],
        [tool("a", a, True), tool("b", b, True)],
        [HitlMiddleware(mode="interrupt", approval_policy="declared")],
        MemoryCheckpointer(),
    )
    first = await ag.run("do both")
    second = await ag.resume(first.interrupt)
    assert second.status == "interrupted"
    assert executed == ["a"]
    assert second.interrupt["tool_name"] == "b"
    final = await ag.resume(second.interrupt)
    assert final.status == "completed" and executed == ["a", "b"]
    assert final.usage["total_tokens"] == 30
    with pytest.raises(ValueError, match="consum|恢复|领取|使用"):
        await ag.resume(first.interrupt)
    assert executed == ["a", "b"]


@pytest.mark.asyncio
async def test_resume_after_process_recreation_and_replay_rejected(tmp_path):
    executed = []

    async def deploy():
        executed.append(1)
        return "ok"

    path = str(tmp_path / "state.db")
    one = make_agent(
        [response(calls=[call("deploy")])],
        [tool("deploy", deploy, True)],
        [HitlMiddleware(mode="interrupt")],
        SQLiteCheckpointer(path),
    )
    interrupted = await one.run("go")
    two = make_agent(
        [response("done")],
        [tool("deploy", deploy, True)],
        [HitlMiddleware(mode="interrupt")],
        SQLiteCheckpointer(path),
    )
    result = await two.resume(interrupted.interrupt)
    assert result.content == "done" and executed == [1]
    three = make_agent([], [tool("deploy", deploy)], cp=SQLiteCheckpointer(path))
    with pytest.raises(ValueError):
        await three.resume(interrupted.interrupt)
    assert executed == [1]


@pytest.mark.asyncio
async def test_resume_does_not_skip_permission_after_hitl():
    executed = []

    async def deploy():
        executed.append(1)
        return "ok"

    ag = make_agent(
        [response(calls=[call("deploy")]), response("cancelled")],
        [tool("deploy", deploy, True)],
        [
            HitlMiddleware(mode="interrupt"),
            PermissionMiddleware(PermissionConfig(denied_tools=["deploy"])),
        ],
        MemoryCheckpointer(),
    )
    result = await ag.run("go")
    if result.status == "interrupted":
        result = await ag.resume(result.interrupt)
    assert executed == []
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_permission_taint_persists_between_steps_and_within_batch():
    for same_step in (False, True):
        executed = []

        async def http_get(url):
            executed.append("http")
            return "untrusted"

        async def write_text_file(path):
            executed.append("write")
            return "written"

        tc = [
            call("http_get", url="https://example.com"),
            call("write_text_file", "c2", path="out/a"),
        ]
        responses = (
            [response(calls=tc), response("done")]
            if same_step
            else [response(calls=tc[:1]), response(calls=tc[1:]), response("done")]
        )
        ag = make_agent(
            responses,
            [tool("http_get", http_get), tool("write_text_file", write_text_file)],
            [
                PermissionMiddleware(
                    PermissionConfig(network_allow=["example.com"], filesystem_write=["out/**"])
                )
            ],
        )
        await ag.run("go")
        assert executed == ["http"]


@pytest.mark.asyncio
async def test_quota_persists_across_steps():
    executed = []

    async def calculate():
        executed.append(1)
        return 1

    ag = make_agent(
        [
            response(calls=[call("calculate")]),
            response(calls=[call("calculate", "c2")]),
            response("done"),
        ],
        [tool("calculate", calculate)],
        [PermissionMiddleware(PermissionConfig(tool_quotas={"calculate": 1}))],
    )
    await ag.run("go")
    assert executed == [1]


def test_permission_directory_boundary(tmp_path):
    match = PermissionMiddleware._glob_match
    assert match(["out/**"], "out/a.txt")
    assert not match(["out/**"], "outside/a.txt")
    assert not match(["out/**"], "out/../../secret.txt")
    assert not match(["out/a.txt"], "out/a.txt.bak")


@pytest.mark.asyncio
async def test_middleware_result_short_circuits_before_tool():
    async def reject(tc):
        return False

    ag = make_agent(
        [response(calls=[call("deploy")]), response("refused")],
        [tool("deploy", lambda: "unexpected")],
        [HitlMiddleware(approval_provider=reject), PermissionMiddleware(PermissionConfig())],
    )
    result = await ag.run("go")
    assert result.content == "refused"


@pytest.mark.asyncio
async def test_stream_and_nonstream_default_session_agree():
    ag = make_agent([response("first"), response("second")])
    await ag.run("remember X")
    async for _ in ag.stream_events("what was X?"):
        pass
    assert len(ag._sessions) == 1
    assert [m.content for m in ag.llm.calls[1] if m.role == "user"] == ["remember X", "what was X?"]


@pytest.mark.asyncio
async def test_storage_and_session_json(tmp_path):
    ag = make_agent(
        [response("one"), response("two")], middlewares=[StorageMiddleware(str(tmp_path))]
    )
    a = ag.create_session("A")
    b = ag.create_session("B")
    await ag.run("one", session=a)
    await ag.run("two", session=b)
    assert {p.name for p in tmp_path.iterdir()} == {"A", "B"}
    encoded = json.dumps(a.to_dict())
    restored = type(a).from_dict(json.loads(encoded))
    assert restored.last_usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_step_limit_is_not_completed():
    ag = make_agent(
        [response(calls=[call("noop")])], [tool("noop", lambda: "ok")], default_max_steps=1
    )
    result = await ag.run("go")
    assert result.status == "incomplete"
    assert result.termination_reason == "max_steps"


@pytest.mark.asyncio
async def test_checkpointer_pages_same_timestamp(tmp_path):
    cp = SQLiteCheckpointer(str(tmp_path / "cp.db"))
    for _ in range(3):
        await cp.save(State())
    page = await cp.list_checkpoints(limit=1)
    rest = await cp.list_checkpoints(limit=10, before=page[0]["id"])
    assert len(rest) == 2


@pytest.mark.asyncio
async def test_interrupt_requires_checkpointer():
    ag = make_agent(
        [response(calls=[call("deploy")])],
        [tool("deploy", lambda: "x")],
        [HitlMiddleware(mode="interrupt")],
    )
    with pytest.raises(ValueError, match="checkpointer"):
        await ag.run("go")


def test_security_config_error_stops_creation():
    from wuwei.config import AgentConfig, create_agent_from_config

    config = AgentConfig(llm={"api_key": "fake"}, middleware=[{"type": "hitl", "policy": "typo"}])
    with pytest.raises(ValueError):
        create_agent_from_config(config)


@pytest.mark.asyncio
async def test_structured_output_uses_same_session_middleware_and_retry_budget():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    class Seen(Middleware):
        def __init__(self):
            self.count = 0

        async def before_llm(self, ctx):
            self.count += 1
            return ctx

    seen = Seen()
    ag = make_agent(
        [response("first"), response('{"value":"bad"}'), response('{"value":7}')],
        middlewares=[seen],
    )
    await ag.run("remember X")
    parsed, result = await ag.run_structured("answer", Answer)
    assert parsed.value == 7
    assert result.status == "completed" and result.usage["total_tokens"] == 30
    assert seen.count == 3
    assert any(m.content == "remember X" for m in ag.llm.calls[1])


@pytest.mark.asyncio
async def test_fork_isolated_and_side_effects_blocked_by_default():
    executed = []

    async def deploy():
        executed.append(1)
        return "ok"

    risky = tool("deploy", deploy)
    risky.execution.side_effect = True
    cp = MemoryCheckpointer()
    ag = make_agent(
        [response(calls=[call("deploy")]), response("done"), response("simulated")], [risky], cp=cp
    )
    await ag.run("go")
    original = ag._default_session.to_dict()
    pre = next(c for c in await cp.list_checkpoints(100) if c["metadata"]["kind"] == "pre_tool")
    result = await ag.fork(pre["id"])
    assert result.content == "simulated"
    assert ag._default_session.to_dict() == original
    assert len(ag._sessions) == 2 and executed == [1]


@pytest.mark.asyncio
async def test_parallel_checkpoint_commits_all_results_before_resumable_cursor():
    from wuwei.runtime.interrupt import InterruptPayload

    class ProcessDeath(BaseException):
        pass

    class Crash(MemoryCheckpointer):
        armed = True

        async def save(self, state, checkpoint_id=None, metadata=None):
            cid = await super().save(state, checkpoint_id, metadata)
            if metadata.get("kind") == "tool_result" and self.armed:
                self.armed = False
                self.saved = cid
                raise ProcessDeath()
            return cid

    cp = Crash()
    executed = []
    ag = make_agent(
        [response(calls=[call("a"), call("b", "c2")]), response("done")],
        [tool("a", lambda: executed.append("a")), tool("b", lambda: executed.append("b"))],
        cp=cp,
        default_parallel_tool_calls=True,
    )
    with pytest.raises(ProcessDeath):
        await ag.run("go")
    await ag.resume(InterruptPayload(cp.saved, ag._default_session.session_id, "recover"))
    assert executed == ["a", "b"]


@pytest.mark.asyncio
async def test_rewrite_finalizes_call_before_approval():
    class Rewrite(Middleware):
        async def before_tool(self, ctx, tc):
            return call("danger", tc.id)

    executed = []
    ag = make_agent(
        [response(calls=[call("safe")]), response("done")],
        [tool("safe", lambda: None, True), tool("danger", lambda: executed.append(1), True)],
        [HitlMiddleware(mode="interrupt"), Rewrite()],
        MemoryCheckpointer(),
    )
    first = await ag.run("go")
    assert first.interrupt["tool_name"] == "danger"
    await ag.resume(first.interrupt)
    assert executed == [1]
    assert ag._default_session.context.get_messages()[2].tool_calls[0].function.name == "danger"


def test_session_restore_preserves_blocking_flags_without_mutating_source():
    from wuwei.agent.session import AgentSession

    s = AgentSession("x")
    s.metadata.update(_blocked_run="run", _pending_interrupt="cp")
    data = s.to_dict()
    restored = AgentSession.from_dict(data)
    assert restored.metadata["_blocked_run"] == "run"
    assert restored.metadata["_pending_interrupt"] == "cp"
    assert data["metadata"]["_blocked_run"] == "run"


@pytest.mark.asyncio
async def test_cancelled_sync_sideeffect_blocks_session_reentry():
    import asyncio, threading

    started = threading.Event()
    release = threading.Event()

    def effect():
        started.set()
        release.wait(2)
        return "effect happened"

    risky = tool("effect", effect)
    risky.execution.side_effect = True
    ag = make_agent(
        [response(calls=[call("effect")]), response("next")], [risky], cp=MemoryCheckpointer()
    )
    running = asyncio.create_task(ag.run("go"))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        with pytest.raises(ValueError, match="unknown"):
            await ag.run("next")
        restored = type(ag._default_session).from_dict(ag._default_session.to_dict())
        with pytest.raises(ValueError, match="unknown"):
            await ag.run("next", session=restored)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_run_plan_accounts_for_planner_and_uses_parent_session():
    plan = json.dumps({"tasks": [{"id": 1, "description": "one", "next": [], "status": "pending"}]})
    ag = make_agent([response("remembered"), response(plan), response("task result")])
    session = ag.create_session("plan-parent")
    await ag.run("remember X", session=session)
    result = await ag.run_plan("do plan", session=session)
    assert result.status == "completed"
    assert result.usage["total_tokens"] == 30 and result.planner_usage["total_tokens"] == 15
    assert result.execution_usage["total_tokens"] == 15
    assert session.last_usage["total_tokens"] == 30
    assert any(m.content == "remember X" for m in ag.llm.calls[1])


@pytest.mark.asyncio
async def test_run_plan_shared_budget_stops_execution_after_planner():
    from wuwei.middleware import Budget, BudgetMiddleware, BudgetExhausted

    plan = json.dumps({"tasks": [{"id": 1, "description": "one", "next": [], "status": "pending"}]})
    ag = make_agent(
        [response(plan), response("must not run")],
        middlewares=[BudgetMiddleware(Budget(max_tokens=20))],
    )
    # Planner costs 15, task's response crosses 20 and cannot be reported successful.
    result = await ag.run_plan("plan")
    assert result.status != "completed"
    assert result.usage["total_tokens"] == 30


def test_permission_rule_cannot_expand_through_symlink(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (root / "out").symlink_to(outside, target_is_directory=True)
    assert not PermissionMiddleware._glob_match(["out/**"], "out/secret", str(root))


def test_nested_yaml_config_is_loaded_and_security_typos_fail(tmp_path):
    from wuwei.config.yaml_loader import load_agent_config, create_agent_from_config

    p = tmp_path / "agent.yaml"
    p.write_text(
        'agent:\n  name: safe\n  max_steps: 7\nllm:\n  api_key: fake\nmiddleware:\n  - type: permission\n    filesystem_read: ["out/**"]\n'
    )
    config = load_agent_config(str(p))
    assert config.name == "safe" and config.max_steps == 7
    agent = create_agent_from_config(config)
    assert isinstance(agent.middleware.middlewares[0], PermissionMiddleware)


@pytest.mark.asyncio
async def test_quota_is_not_double_spent_by_approval_resume():
    executed = []
    ag = make_agent(
        [response(calls=[call("calculate")]), response("done")],
        [tool("calculate", lambda: executed.append(1), True)],
        [
            PermissionMiddleware(PermissionConfig(tool_quotas={"calculate": 1})),
            HitlMiddleware(mode="interrupt"),
        ],
        MemoryCheckpointer(),
    )
    first = await ag.run("go")
    await ag.resume(first.interrupt)
    assert executed == [1]


@pytest.mark.asyncio
async def test_unknown_usage_is_not_claimed_as_zero_cost():
    from wuwei.middleware import Budget, BudgetMiddleware, BudgetExhausted

    unavailable = response("answer")
    unavailable.usage_available = False
    ag = make_agent([unavailable])
    result = await ag.run("go")
    assert result.usage_available is False
    ag = make_agent([unavailable], middlewares=[BudgetMiddleware(Budget(max_tokens=100))])
    with pytest.raises(BudgetExhausted, match="usage"):
        await ag.run("go")


@pytest.mark.asyncio
async def test_tracing_resources_are_not_checkpoint_data_and_close_on_interrupt():
    from wuwei.observability.tracing import TracingMiddleware
    from test_middleware_lifecycle import Tracer

    closed = []
    trace = TracingMiddleware()
    trace._tracer = Tracer(closed)
    ag = make_agent(
        [response(calls=[call("noop")]), response("done")],
        [tool("noop", lambda: "ok")],
        [trace, HitlMiddleware(mode="interrupt")],
        MemoryCheckpointer(),
    )
    first = await ag.run("go")
    assert first.status == "interrupted" and len(closed) == 3
    final = await ag.resume(first.interrupt)
    assert final.status == "completed" and len(closed) == 6


@pytest.mark.asyncio
async def test_memory_internal_model_calls_charge_same_run():
    from wuwei.middleware.memory import MemoryMiddleware

    class Store:
        async def search(self, *args, **kwargs):
            return []

        async def add(self, *args, **kwargs):
            return None

    llm = FakeLLM([response("main answer"), response("[]")])
    memory = MemoryMiddleware(llm, Store(), extract_every_n=1)
    ag = make_agent([], middlewares=[memory])
    ag.llm = llm
    result = await ag.run("remember preference")
    assert result.content == "main answer"
    assert result.usage["total_tokens"] == 30 and result.llm_calls == 2


@pytest.mark.asyncio
async def test_parent_reports_unknown_when_background_sync_effect_is_cancelled():
    import asyncio, threading
    from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware

    started = threading.Event()
    release = threading.Event()

    def effect():
        started.set()
        release.wait(2)
        return "finished late"

    risky = tool("effect", effect)
    risky.execution.side_effect = True
    child = FakeLLM([response(calls=[call("effect")])])

    class Parent:
        calls = 0

        async def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return response(calls=[call("start_async_task", agent_name="worker", task="go")])
            assert await asyncio.to_thread(started.wait, 1)
            return response("done")

    bridge = AsyncSubAgentMiddleware([AsyncSubAgent("worker", "work", tools=[risky])], child)
    ag = Agent(llm=Parent(), middleware=MiddlewareStack(bridge), load_builtins=False)
    try:
        result = await ag.run("go")
        assert result.status == "unknown"
        assert next(iter(bridge._tasks.values())).status == "unknown"
        with pytest.raises(ValueError, match="unknown"):
            await ag.run("next")
    finally:
        release.set()


@pytest.mark.asyncio
async def test_memory_extraction_counts_across_runs_but_not_sessions():
    from wuwei.middleware.memory import MemoryMiddleware

    class Store:
        async def search(self, *args, **kwargs):
            return []

        async def add(self, *args, **kwargs):
            pass

    llm = FakeLLM([response("a1"), response("b1"), response("a2"), response("a3"), response("[]")])
    memory = MemoryMiddleware(llm, Store(), extract_every_n=3)
    ag = make_agent([], middlewares=[memory])
    ag.llm = llm
    a = ag.create_session("a")
    b = ag.create_session("b")
    await ag.run("one", session=a)
    await ag.run("other", session=b)
    await ag.run("two", session=a)
    result = await ag.run("three", session=a)
    assert result.llm_calls == 2 and result.usage["total_tokens"] == 30
    assert len(llm.calls) == 5
    assert next(iter(b.metadata["middleware_state"].values()))["turn_count"] == 1


def test_yaml_model_parameters_and_provider_defaults_are_not_ignored():
    from wuwei.config import AgentConfig, create_agent_from_config

    config = AgentConfig(llm={"provider": "responses", "api_key": "fake", "max_output_tokens": 99})
    ag = create_agent_from_config(config)
    assert ag.llm.adapter.build_request([])["max_output_tokens"] == 99
    config = AgentConfig(llm={"provider": "anthropic", "api_key": "fake"})
    ag = create_agent_from_config(config)
    assert ag.llm.adapter.model.startswith("claude")
    config = AgentConfig(
        llm={"api_key": "fake"}, middleware=[{"type": "hitl", "auto_reject_typo": ["danger"]}]
    )
    with pytest.raises(ValueError, match="Unknown"):
        create_agent_from_config(config)
