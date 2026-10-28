import json

import pytest

from tests.test_runtime_hardening import FakeLLM, response
from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware
from wuwei.graph.state import State
from wuwei.middleware import Middleware, MiddlewareStack
from wuwei.middleware.base import MiddlewareContext
from wuwei.planning import PlanExecutor, Task, TaskList


@pytest.mark.asyncio
async def test_subagent_inherits_parent_wrap_and_uses_runner():
    seen = []

    class Guard(Middleware):
        async def wrap_model_call(self, ctx, messages, tools, next_handler):
            seen.append(True)
            return await next_handler(messages, tools)

    bridge = SubAgentMiddleware([SubAgent("worker", "work")], FakeLLM([response("done")]))
    ctx = MiddlewareContext(state=State(), config={"middleware": MiddlewareStack(Guard(), bridge)})
    msg = await bridge.handle_task_tool("task_worker", {"task": "work"}, ctx)
    result = json.loads(msg.content)
    assert seen == [True]
    assert result["status"] == "completed" and result["usage"]["total_tokens"] == 15
    assert bridge.is_tool_handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tasks",
    [
        [
            Task(id=1, description="a", next=[], status="pending"),
            Task(id=1, description="b", next=[], status="pending"),
        ],
        [Task(id=1, description="a", next=[2], status="pending")],
        [Task(id=1, description="a", next=[1], status="pending")],
    ],
)
async def test_plan_rejects_invalid_dag(tasks):
    with pytest.raises(ValueError):
        await PlanExecutor(FakeLLM([])).execute(TaskList(tasks=tasks))


@pytest.mark.asyncio
async def test_plan_incomplete_does_not_succeed_or_retry():
    llm = FakeLLM([response(calls=[])])
    # A worker with zero steps cannot complete the task.
    result = await PlanExecutor(llm, max_steps_per_task=0).execute(
        [
            Task(id=1, description="a", next=[2], status="pending"),
            Task(id=2, description="b", next=[], status="pending"),
        ]
    )
    assert result.tasks[0].status != "completed"
    assert result.tasks[1].status == "blocked"


@pytest.mark.asyncio
async def test_nested_interrupt_resume_does_not_replay_child_tool():
    from tests.test_runtime_hardening import call, tool
    from wuwei import Agent
    from wuwei.graph.checkpoint import MemoryCheckpointer
    from wuwei.middleware import HitlMiddleware

    done = []

    async def deploy():
        done.append("deployed")
        return "ok"

    child_llm = FakeLLM([response(calls=[call("deploy")]), response("child done")])
    bridge = SubAgentMiddleware(
        [SubAgent("worker", "work", tools=[tool("deploy", deploy, True)], model=child_llm)], None
    )
    parent = Agent(
        llm=FakeLLM([response(calls=[call("task_worker", task="go")]), response("parent done")]),
        middleware=MiddlewareStack(
            bridge, HitlMiddleware(mode="interrupt", approval_policy="declared")
        ),
        checkpointer=MemoryCheckpointer(),
        load_builtins=False,
    )
    first = await parent.run("go")
    assert first.status == "interrupted" and done == []
    bridge.sub_agents["worker"].max_steps = 0
    final = await parent.resume(first.interrupt)
    assert final.status == "completed" and done == ["deployed"]
    assert final.usage["total_tokens"] == 60


@pytest.mark.asyncio
async def test_plan_interrupted_retains_payload_no_retry():
    from tests.test_runtime_hardening import call, tool
    from wuwei.graph.checkpoint import MemoryCheckpointer
    from wuwei.middleware import HitlMiddleware
    from wuwei.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool("deploy", lambda: "ok", True))
    llm = FakeLLM([response(calls=[call("deploy")])])
    result = await PlanExecutor(
        llm,
        tools=registry,
        checkpointer=MemoryCheckpointer(),
        middleware=MiddlewareStack(HitlMiddleware(mode="interrupt", approval_policy="declared")),
    ).execute([Task(id=1, description="do", next=[], status="pending")])
    assert result.status == "interrupted" and result.tasks[0].interrupt["checkpoint_id"]
    assert result.usage["total_tokens"] == 15 and len(llm.calls) == 1


@pytest.mark.asyncio
async def test_async_handle_reports_usage_and_is_session_scoped():
    from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware

    bridge = AsyncSubAgentMiddleware([AsyncSubAgent("worker", "work")], FakeLLM([response("done")]))
    ctx = MiddlewareContext(
        state=State(metadata={"session_id": "owner"}),
        metadata={
            "session_id": "owner",
            "usage_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "llm_calls": 0,
            "latency_ms": 0,
        },
        config={"middleware": MiddlewareStack(bridge)},
    )
    started = json.loads(await bridge._start_task("worker", "do", ctx))
    await bridge._tasks[started["task_id"]]._task
    payload = json.loads(await bridge._check_task(started["task_id"]))
    own = await bridge._handle_async_tool(
        "check_async_task", {"task_id": started["task_id"]}, ctx, "own"
    )
    assert json.loads(own.content)["ok"]
    assert payload["status"] == "completed" and payload["usage"]["total_tokens"] == 15
    assert ctx.metadata["usage_total"]["total_tokens"] == 15
    other = MiddlewareContext(state=State(), metadata={"session_id": "other"})
    rejected = await bridge._handle_async_tool(
        "check_async_task", {"task_id": started["task_id"]}, other, "check"
    )
    assert not json.loads(rejected.content)["ok"]
    assert bridge.is_tool_handler


@pytest.mark.asyncio
async def test_multi_agent_uses_governed_leader_calls():
    from wuwei import Agent
    from wuwei.agent.multi_agent import MultiAgentGraph

    seen = []

    class Guard(Middleware):
        async def wrap_model_call(self, ctx, messages, tools, next_handler):
            seen.append(True)
            return await next_handler(messages, tools)

    leader = Agent(
        llm=FakeLLM(
            [
                response("[DECISION]\nassignments:\n  - member: worker\n    task: do\n[/DECISION]"),
                response("final"),
            ]
        ),
        load_builtins=False,
        middleware=MiddlewareStack(Guard()),
    )
    worker = Agent(llm=FakeLLM([response("worked")]), load_builtins=False)
    graph = MultiAgentGraph().set_leader(leader).add_worker("worker", worker, "worker")
    assert await graph.run("do") == "final"
    assert seen == [True, True, True]


@pytest.mark.asyncio
async def test_plan_resume_only_runs_remaining_tasks():
    from tests.test_runtime_hardening import call, tool
    from wuwei.graph.checkpoint import MemoryCheckpointer
    from wuwei.middleware import HitlMiddleware
    from wuwei.tools import ToolRegistry

    done = []

    async def deploy():
        done.append("deploy")
        return "ok"

    registry = ToolRegistry()
    registry.register(tool("deploy", deploy, True))
    llm = FakeLLM([response(calls=[call("deploy")]), response("approved"), response("downstream")])
    executor = PlanExecutor(
        llm,
        tools=registry,
        checkpointer=MemoryCheckpointer(),
        middleware=MiddlewareStack(HitlMiddleware(mode="interrupt", approval_policy="declared")),
    )
    plan = TaskList(
        tasks=[
            Task(id=1, description="first", next=[2], status="pending"),
            Task(id=2, description="next", next=[], status="pending"),
        ]
    )
    first = await executor.execute(plan)
    assert first.status == "interrupted"
    final = await executor.resume(plan, 1)
    assert final.status == "completed" and done == ["deploy"]
    assert final.usage["total_tokens"] == 45 and final.llm_calls == 3


@pytest.mark.asyncio
async def test_async_resume_keeps_handle_and_usage():
    from tests.test_runtime_hardening import call, tool
    from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware
    from wuwei.graph.checkpoint import MemoryCheckpointer
    from wuwei.middleware import HitlMiddleware

    done = []

    async def deploy():
        done.append("deploy")
        return "ok"

    bridge = AsyncSubAgentMiddleware(
        [AsyncSubAgent("worker", "work", tools=[tool("deploy", deploy, True)])],
        FakeLLM([response(calls=[call("deploy")]), response("done")]),
    )
    ctx = MiddlewareContext(
        state=State(metadata={"session_id": "owner"}),
        metadata={
            "session_id": "owner",
            "usage_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "llm_calls": 0,
            "latency_ms": 0,
        },
        config={
            "middleware": MiddlewareStack(
                bridge, HitlMiddleware(mode="interrupt", approval_policy="declared")
            ),
            "checkpointer": MemoryCheckpointer(),
        },
    )
    started = json.loads(await bridge._start_task("worker", "do", ctx))
    tid = started["task_id"]
    await bridge._tasks[tid]._task
    assert bridge._tasks[tid].status == "interrupted"
    msg = await bridge._handle_async_tool(
        "resume_async_task", {"task_id": tid, "approved": True}, ctx, "resume"
    )
    assert json.loads(msg.content)["task_id"] == tid
    await bridge._tasks[tid]._task
    assert bridge._tasks[tid].status == "completed" and done == ["deploy"]
    assert ctx.metadata["usage_total"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_planner_output_is_new_pending_work():
    from wuwei.planning import Planner

    llm = FakeLLM(
        [
            response(
                json.dumps(
                    {
                        "tasks": [
                            {"id": 1, "description": "first", "next": [], "status": "in_progress"}
                        ]
                    }
                )
            ),
            response("done"),
        ]
    )
    tasks = await Planner(llm).plan_task("goal")
    assert tasks[0].status == "pending"
    result = await PlanExecutor(llm).execute(tasks)
    assert result.status == "completed" and len(llm.calls) == 2


@pytest.mark.asyncio
async def test_team_failure_cancels_and_awaits_siblings():
    import asyncio

    from wuwei import Agent
    from wuwei.agent.multi_agent import MultiAgentGraph

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowLLM:
        async def generate(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class FailingLLM:
        async def generate(self, *args, **kwargs):
            await started.wait()
            raise RuntimeError("member failed")

    leader = Agent(
        llm=FakeLLM(
            [
                response(
                    "[DECISION]\nassignments:\n  - member: slow\n    task: work\n"
                    "  - member: fail\n    task: work\n[/DECISION]"
                )
            ]
        ),
        load_builtins=False,
    )
    graph = MultiAgentGraph().set_leader(leader)
    graph.add_worker("slow", Agent(llm=SlowLLM(), load_builtins=False), "slow")
    graph.add_worker("fail", Agent(llm=FailingLLM(), load_builtins=False), "fail")
    with pytest.raises(RuntimeError, match="member failed") as error:
        await graph.run("go")
    assert cancelled.is_set()
    assert {entry["member"]: entry["status"] for entry in error.value.delegation_outcomes} == {
        "slow": "unknown",
        "fail": "failed",
    }


@pytest.mark.asyncio
async def test_same_session_can_cancel_running_background_task():
    import asyncio

    from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware

    started = asyncio.Event()

    class SlowLLM:
        async def generate(self, *args, **kwargs):
            started.set()
            await asyncio.sleep(60)

    bridge = AsyncSubAgentMiddleware([AsyncSubAgent("worker", "work")], SlowLLM())
    ctx = MiddlewareContext(
        state=State(metadata={"session_id": "owner"}),
        metadata={"session_id": "owner"},
        config={"middleware": MiddlewareStack(bridge)},
    )
    info = json.loads(await bridge._start_task("worker", "work", ctx))
    await started.wait()
    result = await bridge._handle_async_tool(
        "cancel_async_task", {"task_id": info["task_id"]}, ctx, "c"
    )
    assert json.loads(result.content)["status"] == "cancelled"
    assert bridge._tasks[info["task_id"]]._task.done()


@pytest.mark.asyncio
async def test_background_cancel_exposes_unknown_thread_sideeffect():
    import asyncio
    import threading

    from tests.test_runtime_hardening import call, tool
    from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware

    started, release = threading.Event(), threading.Event()

    def effect():
        started.set()
        release.wait(2)
        return "effect"

    bridge = AsyncSubAgentMiddleware(
        [AsyncSubAgent("worker", "work", tools=[tool("effect", effect)])],
        FakeLLM([response(calls=[call("effect")])]),
    )
    ctx = MiddlewareContext(
        state=State(metadata={"session_id": "owner"}),
        metadata={
            "session_id": "owner",
            "usage_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "llm_calls": 0,
            "latency_ms": 0,
        },
        config={"middleware": MiddlewareStack(bridge)},
    )
    info = json.loads(await bridge._start_task("worker", "work", ctx))
    try:
        while not started.is_set():
            await asyncio.sleep(0.001)
        await bridge._cancel_task(info["task_id"])
        handle = json.loads(await bridge._check_task(info["task_id"]))
        assert handle["status"] == "unknown" and handle["reconciliation"]["run_id"]
    finally:
        release.set()
