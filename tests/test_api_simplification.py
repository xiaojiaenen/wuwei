import asyncio

import pytest

import wuwei.agent.agent as agent_module
import wuwei.agent.plan_agent as plan_agent_module
from wuwei.agent import Agent, AgentSession, PlanAgent
from wuwei.llm import (
    AgentEvent,
    AgentRunResult,
    FunctionCall,
    LLMResponse,
    LLMResponseChunk,
    Message,
    ToolCall,
)
from wuwei.memory.storage import FileStorage
from wuwei.planning import PlanRunResult, Task
from wuwei.runtime.agent_runner import AgentRunner
from wuwei.runtime.hitl import ApprovalPolicy
from wuwei.runtime.hooks import HookManager, RuntimeHook
from wuwei.runtime.storage_hook import StorageHook
from wuwei.tools import Tool, ToolExecutor, ToolParameters, ToolRegistry, ToolRetryPolicy


def test_tool_registry_from_builtin_registers_time_tools() -> None:
    registry = ToolRegistry.from_builtin(["time"])

    tool = registry.get("get_now")

    assert tool is not None
    assert tool.name == "get_now"


def test_tool_registry_from_builtin_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="未知的内置工具"):
        ToolRegistry.from_builtin(["missing"])


def test_tool_registry_register_callable_registers_function() -> None:
    registry = ToolRegistry()

    async def get_weather(city: str) -> dict:
        return {"city": city, "condition": "sunny"}

    tool = registry.register_callable(get_weather, description="查询天气")

    assert tool.name == "get_weather"
    assert tool.description == "查询天气"
    assert registry.get("get_weather") == tool
    assert tool.parameters.required == ["city"]
    assert tool.parameters.properties["city"]["type"] == "string"


def test_tool_registry_register_callable_accepts_execution_policy() -> None:
    registry = ToolRegistry()

    def save_note(content: str) -> dict:
        return {"content": content}

    tool = registry.register_callable(
        save_note,
        timeout_seconds=2,
        side_effect=True,
        requires_approval=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_seconds=0),
    )

    assert tool.execution.timeout_seconds == 2
    assert tool.execution.side_effect is True
    assert tool.execution.requires_approval is True
    assert tool.execution.retry_policy.max_attempts == 3


def test_approval_policy_uses_tool_requires_approval_metadata() -> None:
    registry = ToolRegistry()

    @registry.tool(description="保存笔记", requires_approval=True)
    def save_note(content: str) -> dict:
        return {"content": content}

    tool = registry.get("save_note")
    tool_call = ToolCall(
        id="call-approval",
        type="function",
        function=FunctionCall(name="save_note", arguments={"content": "hello"}),
    )

    assert tool is not None
    assert ApprovalPolicy().requires_tool_approval(
        tool_call,
        session=AgentSession(session_id="approval-session"),
        tool=tool,
    )


@pytest.mark.asyncio
async def test_tool_executor_applies_timeout_and_retry_policy() -> None:
    registry = ToolRegistry()
    attempts = 0

    @registry.tool(
        description="不稳定工具",
        timeout_seconds=1,
        retry_policy=ToolRetryPolicy(max_attempts=2, backoff_seconds=0),
    )
    async def flaky_tool() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("temporary failure")
        return {"ok": True, "attempts": attempts}

    message = await ToolExecutor(registry).execute_one(
        ToolCall(
            id="call-retry",
            type="function",
            function=FunctionCall(name="flaky_tool", arguments={}),
        )
    )

    assert attempts == 2
    assert message.content is not None
    assert "temporary failure" not in message.content
    assert "attempts" in message.content

    @registry.tool(description="慢工具", timeout_seconds=0.01)
    async def slow_tool() -> dict:
        await asyncio.sleep(1)
        return {"ok": True}

    timeout_message = await ToolExecutor(registry).execute_one(
        ToolCall(
            id="call-timeout",
            type="function",
            function=FunctionCall(name="slow_tool", arguments={}),
        )
    )
    assert timeout_message.content is not None
    assert "ToolTimeout" in timeout_message.content


def test_agent_from_env_builds_agent_with_builtin_and_callable_tools(monkeypatch) -> None:
    dummy_llm = object()

    def fake_from_env(**kwargs):
        assert kwargs["env_prefix"] == "WUWEI"
        return dummy_llm

    monkeypatch.setattr(agent_module.LLMGateway, "from_env", staticmethod(fake_from_env))

    async def get_weather(city: str) -> dict:
        return {"city": city}

    agent = Agent.from_env(
        builtin_tools=["time"],
        tools=[get_weather],
        system_prompt="test prompt",
        max_steps=3,
        parallel_tool_calls=True,
        env_prefix="WUWEI",
    )

    assert agent.llm is dummy_llm
    assert agent.default_system_prompt == "test prompt"
    assert agent.default_max_steps == 3
    assert agent.default_parallel_tool_calls is True
    assert agent.tool_registry.get("get_now") is not None
    assert agent.tool_registry.get("get_weather") is not None


def test_agent_from_env_accepts_tool_instances(monkeypatch) -> None:
    dummy_llm = object()
    monkeypatch.setattr(agent_module.LLMGateway, "from_env", staticmethod(lambda **_: dummy_llm))

    tool = Tool(
        name="echo",
        description="echo input",
        parameters=ToolParameters(
            properties={"text": {"type": "string", "description": "参数 text"}},
            required=["text"],
        ),
        handler=lambda text: text,
    )

    agent = Agent.from_env(tools=[tool])

    assert agent.llm is dummy_llm
    assert agent.tool_registry.get("echo") == tool


class FakeTextStreamLLM:
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        assert stream is True

        async def iterator():
            yield LLMResponseChunk(content="你好")
            yield LLMResponseChunk(
                content="世界",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            )

        return iterator()


class FakeReasoningStreamLLM:
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        assert stream is True

        async def iterator():
            yield LLMResponseChunk(content="", reasoning_content="先分析")
            yield LLMResponseChunk(content="", reasoning_content="再确认")
            yield LLMResponseChunk(
                content="最终回答",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 4,
                    "total_tokens": 9,
                },
            )

        return iterator()


class FakeToolStreamLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        assert stream is True
        self.calls += 1

        async def first_pass():
            yield LLMResponseChunk(
                content="",
                tool_calls_complete=[
                    ToolCall(
                        id="call-1",
                        type="function",
                        function=FunctionCall(name="get_weather", arguments={"city": "北京"}),
                    )
                ],
                finish_reason="tool_calls",
                usage={
                    "prompt_tokens": 6,
                    "completion_tokens": 2,
                    "total_tokens": 8,
                },
            )

        async def second_pass():
            yield LLMResponseChunk(
                content="天气查询完成",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                },
            )

        return first_pass() if self.calls == 1 else second_pass()


class FakeTextLLM:
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        assert stream is False
        return LLMResponse(
            message=Message(role="assistant", content="直接回答"),
            finish_reason="stop",
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "total_tokens": 11,
            },
            model="fake-model",
            latency_ms=123,
        )


class FakeToolLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        assert stream is False
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            type="function",
                            function=FunctionCall(name="get_weather", arguments={"city": "北京"}),
                        )
                    ],
                ),
                finish_reason="tool_calls",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                model="fake-model",
                latency_ms=50,
            )

        return LLMResponse(
            message=Message(role="assistant", content="最终回答"),
            finish_reason="stop",
            usage={
                "prompt_tokens": 8,
                "completion_tokens": 6,
                "total_tokens": 14,
            },
            model="fake-model",
            latency_ms=70,
        )


class FakePlanner:
    def __init__(self) -> None:
        self.last_usage = {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        }
        self.last_latency_ms = 30
        self.last_llm_calls = 1

    async def plan_task(self, goal: str) -> list[Task]:
        return [
            Task(
                id=1,
                description=f"执行: {goal}",
                next=[],
                status="pending",
            )
        ]


class EventCaptureHook(RuntimeHook):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


class LegacyToolHook(RuntimeHook):
    def __init__(self) -> None:
        self.before_tool_called = False
        self.after_tool_called = False

    async def before_tool(self, session, tool_call, *, step: int, task=None) -> None:
        self.before_tool_called = True

    async def after_tool(self, session, tool_call, tool_message, *, step: int, task=None) -> None:
        self.after_tool_called = True


@pytest.mark.asyncio
async def test_agent_runner_stream_events_yields_text_and_done() -> None:
    registry = ToolRegistry()
    runner = AgentRunner(
        llm=FakeTextStreamLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-text"),
    )

    events = [event async for event in runner.stream_events("你好")]

    assert [event.type for event in events] == ["text_delta", "text_delta", "done"]
    assert events[0].data == {"content": "你好"}
    assert events[1].data == {"content": "世界"}
    assert events[2].session_id == "session-text"
    assert events[2].data["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    assert runner.session.context.get_last_message().content == "你好世界"
    assert runner.session.last_usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    assert runner.session.last_llm_calls == 1


@pytest.mark.asyncio
async def test_agent_runner_stream_events_yields_reasoning_delta() -> None:
    registry = ToolRegistry()
    runner = AgentRunner(
        llm=FakeReasoningStreamLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-reasoning"),
    )

    events = [event async for event in runner.stream_events("你好")]

    assert [event.type for event in events] == [
        "reasoning_delta",
        "reasoning_delta",
        "text_delta",
        "done",
    ]
    assert events[0].data == {"content": "先分析"}
    assert events[1].data == {"content": "再确认"}
    assert events[2].data == {"content": "最终回答"}
    assert runner.session.context.get_last_message().content == "最终回答"
    assert runner.session.context.get_last_message().reasoning_content == "先分析再确认"


@pytest.mark.asyncio
async def test_agent_run_stream_yields_reasoning_chunks() -> None:
    registry = ToolRegistry()
    runner = AgentRunner(
        llm=FakeReasoningStreamLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-reasoning-stream"),
    )

    stream = await runner.run("你好", stream=True)
    chunks = [chunk async for chunk in stream]

    assert [chunk.reasoning_content for chunk in chunks] == ["先分析", "再确认", None]
    assert [chunk.content for chunk in chunks] == ["", "", "最终回答"]
    assert runner.session.context.get_last_message().reasoning_content == "先分析再确认"


@pytest.mark.asyncio
async def test_storage_hook_persists_streaming_assistant_message(tmp_path) -> None:
    storage = FileStorage(tmp_path)
    registry = ToolRegistry()
    session = AgentSession(session_id="session-storage-stream")
    runner = AgentRunner(
        llm=FakeTextStreamLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=session,
        hooks=HookManager([StorageHook(storage)]),
    )

    events = [event async for event in runner.stream_events("你好")]
    stored_session = await storage.load("session-storage-stream")

    assert [event.type for event in events] == ["text_delta", "text_delta", "done"]
    assert stored_session is not None
    assert [
        (message.role, message.content) for message in stored_session.context.get_messages()
    ] == [
        ("system", "你是一个有用的助手"),
        ("user", "你好"),
        ("assistant", "你好世界"),
    ]


@pytest.mark.asyncio
async def test_agent_runner_stream_events_yields_tool_events() -> None:
    registry = ToolRegistry()

    @registry.tool(description="查询天气")
    async def get_weather(city: str) -> dict:
        return {"city": city, "condition": "sunny"}

    runner = AgentRunner(
        llm=FakeToolStreamLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-tool"),
    )

    events = [event async for event in runner.stream_events("北京天气")]

    assert [event.type for event in events] == ["tool_start", "tool_end", "text_delta", "done"]
    assert events[0].data["tool_name"] == "get_weather"
    assert events[0].data["args"] == {"city": "北京"}
    assert events[1].data["tool_name"] == "get_weather"
    assert "sunny" in events[1].data["output"]
    assert events[2].data == {"content": "天气查询完成"}
    assert events[3].data["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
    }
    assert runner.session.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
    }
    assert runner.session.last_llm_calls == 2


@pytest.mark.asyncio
async def test_agent_stream_events_supports_session_id() -> None:
    agent = Agent(
        llm=FakeTextStreamLLM(),
        tools=ToolRegistry(),
    )

    events = [event async for event in agent.stream_events("你好", session_id="session-agent")]

    assert [event.type for event in events] == ["text_delta", "text_delta", "done"]
    assert "session-agent" in agent._sessions
    assert all(event.session_id == "session-agent" for event in events)
    assert agent._sessions["session-agent"].last_usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }


@pytest.mark.asyncio
async def test_agent_run_returns_aggregated_usage_result() -> None:
    registry = ToolRegistry()
    runner = AgentRunner(
        llm=FakeTextLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-run"),
    )

    result = await runner.run("你好", stream=False)

    assert isinstance(result, AgentRunResult)
    assert result.content == "直接回答"
    assert result.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "total_tokens": 11,
    }
    assert result.latency_ms == 123
    assert result.llm_calls == 1
    assert runner.session.last_usage == result.usage


@pytest.mark.asyncio
async def test_agent_run_aggregates_usage_across_tool_rounds() -> None:
    registry = ToolRegistry()

    @registry.tool(description="查询天气")
    async def get_weather(city: str) -> dict:
        return {"city": city, "condition": "sunny"}

    runner = AgentRunner(
        llm=FakeToolLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-run-tool"),
    )

    result = await runner.run("北京天气", stream=False)

    assert result.content == "最终回答"
    assert result.usage == {
        "prompt_tokens": 18,
        "completion_tokens": 8,
        "total_tokens": 26,
    }
    assert result.latency_ms == 120
    assert result.llm_calls == 2
    assert runner.session.last_usage == result.usage


@pytest.mark.asyncio
async def test_agent_runner_emits_unified_runtime_events() -> None:
    registry = ToolRegistry()

    @registry.tool(description="查询天气")
    async def get_weather(city: str) -> dict:
        return {"city": city, "condition": "sunny"}

    capture = EventCaptureHook()
    runner = AgentRunner(
        llm=FakeToolLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-events"),
        hooks=HookManager([capture]),
    )

    result = await runner.run("北京天气", stream=False)
    event_types = [event.type for event in capture.events]

    assert result.content == "最终回答"
    assert event_types == [
        "run_start",
        "llm_start",
        "llm_end",
        "tool_start",
        "tool_end",
        "llm_start",
        "llm_end",
        "run_end",
    ]
    assert {event.run_id for event in capture.events} - {None}
    assert capture.events[3].data["tool_name"] == "get_weather"
    assert capture.events[4].data["tool_call_id"] == "call-1"


@pytest.mark.asyncio
async def test_hook_manager_keeps_legacy_tool_hook_signature_compatible() -> None:
    registry = ToolRegistry()

    @registry.tool(description="查询天气")
    async def get_weather(city: str) -> dict:
        return {"city": city, "condition": "sunny"}

    legacy_hook = LegacyToolHook()
    runner = AgentRunner(
        llm=FakeToolLLM(),
        tools=registry.list_tools(),
        tool_executor=ToolExecutor(registry),
        session=AgentSession(session_id="session-legacy-hook"),
        hooks=HookManager([legacy_hook]),
    )

    await runner.run("北京天气", stream=False)

    assert legacy_hook.before_tool_called is True
    assert legacy_hook.after_tool_called is True


def test_plan_agent_from_env_builds_agent(monkeypatch) -> None:
    dummy_llm = object()

    def fake_from_env(**kwargs):
        assert kwargs["env_prefix"] == "WUWEI"
        return dummy_llm

    monkeypatch.setattr(plan_agent_module.LLMGateway, "from_env", staticmethod(fake_from_env))

    async def get_weather(city: str) -> dict:
        return {"city": city}

    planner = FakePlanner()
    agent = PlanAgent.from_env(
        builtin_tools=["time"],
        tools=[get_weather],
        planner=planner,
        system_prompt="plan prompt",
        max_steps=4,
        parallel_tool_calls=True,
        env_prefix="WUWEI",
    )

    assert agent.llm is dummy_llm
    assert agent.planner is planner
    assert agent.default_system_prompt == "plan prompt"
    assert agent.default_max_steps == 4
    assert agent.default_parallel_tool_calls is True
    assert agent.tool_registry.get("get_now") is not None
    assert agent.tool_registry.get("get_weather") is not None


@pytest.mark.asyncio
async def test_plan_agent_stream_events_adds_task_context() -> None:
    agent = PlanAgent(
        llm=FakeTextStreamLLM(),
        tools=ToolRegistry(),
        planner=FakePlanner(),
    )

    events = [event async for event in agent.stream_events("测试任务", session_id="plan-session")]

    assert [event.type for event in events] == ["text_delta", "text_delta", "done"]
    assert events[0].data["task_id"] == 1
    assert events[0].data["root_session_id"] == "plan-session"
    assert events[2].data["task_description"] == "执行: 测试任务"


@pytest.mark.asyncio
async def test_plan_agent_stream_events_preserves_reasoning_delta_context() -> None:
    agent = PlanAgent(
        llm=FakeReasoningStreamLLM(),
        tools=ToolRegistry(),
        planner=FakePlanner(),
    )

    events = [event async for event in agent.stream_events("测试任务", session_id="plan-reasoning")]

    assert [event.type for event in events] == [
        "reasoning_delta",
        "reasoning_delta",
        "text_delta",
        "done",
    ]
    assert events[0].data["content"] == "先分析"
    assert events[0].data["task_id"] == 1
    assert events[0].data["root_session_id"] == "plan-reasoning"


@pytest.mark.asyncio
async def test_plan_agent_run_returns_aggregated_plan_result() -> None:
    agent = PlanAgent(
        llm=FakeTextLLM(),
        tools=ToolRegistry(),
        planner=FakePlanner(),
    )

    result = await agent.run("测试任务", session=agent.create_session(session_id="plan-run"))

    assert isinstance(result, PlanRunResult)
    assert len(result.tasks) == 1
    assert result.tasks[0].result == "直接回答"
    assert result.planner_usage == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert result.execution_usage == {
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "total_tokens": 11,
    }
    assert result.usage == {
        "prompt_tokens": 9,
        "completion_tokens": 5,
        "total_tokens": 14,
    }
    assert result.planner_latency_ms == 30
    assert result.execution_latency_ms == 123
    assert result.latency_ms == 153
    assert result.planner_llm_calls == 1
    assert result.execution_llm_calls == 1
    assert result.llm_calls == 2


@pytest.mark.asyncio
async def test_plan_agent_emits_task_runtime_events() -> None:
    capture = EventCaptureHook()
    agent = PlanAgent(
        llm=FakeTextLLM(),
        tools=ToolRegistry(),
        planner=FakePlanner(),
        hooks=HookManager([capture]),
    )

    result = await agent.run("测试任务", session=agent.create_session(session_id="plan-events"))
    task_events = [event for event in capture.events if event.type in {"task_start", "task_end"}]

    assert isinstance(result, PlanRunResult)
    assert [event.type for event in task_events] == ["task_start", "task_end"]
    assert task_events[0].data["task_id"] == 1
    assert task_events[1].data["status"] == "completed"
