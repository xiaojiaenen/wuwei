import pytest
from wuwei.runtime.contracts import RunContext, RunState
from wuwei.tools import ToolRegistry, ToolExecutor, ToolOutcome
from wuwei.core.message import ToolCall, FunctionCall
from wuwei.runtime.context import current_context
from wuwei.middleware.base import MiddlewareContext
from wuwei.graph.state import State


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"status": "unknown"},
        '{"status":"unknown"}',
        '{"ok":false}',
        {"nested": {"status": "unknown"}},
    ],
)
async def test_business_content_not_execution_control(body):
    registry = ToolRegistry()
    registry.register_callable(lambda: body, name="lookup")
    call = ToolCall(id="x", function=FunctionCall(name="lookup", arguments={}))
    message = await ToolExecutor(registry).execute_one(call)
    from wuwei.runtime.agent_runner import AgentRunner

    converted = AgentRunner._to_core_tool_message(message, call)
    assert converted.status == "success"
    assert not converted.metadata.get("tool_outcome", {}).get("status") == "unknown"


@pytest.mark.asyncio
async def test_explicit_unknown_envelope():
    registry = ToolRegistry()
    registry.register_callable(
        lambda: ToolOutcome(status="unknown", content="external timeout"), name="write"
    )
    message = await ToolExecutor(registry).execute_one(
        ToolCall(id="x", function=FunctionCall(name="write", arguments={}))
    )
    assert message.metadata["tool_outcome"]["status"] == "unknown"
    assert message.status == "error"


@pytest.mark.asyncio
async def test_injected_context_not_public_and_not_spoofable():
    registry = ToolRegistry()

    def lookup(value: str, ctx: RunContext[dict]):
        return ctx.deps["tenant"] + value

    tool = registry.register_callable(lookup)
    assert "ctx" not in tool.to_schema()["function"]["parameters"]["properties"]
    token = current_context.set(
        MiddlewareContext(state=State(), config={"run_context": RunContext({"tenant": "a"})})
    )
    try:
        assert await tool.invoke({"value": "b"}) == "ab"
        with pytest.raises(ValueError, match="Injected context"):
            await tool.invoke({"value": "b", "ctx": {}})
    finally:
        current_context.reset(token)


def test_run_state_v1_migration_preserves_extensions_rejects_future():
    assert RunState.model_validate({"schema_version": 1, "custom": {"x": 3}}).model_dump()[
        "custom"
    ] == {"x": 3}
    assert RunState.model_validate({}).schema_version == 2
    with pytest.raises(ValueError, match="Unsupported"):
        RunState.model_validate({"schema_version": 3})


@pytest.mark.asyncio
async def test_unknown_reconciliation_is_audited_and_never_replays():
    from wuwei.agent.agent import Agent
    from wuwei.graph.checkpoint import MemoryCheckpointer
    from wuwei.llm import LLMResponse
    from wuwei.core.message import AIMessage

    class LLM:
        calls = 0

        async def generate(self, messages, tools=None, stream=False):
            self.calls += 1
            msg = (
                AIMessage(content="done")
                if self.calls > 1
                else AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(id="write1", function=FunctionCall(name="write", arguments={}))
                    ],
                )
            )
            return LLMResponse(
                message=msg,
                finish_reason="stop" if self.calls > 1 else "tool_calls",
                model="fake",
                usage={},
            )

    count = []
    registry = ToolRegistry()

    def write():
        count.append(1)
        return ToolOutcome(status="unknown", content="pending")

    registry.register_callable(write)
    cp = MemoryCheckpointer()
    agent = Agent(llm=LLM(), tools=registry.list_tools(), load_builtins=False, checkpointer=cp)
    session = agent.create_session()
    result = await agent.run("go", session=session)
    assert result.status == "unknown"
    head = cp._heads[session.metadata["_blocked_run"]]
    with pytest.raises(ValueError, match="evidence"):
        await agent.reconcile(
            head,
            tool_call_id="write1",
            outcome=ToolOutcome(content="confirmed"),
            operator="operator",
            evidence="",
        )
    result = await agent.reconcile(
        head,
        tool_call_id="write1",
        outcome=ToolOutcome(content="confirmed"),
        operator="operator",
        evidence="remote receipt 1",
    )
    assert result.status == "completed"
    assert count == [1]
    with pytest.raises(ValueError):
        await agent.reconcile(
            head,
            tool_call_id="write1",
            outcome=ToolOutcome(content="confirmed"),
            operator="operator",
            evidence="remote receipt 1",
        )


@pytest.mark.asyncio
async def test_structured_tool_mode_captures_without_execution():
    from pydantic import BaseModel
    from wuwei.agent.agent import Agent
    from wuwei.llm import LLMResponse
    from wuwei.llm.output import OutputSpec, STRUCTURED_TOOL_NAME
    from wuwei.core.message import AIMessage

    class Answer(BaseModel):
        value: int

    class LLM:
        async def generate(self, messages, tools=None, stream=False, output_spec=None):
            assert output_spec.mode == "tool"
            return LLMResponse(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="schema",
                            function=FunctionCall(
                                name=STRUCTURED_TOOL_NAME, arguments={"value": 7}
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
                usage={},
                model="fake",
            )

    agent = Agent(llm=LLM(), load_builtins=False)
    value, result = await agent.run_structured(
        "go", Answer, output_spec=OutputSpec(schema=Answer.model_json_schema(), mode="tool")
    )
    assert value.value == 7
    assert result.status == "completed"
    assert not any(m.role == "tool" for m in agent._default_session.context.get_messages())


@pytest.mark.asyncio
async def test_output_spec_survives_interrupt_resume():
    from pydantic import BaseModel
    from wuwei.agent.agent import Agent
    from wuwei.graph.checkpoint import MemoryCheckpointer
    from wuwei.middleware.hitl import HitlMiddleware
    from wuwei.llm import LLMResponse
    from wuwei.llm.output import OutputSpec, STRUCTURED_TOOL_NAME
    from wuwei.core.message import AIMessage

    class Answer(BaseModel):
        value: int

    class LLM:
        calls = 0

        async def generate(self, messages, tools=None, stream=False, output_spec=None):
            assert output_spec.mode == "tool"
            self.calls += 1
            name = "lookup" if self.calls == 1 else STRUCTURED_TOOL_NAME
            args = {} if self.calls == 1 else {"value": 7}
            return LLMResponse(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=str(self.calls), function=FunctionCall(name=name, arguments=args)
                        )
                    ],
                ),
                finish_reason="tool_calls",
                usage={},
                model="fake",
            )

    registry = ToolRegistry()
    registry.register_callable(lambda: "ok", name="lookup")
    from wuwei.middleware.stack import MiddlewareStack

    stack = MiddlewareStack()
    stack.add(HitlMiddleware(mode="interrupt"))
    agent = Agent(
        llm=LLM(),
        tools=registry.list_tools(),
        load_builtins=False,
        checkpointer=MemoryCheckpointer(),
        middleware=stack,
    )
    _, result = await agent.run_structured(
        "go", Answer, output_spec=OutputSpec(schema=Answer.model_json_schema(), mode="tool")
    )
    assert result.status == "interrupted"
    result = await agent.resume(result.interrupt)
    assert result.status == "completed"
    assert result.output == {"value": 7}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_result_and_events_expose_final_parts_provider_metadata(stream):
    from wuwei.agent.agent import Agent
    from wuwei.llm import LLMResponse, LLMResponseChunk
    from wuwei.core.message import AIMessage, TextPart, CitationPart

    parts = [
        TextPart(text="answer"),
        CitationPart(url="https://example.test/source", title="Source"),
    ]
    metadata = {"provider_output": [{"type": "future_item", "opaque": 1}], "refusal": None}

    class LLM:
        async def generate(self, messages, tools=None, stream=False):
            if not stream:
                return LLMResponse(
                    message=AIMessage(content="answer", content_parts=parts, metadata=metadata),
                    finish_reason="stop",
                    usage={},
                    model="fake",
                )

            async def chunks():
                yield LLMResponseChunk(
                    content="answer", content_parts=parts, metadata=metadata, finish_reason="stop"
                )

            return chunks()

    agent = Agent(llm=LLM(), load_builtins=False)
    if stream:
        events = [event async for event in agent.stream_events("go")]
        data = next(event.data for event in events if event.type == "run_end")
        assert data["content_parts"][1]["url"] == "https://example.test/source"
        assert data["metadata"]["provider_output"][0]["opaque"] == 1
    else:
        result = await agent.run("go")
        assert result.content_parts[1].url == "https://example.test/source"
        assert result.metadata["provider_output"][0]["opaque"] == 1


def test_handoff_artifact_refs_keep_owner_and_metadata():
    from wuwei.artifacts import ArtifactRef
    from wuwei.runtime.handoff import DelegationInput

    ref = ArtifactRef(id="a" * 32, owner="tenant-a", media_type="image/png", size=3)
    restored = DelegationInput.model_validate_json(
        DelegationInput(task="inspect", artifact_refs=[ref]).model_dump_json()
    )
    assert restored.artifact_refs == [ref]
    assert restored.artifact_refs[0].owner == "tenant-a"


@pytest.mark.asyncio
async def test_child_receives_only_selected_owned_artifact_and_typed_output(tmp_path):
    import json
    from pydantic import BaseModel
    from wuwei import ArtifactStore
    from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware
    from tests.test_runtime_hardening import FakeLLM, response

    class Answer(BaseModel):
        value: int

    store = ArtifactStore(tmp_path)
    ref = store.put(b"%PDF-file", owner="tenant-a", media_type="application/pdf")
    llm = FakeLLM([response('{"value":7}')])
    llm.artifact_store, llm.artifact_owner = store, "tenant-a"
    child = SubAgent(
        "reader", "read authorized report", artifact_keys=("report",), output_model=Answer
    )
    ctx = MiddlewareContext(state=State(), metadata={"report": ref, "unselected": "private"})
    bridge = SubAgentMiddleware([child], llm)
    result = json.loads(await bridge._run_sub_agent(child, "read", llm, ctx=ctx))
    assert result["result"] == {"value": 7}
    user = next(m for m in llm.calls[0] if m.role == "user")
    assert user.content_parts[0].artifact == ref
    assert "private" not in user.content
    llm.artifact_owner = "other-tenant"
    with pytest.raises(PermissionError):
        await bridge._run_sub_agent(child, "read", llm, ctx=ctx)
