import asyncio
import json
import time
import pytest
from wuwei.tools.registry import ToolRegistry
from wuwei.middleware.ledger import ContextLedger, ledger_tools


@pytest.mark.asyncio
async def test_typed_list_and_validation():
    registry = ToolRegistry()
    called = []

    def collect(values: list[str]):
        called.append(values)

    tool = registry.register_callable(collect)
    assert tool.parameters.properties["values"]["type"] == "array"
    with pytest.raises((ValueError, TypeError)):
        await tool.invoke({"values": "bad"})
    assert called == []


@pytest.mark.asyncio
async def test_sync_handler_not_blocking():
    registry = ToolRegistry()
    tool = registry.register_callable(lambda: time.sleep(0.08))
    task = asyncio.create_task(tool.invoke())
    await asyncio.sleep(0.01)
    assert not task.done()
    await task


def test_ledger_recall_and_pin(tmp_path):
    ledger = ContextLedger(str(tmp_path), max_entries=1)
    first = ledger.put("first")
    ledger.put("second")
    assert ledger.get(first).content == "first"
    assert ledger.pin(first)
    assert ContextLedger(str(tmp_path)).get(first).pinned
    assert ledger_tools(ledger)[0].parameters.required == ["entry_id"]


@pytest.mark.asyncio
async def test_a2a_standard_request_and_error():
    import httpx
    from types import SimpleNamespace
    from wuwei.serve.a2a import build_a2a_app

    class Agent:
        async def run(self, text):
            return SimpleNamespace(
                content="approve", status="interrupted", interrupt={"checkpoint_id": "cp"}
            )

    app = build_a2a_app(Agent(), name="test", description="test")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        card = (await client.get("/.well-known/agent-card.json")).json()
        assert card["protocolVersion"] == "0.3.0"
        assert "supportedInterfaces" not in card
        response = (
            await client.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "message/send",
                    "params": {
                        "message": {
                            "kind": "message",
                            "messageId": "m",
                            "role": "user",
                            "parts": [{"kind": "text", "text": "hello"}],
                        }
                    },
                },
            )
        ).json()
        assert response["id"] == 0
        assert response["result"]["status"]["state"] == "input-required"
        assert response["result"]["metadata"]["interrupt"]["checkpoint_id"] == "cp"
        error = (await client.post("/", json={"jsonrpc": "2.0", "id": 2, "method": "bad"})).json()
        assert error["jsonrpc"] == "2.0" and error["id"] == 2 and error["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_agui_session_and_close():
    from wuwei.serve.ag_ui import ag_ui_events

    seen = []

    class Agent:
        async def stream_events(self, text, session_id=None):
            seen.append(session_id)
            try:
                yield {"type": "run_start"}
                yield {"type": "error", "data": {"message": "bad"}}
            finally:
                seen.append("closed")

    result = [event async for event in ag_ui_events(Agent(), "hello", "thread")]
    assert seen == ["thread", "closed"]
    assert result[-1]["type"] == "RUN_ERROR"


@pytest.mark.asyncio
async def test_timeout_side_effect_unknown_without_retry():
    from wuwei.tools.executor import ToolExecutor
    from wuwei.core.message import ToolCall, FunctionCall

    registry = ToolRegistry()
    calls = []

    def write():
        calls.append("start")
        time.sleep(0.05)
        calls.append("done")

    registry.register_callable(write, side_effect=True, timeout_seconds=0.005)
    message = await ToolExecutor(registry).execute_one(
        ToolCall(function=FunctionCall(name="write", arguments={}))
    )
    payload = json.loads(message.content)
    assert payload["status"] == "unknown"
    assert payload["attempts"] == 1
    await asyncio.sleep(0.08)
    assert calls == ["start", "done"]


@pytest.mark.asyncio
async def test_nested_code_requires_governance():
    from wuwei.codemode import run_code
    from wuwei.tools.executor import ToolExecutor, bind_tool_execution

    registry = ToolRegistry()
    calls = []
    registry.register_callable(lambda: calls.append("unsafe"), name="danger")
    executor = ToolExecutor(registry)
    output = json.loads(await run_code("await tools.danger()", executor))
    assert not output["ok"] and not calls

    async def governed(call):
        from wuwei.core.message import ToolMessage

        calls.append(call.function.name)
        return ToolMessage(tool_call_id=call.id, name="danger", content="denied")

    with bind_tool_execution(governed):
        output = json.loads(await run_code("await tools.danger()", executor))
    assert output["ok"] and calls == ["danger"]


@pytest.mark.asyncio
async def test_graph_reducers_stream_invoke_match():
    from wuwei.graph import StateGraph, END

    async def node(state, config):
        return {"answer": 42}

    graph = StateGraph()
    graph.add_node("n", node)
    graph.set_entry_point("n")
    graph.add_edge("n", END)
    compiled = graph.compile()
    result = await compiled.invoke()
    events = [event async for event in compiled.stream()]
    assert events[-1]["state"].metadata == result.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "unknown", "error", "cancelled"])
async def test_agui_nonstream_noncompletion_and_session(status):
    from wuwei.serve.ag_ui import ag_ui_events

    seen = []

    class Agent:
        async def run(self, text, session_id=None):
            seen.append(session_id)
            return {"content": "unfinished", "status": status}

    events = [event async for event in ag_ui_events(Agent(), "hello", "thread")]
    assert seen == ["thread"]
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == status
    assert not any(event["type"] == "RUN_FINISHED" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, expected",
    [
        ("completed", "completed"),
        ("interrupted", "input-required"),
        ("incomplete", "failed"),
        ("unknown", "failed"),
        ("cancelled", "canceled"),
    ],
)
async def test_a2a_official_03_sdk_request_card_response(status, expected):
    from importlib.metadata import version

    pytest.importorskip("a2a.types")
    if not version("a2a-sdk").startswith("0.3."):
        pytest.skip("Fixture specifically targets A2A 0.3 SDK")
    import httpx
    from a2a.types import (
        AgentCard,
        SendMessageRequest,
        SendMessageResponse,
        SendMessageSuccessResponse,
        JSONRPCErrorResponse,
    )
    from wuwei.serve.a2a import build_a2a_app

    class Agent:
        async def run(self, text):
            return {
                "content": text,
                "status": status,
                "interrupt": {"checkpoint_id": "cp"} if status == "interrupted" else None,
            }

    app = build_a2a_app(Agent(), name="typed", description="A2A 0.3 fixture")
    request = SendMessageRequest.model_validate(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "messageId": "client-message",
                    "role": "user",
                    "parts": [{"kind": "text", "text": "typed request"}],
                }
            },
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        card = AgentCard.model_validate((await client.get("/.well-known/agent-card.json")).json())
        assert card.protocol_version == "0.3.0"
        response = (
            await client.post(
                "/", json=request.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
        ).json()
        validated = SendMessageResponse.model_validate(response)
        assert isinstance(validated.root, SendMessageSuccessResponse)
        assert validated.root.id == 0
        assert validated.root.result.history[0].message_id
        assert validated.root.result.history[0].kind == "message"
        assert validated.root.result.status.state.value == expected
        if status == "interrupted":
            assert validated.root.result.metadata["interrupt"]["checkpoint_id"] == "cp"
        bad = (
            await client.post("/", json={"jsonrpc": "2.0", "id": "bad", "method": "unknown"})
        ).json()
        assert JSONRPCErrorResponse.model_validate(bad).id == "bad"


@pytest.mark.asyncio
async def test_agui_nonstream_session_object_and_exception():
    from wuwei.serve.ag_ui import ag_ui_events

    session = object()

    class Agent:
        def create_or_get_session(self, session_id):
            assert session_id == "thread"
            return session

        async def run(self, text, session=None):
            assert session is not None
            raise RuntimeError("failed")

    events = [event async for event in ag_ui_events(Agent(), "hello", "thread")]
    assert [event["type"] for event in events] == ["RUN_STARTED", "RUN_ERROR"]
    assert events[-1]["code"] == "RuntimeError"


@pytest.mark.asyncio
async def test_mcp_run_task_keeps_recovery_payload_and_explicit_session():
    from wuwei.serve.mcp_server import build_mcp_server
    from tests.test_runtime_hardening import make_agent, response, call, tool
    from wuwei.middleware import HitlMiddleware
    from wuwei.graph.checkpoint import MemoryCheckpointer

    agent = make_agent(
        [response(calls=[call("deploy")])],
        [tool("deploy", lambda: "ok", True)],
        [HitlMiddleware(mode="interrupt")],
        MemoryCheckpointer(),
    )
    server = build_mcp_server(agent)
    result = await server.call_tool("run_task", {"task": "go", "session_id": "explicit"})
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "interrupted"
    assert payload["interrupt"]["session_id"] == "explicit"
    assert payload["usage_available"] is True
