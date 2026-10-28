"""wuwei.serve（协议出口：MCP / A2A / AG-UI）测试。

全部进程内完成：
- MCP 用 mcp SDK 的进程内 API（list_tools / call_tool）；
- HTTP 层用 httpx ASGITransport 直接调 ASGI app，不启动真实网络；
- agent 使用 duck-type fake（不触发真实 LLM）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from wuwei.serve import (
    ag_ui_events,
    build_a2a_app,
    build_ag_ui_app,
    build_mcp_server,
)
from wuwei.serve.a2a import extract_text_from_parts
from wuwei.serve.ag_ui import map_agent_event

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── fakes ────────────────────────────────────────────────────────────


class FakeResult:
    def __init__(self, content: str, status: str = "completed") -> None:
        self.content = content
        self.usage = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        self.status = status
        self.latency_ms = 42


class FakeAgent:
    """duck-type agent：仅实现 async run。"""

    version = "0.9.0"
    default_system_prompt = "fake prompt"
    default_max_steps = 3

    def __init__(self, content: str = "echo: {input}") -> None:
        self._content = content
        self.calls: list[str] = []

    async def run(self, user_input, session=None, stream=False):
        self.calls.append(user_input)
        return FakeResult(self._content.format(input=user_input))


class FakeStreamAgent:
    """duck-type agent：实现 async generator stream_events，产出 AgentEvent 形状的事件。"""

    def __init__(self) -> None:
        self.run_calls: list[str] = []

    async def run(self, user_input, session=None, stream=False):
        self.run_calls.append(user_input)
        return FakeResult("结果是42")

    async def stream_events(self, user_input, session=None, session_id=None):
        def ev(type_, data, run_id="run-1", step=0):
            return SimpleNamespace(
                type=type_, data=data, run_id=run_id, step=step, session_id="s1"
            )

        yield ev("run_start", {"input": user_input})
        yield ev("text_delta", {"content": "你好"})
        yield ev("text_delta", {"content": "，算一下"})
        yield ev(
            "tool_start",
            {"tool_name": "calc", "tool_call_id": "call_1", "args": {"expr": "1+1"}},
        )
        yield ev(
            "tool_end", {"tool_name": "calc", "tool_call_id": "call_1", "output": "2"}
        )
        yield ev("text_delta", {"content": "结果是2"})
        done_data = {
            "content": "结果是2",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            "latency_ms": 10,
            "llm_calls": 1,
            "status": "completed",
        }
        yield ev("done", done_data)
        yield ev("run_end", dict(done_data))


def _http_client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ── MCP 出口 ─────────────────────────────────────────────────────────


class TestMcpServer:
    async def test_build_mcp_server_lists_tools(self):
        server = build_mcp_server(FakeAgent(), name="test-agent")
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert {"run_task", "get_status"} <= names

        run_task = next(t for t in tools if t.name == "run_task")
        assert run_task.input_schema["type"] == "object"
        assert "task" in run_task.input_schema["properties"]
        assert run_task.description

    async def test_run_tool_roundtrip(self):
        agent = FakeAgent()
        server = build_mcp_server(agent)
        result = await server.call_tool("run_task", {"task": "帮我算账"})
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["content"] == "echo: 帮我算账"
        assert payload["usage"]["total_tokens"] == 12
        assert payload["status"] == "completed"
        assert agent.calls == ["帮我算账"]

    async def test_get_status_tool(self):
        agent = FakeAgent()
        server = build_mcp_server(agent, name="status-agent")
        result = await server.call_tool("get_status", {})
        payload = json.loads(result.content[0].text)
        assert payload["name"] == "status-agent"
        assert payload["default_max_steps"] == 3
        assert "tool_count" not in payload  # fake agent 无 tool_registry

    async def test_unknown_tool_raises(self):
        server = build_mcp_server(FakeAgent())
        with pytest.raises(Exception):
            await server.call_tool("nope", {})

    def test_serve_mcp_and_cli_exist(self):
        from wuwei.serve import serve_mcp

        assert callable(serve_mcp)
        from wuwei.serve.mcp_server import main

        assert callable(main)

    def test_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "wuwei.serve.mcp_server", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0
        assert "--stdio" in proc.stdout
        assert "--name" in proc.stdout


# ── A2A 出口 ─────────────────────────────────────────────────────────


class TestA2A:
    @pytest.fixture()
    def app(self):
        return build_a2a_app(
            FakeAgent(),
            name="wuwei-agent",
            description="测试用 wuwei agent",
            url="http://localhost:9000",
        )

    async def test_agent_card(self, app):
        for path in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            async with _http_client(app) as client:
                resp = await client.get(path)
            assert resp.status_code == 200
            card = resp.json()
            assert card["name"] == "wuwei-agent"
            assert card["description"] == "测试用 wuwei agent"
            assert card["url"] == "http://localhost:9000"
            assert card["version"]
            assert card["protocolVersion"]
            assert card["capabilities"] == {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            }
            assert card["skills"] and card["skills"][0]["id"]
            # This endpoint explicitly serves A2A 0.3; do not claim a mixed 1.0 card.
            assert card["protocolVersion"] == "0.3.0"
            assert card["preferredTransport"] == "JSONRPC"
            assert "supportedInterfaces" not in card

    async def test_tasks_send_direct(self, app):
        async with _http_client(app) as client:
            resp = await client.post(
                "/tasks/send",
                json={
                    "id": "task-1",
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "写首诗"}],
                    },
                },
            )
        assert resp.status_code == 200
        task = resp.json()
        assert task["id"] == "task-1"
        assert task["status"]["state"] == "completed"
        assert task["kind"] == "task"
        artifacts = task["artifacts"]
        assert artifacts and artifacts[0]["parts"]
        text = extract_text_from_parts(artifacts[0]["parts"])
        assert text == "echo: 写首诗"
        # history 中包含 agent 回复
        assert task["history"][0]["role"] == "agent"
        assert extract_text_from_parts(task["history"][0]["parts"]) == "echo: 写首诗"

    async def test_tasks_send_jsonrpc_envelope(self, app):
        async with _http_client(app) as client:
            resp = await client.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tasks/send",
                    "params": {
                        "id": "task-2",
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "hi"}],
                        },
                    },
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 7
        assert body["result"]["id"] == "task-2"
        assert body["result"]["status"]["state"] == "completed"

    async def test_tasks_send_missing_message_error(self, app):
        async with _http_client(app) as client:
            resp = await client.post("/tasks/send", json={"id": "task-3"})
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == -32602  # INVALID_PARAMS
        assert "message" in err

    async def test_tasks_send_empty_text_error(self, app):
        async with _http_client(app) as client:
            resp = await client.post(
                "/tasks/send",
                json={"id": "task-4", "message": {"role": "user", "parts": [{"kind": "text", "text": "  "}]}},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == -32602

    async def test_tasks_send_agent_failure_maps_to_internal_error(self):
        class BoomAgent:
            async def run(self, user_input, session=None, stream=False):
                raise RuntimeError("boom")

        app = build_a2a_app(BoomAgent(), name="boom", description="d")
        async with _http_client(app) as client:
            resp = await client.post(
                "/tasks/send",
                json={
                    "id": "t",
                    "message": {"role": "user", "parts": [{"kind": "text", "text": "x"}]},
                },
            )
        assert resp.status_code == 500
        err = resp.json()["error"]
        assert err["code"] == -32603  # INTERNAL_ERROR
        assert "boom" in err["message"]

    async def test_invalid_json_is_parse_error(self, app):
        async with _http_client(app) as client:
            resp = await client.post(
                "/tasks/send",
                content=b"{not json",
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == -32700  # PARSE_ERROR

    async def test_unknown_route_404(self, app):
        async with _http_client(app) as client:
            resp = await client.get("/nope")
        assert resp.status_code == 404


# ── AG-UI 出口 ───────────────────────────────────────────────────────


class TestAgUiEvents:
    async def test_full_event_mapping(self):
        agent = FakeStreamAgent()
        events = [e async for e in ag_ui_events(agent, "算一下 1+1", "thread-1")]
        types = [e["type"] for e in events]

        # 边界：RUN_STARTED 开头，RUN_FINISHED 结尾
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"
        assert types.count("RUN_FINISHED") == 1

        started = events[0]
        assert started["threadId"] == "thread-1"
        assert started["runId"] == "run-1"

        # 文本：START → CONTENT×2，工具调用打断时 END
        assert types[1] == "TEXT_MESSAGE_START"
        msg_id = started_message = events[1]["messageId"]
        assert events[2]["type"] == "TEXT_MESSAGE_CONTENT"
        assert events[2]["messageId"] == msg_id
        assert events[2]["delta"] == "你好"
        assert events[3]["delta"] == "，算一下"
        assert events[4]["type"] == "TEXT_MESSAGE_END"
        assert events[4]["messageId"] == msg_id

        # 工具调用：START → ARGS → END → RESULT
        assert types[5] == "TOOL_CALL_START"
        assert events[5]["toolCallId"] == "call_1"
        assert events[5]["toolCallName"] == "calc"
        assert types[6] == "TOOL_CALL_ARGS"
        assert json.loads(events[6]["delta"]) == {"expr": "1+1"}
        assert types[7] == "TOOL_CALL_END"
        assert types[8] == "TOOL_CALL_RESULT"
        assert events[8]["toolCallId"] == "call_1"
        assert events[8]["content"] == "2"

        # 第二段文本独立 messageId
        assert types[9] == "TEXT_MESSAGE_START"
        second_msg = events[9]["messageId"]
        assert second_msg != msg_id
        assert events[10]["delta"] == "结果是2"

        # RUN_FINISHED 携带用量（camelCase）
        finished = events[-1]
        assert finished["usage"] == {
            "promptTokens": 5,
            "completionTokens": 7,
            "totalTokens": 12,
        }

    async def test_error_event_maps_to_run_error(self):
        class ErrorStreamAgent:
            async def stream_events(self, user_input, session=None, session_id=None):
                yield SimpleNamespace(
                    type="run_start", data={"input": user_input}, run_id="r1", step=0
                )
                yield SimpleNamespace(
                    type="error",
                    data={"message": "LLM 超时", "error_type": "TimeoutError"},
                    run_id="r1",
                    step=1,
                )

        events = [e async for e in ag_ui_events(ErrorStreamAgent(), "x", "t")]
        assert [e["type"] for e in events] == ["RUN_STARTED", "RUN_ERROR"]
        assert events[1]["message"] == "LLM 超时"
        assert events[1]["code"] == "TimeoutError"

    async def test_map_agent_event_skips_internal_events(self):
        state = map_agent_event.__globals__  # noqa: F841 —— 仅确认模块可引用
        from wuwei.serve.ag_ui import _EventState

        s = _EventState("t")
        llm_start = SimpleNamespace(
            type="llm_start", data={}, run_id="r", step=1
        )
        assert map_agent_event(llm_start, s) == []

    async def test_fallback_without_stream_events(self):
        # agent 只有 run()，无 stream_events → 退化路径
        events = [e async for e in ag_ui_events(FakeAgent("hi {input}"), "hello", "t1")]
        types = [e["type"] for e in events]
        assert types == [
            "RUN_STARTED",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "RUN_FINISHED",
        ]
        assert events[2]["delta"] == "hi hello"
        assert events[4]["usage"]["totalTokens"] == 12


class TestAgUiApp:
    async def test_run_sse_stream(self):
        app = build_ag_ui_app(FakeStreamAgent())
        async with _http_client(app) as client:
            resp = await client.post(
                "/run",
                json={
                    "threadId": "thread-9",
                    "messages": [
                        {"role": "user", "content": "问题A"},
                        {"role": "assistant", "content": "答A"},
                        {"role": "user", "content": "问题B"},
                    ],
                },
                headers={"accept": "text/event-stream"},
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        payloads = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        assert payloads[0]["type"] == "RUN_STARTED"
        assert payloads[0]["threadId"] == "thread-9"
        assert payloads[-1]["type"] == "RUN_FINISHED"
        # 取最后一条用户消息运行
        contents = [
            p["delta"] for p in payloads if p["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        assert "结果是2" in "".join(contents)

    async def test_run_ndjson_stream(self):
        app = build_ag_ui_app(FakeStreamAgent())
        async with _http_client(app) as client:
            resp = await client.post(
                "/run?format=ndjson",
                json={"thread_id": "t", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
        assert events[0]["type"] == "RUN_STARTED"
        assert events[-1]["type"] == "RUN_FINISHED"

    async def test_run_missing_user_message(self):
        app = build_ag_ui_app(FakeStreamAgent())
        async with _http_client(app) as client:
            resp = await client.post("/run", json={"threadId": "t", "messages": []})
        assert resp.status_code == 400
        assert resp.json()["type"] == "RUN_ERROR"

    async def test_run_missing_thread_id(self):
        app = build_ag_ui_app(FakeStreamAgent())
        async with _http_client(app) as client:
            resp = await client.post(
                "/run", json={"messages": [{"role": "user", "content": "x"}]}
            )
        assert resp.status_code == 400

    async def test_unknown_method_405(self):
        app = build_ag_ui_app(FakeStreamAgent())
        async with _http_client(app) as client:
            resp = await client.get("/run")
        assert resp.status_code == 404


# ── 包出口 ───────────────────────────────────────────────────────────


def test_package_exports():
    import wuwei.serve as serve

    for name in (
        "build_mcp_server",
        "serve_mcp",
        "build_a2a_app",
        "ag_ui_events",
        "build_ag_ui_app",
    ):
        assert hasattr(serve, name)
