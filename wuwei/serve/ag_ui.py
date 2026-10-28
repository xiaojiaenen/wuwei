"""AG-UI（Agent-User Interaction）协议出口。

把 wuwei 的事件流（``AgentEvent``）映射为 AG-UI 1.0 风格的事件 JSON 序列：

- ``RUN_STARTED`` / ``RUN_FINISHED`` / ``RUN_ERROR``
- ``TEXT_MESSAGE_START`` / ``TEXT_MESSAGE_CONTENT`` / ``TEXT_MESSAGE_END``
- ``TOOL_CALL_START`` / ``TOOL_CALL_ARGS`` / ``TOOL_CALL_END`` / ``TOOL_CALL_RESULT``
- ``REASONING_MESSAGE_*``（reasoning_delta 映射）
- ``CUSTOM``（approval_required 等无对应 AG-UI 事件时）

字段名遵循 AG-UI 规范的 camelCase（threadId/runId/messageId/toolCallId/delta/usage）。
事件类型常量为 AG-UI 1.0 SDK 的 SCREAMING_SNAKE 形态。

参考：https://docs.ag-ui.com/concepts/events

另提供 ``build_ag_ui_app(agent)``：纯 ASGI 实现（零第三方依赖），
``POST /run`` 接收 ``{threadId, messages:[{role, content}]}``，
取最后一条用户消息运行 agent，以 SSE（默认）或 NDJSON 返回事件流。
"""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from wuwei.serve._http import HTTPError, Request, Response, make_asgi_app

__all__ = [
    "ag_ui_events",
    "build_ag_ui_app",
    "map_agent_event",
]

# AG-UI 1.0 事件类型常量
RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
TOOL_CALL_START = "TOOL_CALL_START"
TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
TOOL_CALL_END = "TOOL_CALL_END"
TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
REASONING_MESSAGE_START = "REASONING_MESSAGE_START"
REASONING_MESSAGE_CONTENT = "REASONING_MESSAGE_CONTENT"
REASONING_MESSAGE_END = "REASONING_MESSAGE_END"
CUSTOM = "CUSTOM"

PROTOCOL_VERSION = "1.0"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class _EventState:
    """单个 run 的映射状态（消息/工具调用的 id 分配与开闭配对）。"""

    def __init__(self, thread_id: str, run_id: str | None = None) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self.message_id: str | None = None
        self.reasoning_message_id: str | None = None
        self.usage: dict[str, int] = {}

    def open_text_message(self) -> dict[str, Any]:
        if self.message_id is None:
            self.message_id = _new_id("msg")
            return {
                "type": TEXT_MESSAGE_START,
                "messageId": self.message_id,
                "role": "assistant",
            }
        return {}

    def close_text_message(self) -> dict[str, Any]:
        if self.message_id is not None:
            event = {"type": TEXT_MESSAGE_END, "messageId": self.message_id}
            self.message_id = None
            return event
        return {}


def _get(event: Any, key: str, default: Any = None) -> Any:
    """从 wuwei AgentEvent（对象）或 dict 中鸭子类型取字段。"""
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _data_get(event: Any, key: str, default: Any = None) -> Any:
    data = _get(event, "data", {}) or {}
    return data.get(key, default)


def map_agent_event(event: Any, state: _EventState) -> list[dict[str, Any]]:
    """把单个 wuwei AgentEvent 映射为 0..n 个 AG-UI 事件 dict。"""
    event_type = _get(event, "type", "")
    out: list[dict[str, Any]] = []

    if event_type == "run_start":
        state.run_id = state.run_id or _get(event, "run_id") or _new_id("run")
        out.append({"type": RUN_STARTED, "threadId": state.thread_id, "runId": state.run_id})

    elif event_type == "text_delta":
        delta = _data_get(event, "content") or _data_get(event, "text") or ""
        if delta:
            start = state.open_text_message()
            if start:
                out.append(start)
            out.append(
                {"type": TEXT_MESSAGE_CONTENT, "messageId": state.message_id, "delta": delta}
            )

    elif event_type == "reasoning_delta":
        delta = _data_get(event, "content") or ""
        if delta:
            if state.reasoning_message_id is None:
                state.reasoning_message_id = _new_id("reason")
                out.append(
                    {
                        "type": REASONING_MESSAGE_START,
                        "messageId": state.reasoning_message_id,
                        "role": "reasoning",
                    }
                )
            out.append(
                {
                    "type": REASONING_MESSAGE_CONTENT,
                    "messageId": state.reasoning_message_id,
                    "delta": delta,
                }
            )

    elif event_type == "tool_start":
        # 工具调用打断文本输出：先收尾文本消息
        close = state.close_text_message()
        if close:
            out.append(close)
        tool_call_id = _data_get(event, "tool_call_id") or _new_id("tool")
        out.append(
            {
                "type": TOOL_CALL_START,
                "toolCallId": tool_call_id,
                "toolCallName": _data_get(event, "tool_name") or "unknown",
                "parentMessageId": state.message_id,
            }
        )
        args = _data_get(event, "args")
        if args is not None:
            out.append(
                {
                    "type": TOOL_CALL_ARGS,
                    "toolCallId": tool_call_id,
                    "delta": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                }
            )

    elif event_type in ("tool_end", "tool_error"):
        tool_call_id = _data_get(event, "tool_call_id") or _new_id("tool")
        out.append({"type": TOOL_CALL_END, "toolCallId": tool_call_id})
        output = _data_get(event, "output")
        if output is not None:
            out.append(
                {
                    "type": TOOL_CALL_RESULT,
                    "messageId": _new_id("msg"),
                    "toolCallId": tool_call_id,
                    "content": output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                    "role": "tool",
                }
            )

    elif event_type == "approval_required":
        # AG-UI 无原生审批事件，用 CUSTOM 承载
        out.append(
            {"type": CUSTOM, "name": "approval_required", "value": _get(event, "data", {})}
        )

    elif event_type in ("done", "run_end"):
        close = state.close_text_message()
        if close:
            out.append(close)
        if state.reasoning_message_id is not None:
            out.append(
                {"type": REASONING_MESSAGE_END, "messageId": state.reasoning_message_id}
            )
            state.reasoning_message_id = None
        usage = _data_get(event, "usage") or {}
        state.usage = dict(usage) if isinstance(usage, dict) else {}
        if event_type == "run_end" and _data_get(event, "status", "completed") not in ("completed", "interrupted"):
            out.append({"type": RUN_ERROR, "message": str(_data_get(event, "content") or "Agent did not complete"), "code": str(_data_get(event, "status"))})
        elif event_type == "run_end":
            finished: dict[str, Any] = {
                "type": RUN_FINISHED,
                "threadId": state.thread_id,
                "runId": state.run_id or _new_id("run"),
            }
            if state.usage:
                finished["usage"] = {
                    "promptTokens": state.usage.get("prompt_tokens", 0),
                    "completionTokens": state.usage.get("completion_tokens", 0),
                    "totalTokens": state.usage.get("total_tokens", 0),
                }
            out.append(finished)

    elif event_type == "error":
        close = state.close_text_message()
        if close:
            out.append(close)
        out.append(
            {
                "type": RUN_ERROR,
                "message": str(_data_get(event, "message") or "agent run failed"),
                "code": str(_data_get(event, "error_type") or "AgentError"),
            }
        )

    # llm_start/llm_end/task_start/task_end 等内部事件不映射，静默跳过
    return out


def _finalize(state: _EventState) -> list[dict[str, Any]]:
    """流意外结束时补齐边界事件（AG-UI 要求 run 必须有 START + FINISHED/ERROR）。"""
    out: list[dict[str, Any]] = []
    close = state.close_text_message()
    if close:
        out.append(close)
    if state.reasoning_message_id is not None:
        out.append({"type": REASONING_MESSAGE_END, "messageId": state.reasoning_message_id})
        state.reasoning_message_id = None
    out.append(
        {
            "type": RUN_FINISHED,
            "threadId": state.thread_id,
            "runId": state.run_id or _new_id("run"),
        }
    )
    return out


async def ag_ui_events(
    agent: Any,
    user_input: str,
    thread_id: str,
    *,
    run_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """运行 agent 并产出 AG-UI 事件 dict 的异步序列。

    优先走 ``agent.stream_events``（增量）；agent 不支持时退化为
    ``agent.run`` 单次调用，输出整段文本。
    """
    state = _EventState(thread_id, run_id=run_id)

    stream_events = getattr(agent, "stream_events", None)
    if callable(stream_events):
        emitted_terminal = False
        try:
            parameters = inspect.signature(stream_events).parameters
            kwargs = {"session_id": thread_id} if "session_id" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()) else {}
            stream = stream_events(user_input, **kwargs)
            async for event in stream:
                for mapped in map_agent_event(event, state):
                    if mapped["type"] in (RUN_FINISHED, RUN_ERROR):
                        emitted_terminal = True
                    yield mapped
                    if mapped["type"] == RUN_ERROR:
                        return
            if not emitted_terminal:
                yield {"type": RUN_ERROR, "message": "Agent stream ended without terminal event", "code": "IncompleteStream"}
        except Exception as exc:
            yield {"type": RUN_ERROR, "message": str(exc), "code": type(exc).__name__}
        finally:
            if "stream" in locals() and hasattr(stream, "aclose"):
                await stream.aclose()
        return

    # Duck-typed fallback keeps the same thread mapping and honest terminal state.
    state.run_id = state.run_id or _new_id("run")
    yield {"type": RUN_STARTED, "threadId": thread_id, "runId": state.run_id}
    try:
        run = agent.run
        parameters = inspect.signature(run).parameters
        kwargs = {}
        if "session_id" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            kwargs["session_id"] = thread_id
        elif "session" in parameters and callable(getattr(agent, "create_or_get_session", None)):
            kwargs["session"] = agent.create_or_get_session(session_id=thread_id)
        result = await run(user_input, **kwargs)
    except Exception as exc:
        yield {"type": RUN_ERROR, "message": str(exc), "code": type(exc).__name__}
        return
    content = _get(result, "content", None)
    if content is None:
        content = "" if result is None else str(result)
    status = _get(result, "status", "completed")
    if status == "interrupted":
        yield {"type": CUSTOM, "name": "approval_required", "value": _get(result, "interrupt", {})}
    elif status != "completed":
        yield {"type": RUN_ERROR, "message": content or "Agent did not complete", "code": str(status)}
        return
    message_id = _new_id("msg")
    yield {"type": TEXT_MESSAGE_START, "messageId": message_id, "role": "assistant"}
    if content:
        yield {"type": TEXT_MESSAGE_CONTENT, "messageId": message_id, "delta": content}
    yield {"type": TEXT_MESSAGE_END, "messageId": message_id}
    finished: dict[str, Any] = {
        "type": RUN_FINISHED,
        "threadId": thread_id,
        "runId": state.run_id,
    }
    usage = _get(result, "usage", None)
    if isinstance(usage, dict) and usage:
        finished["usage"] = {
            "promptTokens": usage.get("prompt_tokens", 0),
            "completionTokens": usage.get("completion_tokens", 0),
            "totalTokens": usage.get("total_tokens", 0),
        }
    yield finished


def _sse_bytes(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _stream_response(
    agent: Any,
    user_input: str,
    thread_id: str,
    *,
    sse: bool,
) -> Response:
    async def gen() -> AsyncIterator[bytes]:
        async for event in ag_ui_events(agent, user_input, thread_id):
            if sse:
                yield _sse_bytes(event)
            else:
                yield json.dumps(event, ensure_ascii=False) + "\n"

    if sse:
        return Response(
            gen(),
            content_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return Response(gen(), content_type="application/x-ndjson; charset=utf-8")


def _extract_last_user_message(messages: list[dict[str, Any]] | None) -> str:
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str):
            return content
    return ""


def build_ag_ui_app(agent: Any, *, name: str = "wuwei-agent"):
    """返回暴露 agent 的 AG-UI ASGI 应用（POST /run，SSE/NDJSON 事件流）。"""

    async def run(request: Request) -> Response:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPError(400, {"type": RUN_ERROR, "message": "request body must be an object"})
        thread_id = body.get("threadId") or body.get("thread_id")
        if not thread_id:
            raise HTTPError(400, {"type": RUN_ERROR, "message": "missing 'threadId'"})
        messages = body.get("messages")
        user_input = _extract_last_user_message(messages)
        if not user_input.strip():
            raise HTTPError(
                400,
                {"type": RUN_ERROR, "message": "messages must contain a user message with non-empty content"},
            )
        accept = (request.headers.get("accept") or "").lower()
        fmt = request.query.get("format", "").lower()
        sse = "ndjson" not in fmt and "ndjson" not in accept
        return await _stream_response(agent, user_input, str(thread_id), sse=sse)

    routes = {("POST", "/run"): run}
    return make_asgi_app(routes, name=f"ag-ui:{name}")
