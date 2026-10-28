"""A2A（Agent2Agent）协议最小实现——纯 ASGI，零第三方依赖。

支持 A2A 0.3 JSON-RPC message/send；tasks/send 为弃用兼容入口。
不声明 A2A 1.0 支持。

实现要点：

- ``GET /.well-known/agent.json``（及 v0.3.0+ 的别名
  ``/.well-known/agent-card.json``）→ Agent Card；
- ``POST /tasks/send`` → 接收 ``{id, message:{role, parts:[...]}}``，
  运行 agent 并返回 A2A Task JSON（``status.state`` + ``artifacts``）；
  同时兼容 JSON-RPC 2.0 信封（``{jsonrpc, id, method: "tasks/send", params}``），
  该形态返回 ``{jsonrpc, id, result}``；
- 错误统一为 JSON-RPC 风格 ``{"error": {"code", "message"}}``（配合 HTTP 状态码）。

参考：
- A2A 规范 https://a2a-protocol.org/v0.3.0/specification
- 1.0 Agent Card（supportedInterfaces）https://github.com/a2aproject/A2A
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from wuwei.serve._http import HTTPError, Request, Response, make_asgi_app

__all__ = ["build_a2a_app", "build_agent_card", "extract_text_from_parts"]

AGENT_CARD_PATHS = ("/.well-known/agent.json", "/.well-known/agent-card.json")
TASKS_SEND_PATH = "/tasks/send"

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# A2A 扩展错误码（规范附录）
TASK_NOT_FOUND = -32001
UNSUPPORTED_OPERATION = -32006

PROTOCOL_VERSION = "0.3.0"  # 顶层 protocolVersion 采用向后兼容的 0.3.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _result_field(result: Any, key: str, default: Any = None) -> Any:
    if result is None:
        return default
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def build_agent_card(
    agent: Any,
    *,
    name: str,
    description: str,
    url: str = "http://localhost:8000",
    version: str | None = None,
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造 A2A Agent Card。

    返回 A2A 0.3 的 JSONRPC AgentCard。
    """
    card_version = version or getattr(agent, "version", None) or "0.1.0"
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,
        "version": card_version,
        "protocolVersion": PROTOCOL_VERSION,
        "preferredTransport": "JSONRPC",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": skills
        or [
            {
                "id": "run-task",
                "name": "Run Task",
                "description": description or f"Run a task with the {name} agent.",
                "tags": ["agent", "chat"],
            }
        ],
    }
    return card


def extract_text_from_parts(parts: list[dict[str, Any]] | None) -> str:
    """从 A2A message parts 中提取文本（兼容 ``type`` 与 ``kind`` 两种命名）。"""
    texts: list[str] = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("kind") or part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)


def _text_part(text: str) -> dict[str, Any]:
    # A2A 0.3 text part discriminator.
    return {"kind": "text", "text": text}


def _agent_message(content: str) -> dict[str, Any]:
    return {"kind": "message", "messageId": str(uuid.uuid4()), "role": "agent", "parts": [_text_part(content)]}


def build_task_object(
    task_id: str,
    content: str,
    *,
    state: str = "completed",
    message: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 A2A Task JSON（status + artifacts + history）。"""
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": task_id,
        # sessionId 为 0.2.x 兼容别名
        "sessionId": task_id,
        "kind": "task",
        "status": {
            "state": state,
            "timestamp": _utc_now_iso(),
        },
        "artifacts": [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "response",
                "parts": [_text_part(content)],
            }
        ],
        "history": [message] if message else [],
    }
    return task


def _jsonrpc_error(code: int, message: str, *, status: int = 400, rpc_id: Any = None) -> Response:
    from wuwei.serve._http import json_response

    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if rpc_id is not None:
        body["id"] = rpc_id
    return json_response(body, status=status)


def build_a2a_app(
    agent: Any,
    *,
    name: str,
    description: str,
    url: str = "http://localhost:8000",
    version: str | None = None,
    skills: list[dict[str, Any]] | None = None,
):
    """返回暴露 agent 的 A2A ASGI 应用（零第三方依赖）。"""

    card = build_agent_card(
        agent, name=name, description=description, url=url, version=version, skills=skills
    )

    async def _run_task(params: dict[str, Any], *, standard: bool = False) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise HTTPError(400, {"error": {"code": INVALID_PARAMS, "message": "params must be an object"}})
        task_id = params.get("id") or params.get("task_id") or str(uuid.uuid4())
        message = params.get("message")
        if not isinstance(message, dict):
            raise HTTPError(
                400,
                {
                    "error": {
                        "code": INVALID_PARAMS,
                        "message": "missing 'message' object (expected {role, parts})",
                    }
                },
            )
        if standard and (
            not isinstance(message.get("messageId"), str)
            or not message["messageId"]
            or message.get("role") not in ("user", "agent")
            or message.get("kind", "message") != "message"
            or not isinstance(message.get("parts"), list)
        ):
            raise HTTPError(400, {"error": {"code": INVALID_PARAMS,
                "message": "message/send requires messageId, role, and a parts array"}})
        text = extract_text_from_parts(message.get("parts"))
        if not text.strip():
            raise HTTPError(
                400,
                {
                    "error": {
                        "code": INVALID_PARAMS,
                        "message": "message.parts must contain at least one non-empty text part",
                    }
                },
            )
        try:
            result = await agent.run(text)
        except Exception as exc:  # noqa: BLE001 —— agent 内部错误 → A2A internal error
            raise HTTPError(
                500,
                {
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": f"agent execution failed: {type(exc).__name__}: {exc}",
                    }
                },
            ) from exc

        content = _result_field(result, "content")
        if content is None:
            content = "" if result is None else str(result)
        status = _result_field(result, "status") or "completed"
        state = {"completed": "completed", "interrupted": "input-required", "cancelled": "canceled"}.get(status, "failed")
        if content == "" and state == "completed":
            state = "completed"  # 空输出仍视为完成
        task = build_task_object(
            task_id,
            content,
            state=state,
            message=_agent_message(content),
        )
        interrupt = _result_field(result, "interrupt")
        if interrupt is not None:
            task["metadata"] = {"interrupt": interrupt.model_dump(mode="json") if hasattr(interrupt, "model_dump") else interrupt.to_dict() if hasattr(interrupt, "to_dict") else interrupt}
        return task

    async def agent_card(request: Request) -> Response:
        from wuwei.serve._http import json_response

        return json_response(card)

    async def tasks_send(request: Request) -> Response:
        from wuwei.serve._http import json_response
        rpc_id = None
        rpc = request.path != TASKS_SEND_PATH
        try:
            body = await request.json()
        except HTTPError:
            return json_response({"jsonrpc": "2.0", "id": None, "error": {"code": PARSE_ERROR, "message": "Invalid JSON"}}, status=200 if rpc else 400)
        if isinstance(body, dict) and "method" in body:
            rpc = True
            rpc_id = body.get("id")
            if body.get("jsonrpc") != "2.0":
                return json_response({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": INVALID_REQUEST, "message": "jsonrpc must be 2.0"}})
            if body.get("method") not in ("message/send", "tasks/send"):
                return json_response({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": METHOD_NOT_FOUND, "message": "Method not supported"}})
            params = body.get("params")
        elif rpc:
            return json_response({"jsonrpc": "2.0", "id": None, "error": {"code": INVALID_REQUEST, "message": "Expected JSON-RPC request"}})
        else:
            params = body
        try:
            task = await _run_task(params, standard=isinstance(body, dict) and body.get("method") == "message/send")
        except HTTPError as exc:
            error = exc.payload.get("error", {})
            envelope = {"jsonrpc": "2.0", "id": rpc_id, "error": error} if rpc else {"error": error}
            return json_response(envelope, status=200 if rpc else exc.status)
        return json_response({"jsonrpc": "2.0", "id": rpc_id, "result": task} if rpc else task)

    routes: dict[tuple[str, str], Any] = {
        ("GET", path): agent_card for path in AGENT_CARD_PATHS
    }
    routes[("POST", TASKS_SEND_PATH)] = tasks_send
    # 兼容 JSON-RPC 客户端把请求打到根路径
    routes[("POST", "/")] = tasks_send
    return make_asgi_app(routes, name=f"a2a:{name}")
