"""极小的 ASGI 辅助层（零第三方依赖）。

wuwei/serve 下的 A2A 与 AG-UI HTTP 出口共用这一层：
不依赖 fastapi/starlette，直接实现 ASGI 协议（PEP 3333 的异步版），
足够覆盖「路由固定路径 + JSON 请求 + JSON/SSE/NDJSON 响应」的场景。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

__all__ = [
    "HTTPError",
    "Request",
    "Response",
    "json_response",
    "make_asgi_app",
]

MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB


class HTTPError(Exception):
    """可映射为 HTTP 响应的错误（payload 需可 JSON 序列化）。"""

    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(str(payload))
        self.status = status
        self.payload = payload


class Request:
    """对 ASGI scope/receive 的最小封装。"""

    def __init__(self, scope: dict, receive) -> None:
        self._scope = scope
        self._receive = receive
        self.method: str = scope.get("method", "GET").upper()
        self.path: str = scope.get("path", "/")
        self.headers: dict[str, str] = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        self.query: dict[str, str] = {}
        for chunk in (scope.get("query_string") or b"").decode("latin-1").split("&"):
            if not chunk:
                continue
            key, _, value = chunk.partition("=")
            self.query[key] = value

    async def body(self) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await self._receive()
            if message.get("type") == "http.request":
                data = message.get("body", b"")
                chunks.append(data)
                size += len(data)
                if size > MAX_BODY_BYTES:
                    raise HTTPError(413, {"error": "request body too large"})
                if not message.get("more_body"):
                    break
            elif message.get("type") == "http.disconnect":
                raise HTTPError(400, {"error": "client disconnected"})
            else:  # 未知消息类型，视为流结束
                break
        return b"".join(chunks)

    async def json(self) -> Any:
        raw = await self.body()
        if not raw:
            raise HTTPError(400, {"error": "empty request body, expected JSON"})
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPError(400, {"error": f"invalid JSON body: {exc}"}) from exc


class Response:
    """HTTP 响应；body 可以是 bytes、str 或异步迭代器（流式）。"""

    def __init__(
        self,
        body: bytes | str | AsyncIterator[bytes],
        *,
        status: int = 200,
        content_type: str = "application/json; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.content_type = content_type
        self.headers = headers or {}


def json_response(payload: Any, *, status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        status=status,
    )


Handler = Callable[[Request], Awaitable[Response]]


async def _send_lifespan(receive, send) -> None:
    """优雅处理 lifespan scope（uvicorn/hypercorn 启动钩子）。"""
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


def make_asgi_app(routes: dict[tuple[str, str], Handler], *, name: str = "app"):
    """构建一个固定路由表的 ASGI 应用。

    routes: {(method, path): handler}；handler 返回 Response，
    抛 HTTPError 映射为对应状态码，抛其它异常映射为 500。
    """

    async def app(scope: dict, receive, send) -> None:
        if scope["type"] == "lifespan":
            await _send_lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        request = Request(scope, receive)
        handler = routes.get((request.method, request.path))
        try:
            if handler is None:
                raise HTTPError(
                    404,
                    {"error": f"not found: {request.method} {request.path} ({name})"},
                )
            response = await handler(request)
        except HTTPError as exc:
            response = json_response(exc.payload, status=exc.status)
        except Exception as exc:  # noqa: BLE001 —— 兜底映射为 500
            response = json_response(
                {"error": f"{type(exc).__name__}: {exc}"}, status=500
            )

        headers = [
            (b"content-type", response.content_type.encode("latin-1")),
            *[(k.encode("latin-1"), v.encode("latin-1")) for k, v in response.headers.items()],
        ]
        await send(
            {
                "type": "http.response.start",
                "status": response.status,
                "headers": headers,
            }
        )
        if isinstance(response.body, AsyncIterator):
            try:
                async for chunk in response.body:
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            finally:
                if hasattr(response.body, "aclose"):
                    await response.body.aclose()
        else:
            body = response.body
            if isinstance(body, str):
                body = body.encode("utf-8")
            await send({"type": "http.response.body", "body": body})

    return app
