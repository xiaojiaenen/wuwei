"""MCP 客户端 — 基于官方 `mcp` Python SDK

所有传输（stdio / http / sse）统一通过官方 SDK 实现：

- stdio: ``mcp.client.stdio.stdio_client`` + ``StdioServerParameters``
- http:  ``mcp.client.streamable_http.streamable_http_client``（Streamable HTTP）
- sse: ``mcp.client.sse.sse_client``（旧版 HTTP+SSE，独立握手端点）

本模块保留原有 ``BaseMCPClient`` 接口（connect/disconnect/list_tools/call_tool），
list_tools/call_tool 返回协议层 dict（camelCase 键，如 ``inputSchema`` /
``readOnlyHint``），供 ``MCPToolAdapter`` 等既有代码继续使用。

连接握手（initialize）由 SDK 的 ``ClientSession`` 完成，不再手写 JSON-RPC。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncContextManager

from wuwei.mcp.config import MCPServerConfig

__all__ = [
    "BaseMCPClient",
    "MCPClientError",
    "SDKSessionMCPClient",
    "StdioMCPClient",
    "HTTPMCPClient",
]


class MCPClientError(Exception):
    """MCP 客户端连接/调用失败（错误信息面向用户，需清晰可定位）"""


class BaseMCPClient(ABC):
    """MCP 客户端基类"""

    @abstractmethod
    async def connect(self):
        """连接到 MCP 服务器（完成 initialize 握手）"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict]:
        """列出可用工具（协议层 dict 列表）"""
        ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具，返回协议层 dict（含 content/isError）"""
        ...


class SDKSessionMCPClient(BaseMCPClient):
    """基于官方 SDK ``ClientSession`` 的客户端基类

    子类只需实现 ``_open_streams()``，提供 (read_stream, write_stream) 的
    async 上下文管理器；握手、list_tools、call_tool 的协议细节全部交给 SDK。
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session = None
        self._stack: AsyncExitStack | None = None

    @abstractmethod
    def _open_streams(self) -> AsyncContextManager[tuple[Any, Any]]:
        """打开传输层流，yield (read_stream, write_stream)"""
        ...

    async def connect(self):
        """建立连接并完成 initialize 握手"""
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(self._open_streams())
            self.session = await self._stack.enter_async_context(
                _make_session(read, write, self.config.timeout)
            )
            await self.session.initialize()
        except Exception as exc:
            await self.disconnect()
            raise MCPClientError(
                f"MCP 服务器 '{self.config.name}' 连接失败 "
                f"({self._transport_label()}): {exc}"
            ) from exc

    async def disconnect(self):
        """关闭会话与传输层"""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass  # 断开时的清理错误不应阻塞调用方
            self._stack = None
        self.session = None

    async def list_tools(self) -> list[dict]:
        """列出工具（SDK Tool 对象 → 协议层 camelCase dict）"""
        if self.session is None:
            raise MCPClientError(
                f"MCP 服务器 '{self.config.name}' 未连接，请先调用 connect()"
            )
        result = await self.session.list_tools()
        return [
            tool.model_dump(by_alias=True, exclude_none=True)
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具，返回协议层 dict（content / isError）"""
        if self.session is None:
            raise MCPClientError(
                f"MCP 服务器 '{self.config.name}' 未连接，请先调用 connect()"
            )
        try:
            result = await self.session.call_tool(name, arguments or {})
        except Exception as exc:
            raise MCPClientError(
                f"MCP 工具调用失败 (server='{self.config.name}', tool='{name}'): {exc}"
            ) from exc
        return result.model_dump(by_alias=True, exclude_none=True)

    def _transport_label(self) -> str:
        return self.config.transport


def _make_session(read: Any, write: Any, timeout: float):
    """创建 SDK ClientSession（独立函数便于测试时替换/延迟导入）"""
    from mcp import ClientSession

    return ClientSession(read, write, read_timeout_seconds=timeout)


class StdioMCPClient(SDKSessionMCPClient):
    """Stdio 传输的 MCP 客户端（基于 SDK stdio_client）

    子进程只继承白名单环境变量 + ``config.env``，防止泄漏宿主机敏感信息。
    """

    # 子进程继承的环境变量白名单（防止泄漏 API key 等敏感信息）
    ENV_WHITELIST = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL")

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._server_params = None

    @classmethod
    def build_env(cls, config: MCPServerConfig) -> dict[str, str]:
        """计算传给子进程的环境变量：白名单 + config.env"""
        env = {k: v for k, v in os.environ.items() if k in cls.ENV_WHITELIST}
        env.update(config.env or {})
        return env

    def _open_streams(self) -> AsyncContextManager[tuple[Any, Any]]:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._server_params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=self.build_env(self.config),
        )
        return stdio_client(self._server_params)

    def _transport_label(self) -> str:
        return f"stdio: {self.config.command} {' '.join(self.config.args)}".strip()


class HTTPMCPClient(SDKSessionMCPClient):
    """HTTP/SSE 传输的 MCP 客户端（基于 SDK Streamable HTTP）

    sse 配置走旧版 sse_client；http 配置走 Streamable HTTP。
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)

    def _open_streams(self) -> AsyncContextManager[tuple[Any, Any]]:

        return self._combined_streams()

    @asynccontextmanager
    async def _combined_streams(self):
        """自建 http client（带 headers/timeout）并挂到 streamable http 传输上"""
        if self.config.transport == "sse":
            from mcp.client.sse import sse_client
            async with sse_client(self.config.url, headers=self.config.headers or None,
                                  timeout=self.config.timeout) as streams:
                yield streams[0], streams[1]
            return
        from mcp.client.streamable_http import (
            create_mcp_http_client,
            streamable_http_client,
        )

        client = create_mcp_http_client(
            headers=dict(self.config.headers or {}) or None,
            timeout=_http_timeout(self.config.timeout),
        )
        async with client, streamable_http_client(
            self.config.url,
            http_client=client,
        ) as streams:
            yield streams[0], streams[1]

    def _transport_label(self) -> str:
        return f"http: {self.config.url}"


def _http_timeout(timeout: float):
    """构造 httpx2 超时对象（延迟导入，跟随 SDK 自带的 httpx2 依赖）"""
    from mcp.client.streamable_http import httpx2

    return httpx2.Timeout(timeout)
