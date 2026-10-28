"""MCP 会话管理

connect_all 并发连接所有启用的服务器；单个服务器连接/发现失败只记录到
self.failures，不中断其他服务器。工具发现（discover）只在连接成功后执行一次，
结果缓存在 self._tools，供概览注入与 tool_search 检索复用。
"""

from __future__ import annotations

import asyncio

from wuwei.mcp.client import BaseMCPClient, HTTPMCPClient, StdioMCPClient
from wuwei.mcp.config import MCPConfig
from wuwei.mcp.tools import MCPToolAdapter
from wuwei.tools.tool import Tool


class MCPSessionManager:
    """MCP 会话管理器

    管理多个 MCP 服务器连接和工具发现。

    示例：
        config = MCPConfig.load()
        session = MCPSessionManager(config)
        await session.connect_all()

        tools = session.get_all_tools()
        agent = Agent(llm=llm, tools=tools)

    单服务器失败隔离：connect_all 不会因某个服务器失败而抛出异常，
    失败信息记录在 session.failures 中（server name -> Exception）。
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self._clients: dict[str, BaseMCPClient] = {}
        self._tools: dict[str, list[Tool]] = {}
        self.failures: dict[str, BaseException] = {}

    async def _connect_server(self, name: str, server_config) -> None:
        """连接单个服务器并发现工具（失败抛出异常，由 connect_all 收集）"""
        if server_config.transport == "stdio":
            client: BaseMCPClient = StdioMCPClient(server_config)
        else:
            # http 与 sse 都走 SDK 的 streamable http 实现
            client = HTTPMCPClient(server_config)

        try:
            await client.connect()
            adapter = MCPToolAdapter(client, name)
            tools = await adapter.discover_tools()
        except BaseException:
            await client.disconnect()
            raise

        self._clients[name] = client
        self._tools[name] = tools

    async def connect_all(self) -> None:
        """并发连接所有启用的 MCP 服务器，单失败隔离不中断

        - 并发执行（asyncio.gather + return_exceptions=True）
        - 单个服务器失败记录到 self.failures，不抛出
        - 已连接的服务器跳过（工具只 discover 一次）
        """
        self.failures = {}
        pending = [
            server_config
            for server_config in self.config.get_enabled_servers()
            if server_config.name not in self._clients
        ]
        if not pending:
            return

        results = await asyncio.gather(
            *(
                self._connect_server(server_config.name, server_config)
                for server_config in pending
            ),
            return_exceptions=True,
        )
        for server_config, result in zip(pending, results):
            if isinstance(result, BaseException):
                self.failures[server_config.name] = result

    async def disconnect_all(self) -> None:
        """并发断开所有连接"""
        clients = list(self._clients.values())
        results = await asyncio.gather(
            *(client.disconnect() for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, BaseException):
                self.failures[client.config.name] = result
        self._clients.clear()
        self._tools.clear()

    def get_all_tools(self) -> list[Tool]:
        """获取所有 MCP 工具"""
        tools = []
        for server_tools in self._tools.values():
            tools.extend(server_tools)
        return tools

    def get_tools_by_server(self, server_name: str) -> list[Tool]:
        """获取指定服务器的工具"""
        return self._tools.get(server_name, [])

    def get_client(self, server_name: str) -> BaseMCPClient | None:
        """获取指定服务器的客户端"""
        return self._clients.get(server_name)

    def list_servers(self) -> list[str]:
        """列出所有已连接的服务器"""
        return list(self._clients.keys())
