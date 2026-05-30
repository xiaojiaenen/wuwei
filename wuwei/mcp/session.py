"""MCP 会话管理"""

from wuwei.mcp.config import MCPConfig
from wuwei.mcp.client import BaseMCPClient, StdioMCPClient, HTTPMCPClient
from wuwei.mcp.tools import MCPToolAdapter
from wuwei.tools.base import Tool


class MCPSessionManager:
    """MCP 会话管理器

    管理多个 MCP 服务器连接和工具发现。

    示例：
        config = MCPConfig.load()
        session = MCPSessionManager(config)
        await session.connect_all()

        tools = session.get_all_tools()
        agent = Agent(llm=llm, tools=tools)
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self._clients: dict[str, BaseMCPClient] = {}
        self._tools: dict[str, list[Tool]] = {}

    async def connect_all(self):
        """连接所有启用的 MCP 服务器"""
        for name, server_config in self.config.get_enabled_servers():
            if server_config.transport == "stdio":
                client = StdioMCPClient(server_config)
            else:
                client = HTTPMCPClient(server_config)

            await client.connect()
            self._clients[name] = client

            # 发现工具
            adapter = MCPToolAdapter(client, name)
            self._tools[name] = await adapter.discover_tools()

    async def disconnect_all(self):
        """断开所有连接"""
        for client in self._clients.values():
            await client.disconnect()
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

    def list_servers(self) -> list[str]:
        """列出所有已连接的服务器"""
        return list(self._clients.keys())
