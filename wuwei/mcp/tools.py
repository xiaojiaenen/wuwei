"""MCP 工具适配器

将 MCP 服务器上的工具转换为 wuwei Tool 对象。
这是 MCP 协议层的一部分，不是插件。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wuwei.tools.tool import Tool

if TYPE_CHECKING:
    from wuwei.mcp.client import BaseMCPClient


class MCPToolAdapter:
    """MCP 工具适配器

    将 MCP 服务器上的工具转换为 wuwei Tool。
    工具名格式: mcp__{server_name}__{tool_name}
    """

    def __init__(self, client: BaseMCPClient, server_name: str) -> None:
        self.client = client
        self.server_name = server_name

    async def discover_tools(self) -> list[Tool]:
        """发现 MCP 服务器上的工具"""
        raw_tools = await self.client.list_tools()
        tools: list[Tool] = []
        for raw in raw_tools:
            tool = Tool(
                name=f"mcp__{self.server_name}__{raw['name']}",
                description=raw.get("description", ""),
                parameters=raw.get("inputSchema", {}),
                handler=self._create_handler(raw["name"]),
                side_effect=not raw.get("annotations", {}).get(
                    "readOnlyHint", False
                ),
            )
            tools.append(tool)
        return tools

    def _create_handler(self, tool_name: str):
        """创建工具处理函数"""

        async def handler(**kwargs):
            result = await self.client.call_tool(tool_name, kwargs)
            # 提取文本内容
            content = result.get("content", [])
            texts = [
                c["text"]
                for c in content
                if c.get("type") == "text"
            ]
            return "\n".join(texts) if texts else str(result)

        return handler
