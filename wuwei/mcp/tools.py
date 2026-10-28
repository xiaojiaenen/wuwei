"""MCP 工具适配器

将 MCP 服务器上的工具转换为 wuwei Tool 对象。
这是 MCP 协议层的一部分，不是插件。

v3.1 修复：
- inputSchema 的扩展键（$schema/additionalProperties/format 等）不再被
  pydantic 校验静默丢弃，只提取协议要求的三个字段
- readOnlyHint 注解正确映射到 execution.side_effect（旧实现传了不存在的
  side_effect= 构造参数被 pydantic 静默忽略，标记永远丢失）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wuwei.tools.tool import Tool, ToolExecutionPolicy, ToolParameters

if TYPE_CHECKING:
    from wuwei.mcp.client import BaseMCPClient


def _to_tool_parameters(schema: dict[str, Any] | None) -> ToolParameters:
    """从 MCP inputSchema 提取 ToolParameters，保留 properties/required。"""
    schema = schema if isinstance(schema, dict) else {}
    return ToolParameters(
        type="object",
        properties=schema.get("properties") or {},
        required=schema.get("required") or [],
    )


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
            read_only = bool(raw.get("annotations", {}).get("readOnlyHint", False))
            tool = Tool(
                name=f"mcp__{self.server_name}__{raw['name']}",
                description=raw.get("description", ""),
                parameters=_to_tool_parameters(raw.get("inputSchema")),
                handler=self._create_handler(raw["name"]),
                execution=ToolExecutionPolicy(
                    side_effect=not read_only,
                    requires_approval=not read_only,
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
