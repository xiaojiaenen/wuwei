"""MCP 插件 — 懒加载发现并注册 MCP 工具

在第一次 LLM 调用时连接所有 MCP 服务器、发现工具并注册到 ToolRegistry。
工具名称格式：mcp__{server}__{name}
"""

from __future__ import annotations

from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.mcp.tools import MCPToolAdapter
from wuwei.plugin import PluginContext


def setup(ctx: PluginContext) -> None:
    """注册 MCP 工具发现中间件

    仅在 PluginContext 携带 mcp_manager 时生效。
    中间件在首次 LLM 调用前完成 MCP 服务器连接和工具注册，
    后续调用直接跳过。
    """
    if ctx.mcp_manager is None:
        return

    ctx.middleware_stack.add(MCPToolDiscoveryMiddleware(ctx))


class MCPToolDiscoveryMiddleware(Middleware):
    """MCP 工具发现中间件

    在首次 before_llm 时：
    1. 通过 mcp_manager 连接所有配置的 MCP 服务器
    2. 使用 MCPToolAdapter 发现远程工具
    3. 将工具注册到 ToolRegistry（名称带 mcp__{server}__ 前缀）

    后续调用直接跳过，零开销。
    """

    def __init__(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._discovered = False

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        if self._discovered:
            return ctx

        mcp_manager = self._ctx.mcp_manager
        if mcp_manager is None:
            self._discovered = True
            return ctx

        # 连接所有 MCP 服务器并发现工具
        await mcp_manager.connect_all()

        # 将发现的工具注册到 ToolRegistry
        registry = self._ctx.tool_registry
        for server_name in mcp_manager.list_servers():
            client = mcp_manager._clients.get(server_name)
            if client is None:
                continue

            adapter = MCPToolAdapter(client, server_name)
            tools = await adapter.discover_tools()
            for tool in tools:
                # 避免重复注册（可能在多次调用间被外部注册过）
                if registry.get(tool.name) is None:
                    registry.register(tool)

        self._discovered = True
        return ctx
