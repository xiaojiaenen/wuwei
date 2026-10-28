"""MCP 插件 — 概览注入 + tool_search 按需注册

治理 MCP 工具描述膨胀：接入多个 MCP 服务器时，默认**不再**把全部工具注册
进上下文，而是在首次 LLM 调用时：

1. 连接所有服务器（单失败隔离），每个服务器只 discover 一次
2. 向 system prompt 注入**服务器概览**——每个 server 一行：名称 + 工具数
   + 工具名清单摘要
3. 注册一个 ``tool_search`` 工具，模型按需检索并把匹配的具体工具注册进
   ToolRegistry
4. 配置 ``eager_tools``（list[str]，支持 fnmatch 通配符如
   ``mcp__srv__*``）允许启动即注册指定工具

工具名称格式保持：mcp__{server}__{name}
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from wuwei.core.message import SystemMessage
from wuwei.mcp.session import MCPSessionManager
from wuwei.mcp.tool_search import ToolSearchIndex, build_tool_search_tool
from wuwei.middleware import Middleware, MiddlewareContext
from wuwei.plugin import PluginContext

TOOL_SEARCH_NAME = "tool_search"

MCP_OVERVIEW_INSTRUCTION = """以下是你可用的 MCP 服务器概览：
{overview}

注意：概览只列出服务器与工具名，具体工具尚未加载。需要使用某个 MCP 能力时，
先调用 tool_search 工具检索（输入能力关键词），匹配到的工具会被自动注册，
之后即可直接调用。"""

# 概览中每个 server 最多列出的工具名数量，超出部分省略
MAX_TOOL_NAMES_PER_SERVER = 10

# 插件配置键：ctx.config["mcp"]["eager_tools"] 或 ctx.config["eager_tools"]


def setup(ctx: PluginContext) -> None:
    """注册 MCP 工具发现中间件

    仅在 PluginContext 携带 mcp_manager 时生效。
    """
    if ctx.mcp_manager is None:
        return

    ctx.middleware_stack.add(MCPToolDiscoveryMiddleware(ctx))


def _get_eager_tools(config: dict) -> list[str]:
    """从插件配置读取 eager_tools（顶层或 mcp 小节均可）"""
    if not config:
        return []
    eager = config.get("eager_tools")
    if eager is None:
        eager = (config.get("mcp") or {}).get("eager_tools")
    if isinstance(eager, list):
        return [str(p) for p in eager if p]
    return []


def build_server_overview(
    mcp_manager: MCPSessionManager,
    max_names: int = MAX_TOOL_NAMES_PER_SERVER,
) -> str:
    """构建服务器概览：每个 server 一行（名称 + 工具数 + 工具名摘要）

    包含连接失败的服务器（标注失败原因），让模型知道哪些能力不可用。
    """
    lines: list[str] = []
    for server_name in mcp_manager.list_servers():
        tools = mcp_manager.get_tools_by_server(server_name)
        names = [t.name for t in tools]
        count = len(names)
        if count > max_names:
            summary = ", ".join(names[:max_names]) + f", … 等 {count} 个工具"
        else:
            summary = ", ".join(names) if names else "（无工具）"
        lines.append(f"- {server_name}（{count} 个工具）: {summary}")
    for server_name, error in mcp_manager.failures.items():
        lines.append(f"- {server_name}（连接失败）: {error}")
    return "\n".join(lines)


class MCPToolDiscoveryMiddleware(Middleware):
    """MCP 工具发现中间件（概览注入 + 按需注册）

    在首次 before_llm 时：
    1. 通过 mcp_manager 连接所有配置的 MCP 服务器（单失败隔离）
    2. 建立 ToolSearchIndex（全部已发现工具，只 discover 一次）
    3. 注册 tool_search 工具；按 eager_tools 配置立即注册指定工具
    4. 向 system prompt 注入服务器概览

    后续调用直接跳过，零开销。
    """

    def __init__(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._discovered = False

    def _register_tool(self, tool) -> bool:
        """注册工具到 ToolRegistry，返回是否新注册成功"""
        registry = self._ctx.tool_registry
        if registry.get(tool.name) is not None:
            return False
        registry.register(tool)
        return True

    def _register_eager_tools(self, index: ToolSearchIndex, patterns: list[str]) -> None:
        """按通配符模式立即注册指定工具（如 mcp__srv__*）"""
        for entry in index.entries:
            tool = entry["tool"]
            if any(fnmatchcase(tool.name, pattern) for pattern in patterns):
                self._register_tool(tool)

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        if self._discovered:
            overview = build_server_overview(self._ctx.mcp_manager)
            if overview:
                ctx.state.messages[:] = [
                    m for m in ctx.state.messages if m.metadata.get("source") != "mcp_overview"
                ]
                ctx.state.messages.insert(
                    0,
                    SystemMessage(
                        content=MCP_OVERVIEW_INSTRUCTION.format(overview=overview),
                        metadata={"source": "mcp_overview"},
                    ),
                )
            return ctx

        mcp_manager = self._ctx.mcp_manager
        if mcp_manager is None:
            self._discovered = True
            return ctx

        # 连接所有 MCP 服务器（单失败隔离在 manager 内部处理）
        await mcp_manager.connect_all()

        # 建立工具索引（工具已由 manager 缓存，不再重复 discover）
        index = ToolSearchIndex()
        for server_name in mcp_manager.list_servers():
            index.add_tools(mcp_manager.get_tools_by_server(server_name), server_name)

        # 注册 tool_search 工具（幂等）
        if len(index) > 0 and self._ctx.tool_registry.get(TOOL_SEARCH_NAME) is None:
            self._ctx.tool_registry.register(build_tool_search_tool(index, self._register_tool))

        # eager_tools：启动即注册指定工具
        eager = _get_eager_tools(self._ctx.config)
        if eager:
            self._register_eager_tools(index, eager)

        # 注入服务器概览（每个 server 一行）
        overview = build_server_overview(mcp_manager)
        if overview:
            ctx.state.messages.insert(
                0,
                SystemMessage(
                    content=MCP_OVERVIEW_INSTRUCTION.format(overview=overview),
                    metadata={"source": "mcp_overview"},
                ),
            )

        self._discovered = True
        return ctx
