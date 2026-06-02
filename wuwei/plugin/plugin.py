"""Plugin — wuwei 的统一扩展单元

Plugin 是 wuwei 唯一的扩展机制。每个插件通过 PluginContext 注册自己的能力：
- Tools（工具）→ 注册到 ToolRegistry
- Skills（技能）→ 注入 prompt + 注册工具
- MCP → 发现远程工具并注册
- Middleware → 生命周期钩子

不再需要独立的 hooks 系统，所有生命周期行为通过 Middleware 实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from wuwei.tools.registry import ToolRegistry
    from wuwei.tools.tool import Tool
    from wuwei.skill import SkillManager
    from wuwei.mcp.session import MCPSessionManager
    from wuwei.middleware.stack import MiddlewareStack


class PluginContext:
    """插件上下文 — 插件注册能力的入口

    通过 setup(ctx) 函数接收，插件用它来：
    - ctx.tool_registry.tool(...) — 注册工具
    - ctx.tool_registry.register(tool) — 注册 Tool 对象
    - ctx.middleware_stack.add(mw) — 添加中间件
    - ctx.skill_manager — 访问技能管理器
    - ctx.mcp_manager — 访问 MCP 管理器
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        skill_manager: SkillManager | None = None,
        mcp_manager: MCPSessionManager | None = None,
        middleware_stack: MiddlewareStack | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.skill_manager = skill_manager
        self.mcp_manager = mcp_manager
        self.middleware_stack = middleware_stack
        self.config: dict[str, Any] = config if config is not None else {}


@dataclass
class Plugin:
    """插件数据模型

    每个插件是一个目录，包含 __init__.py（必须有 setup(ctx) 函数），
    可选 plugin.yaml（声明元数据）。

    setup(ctx: PluginContext) 是插件的入口，负责向 ctx 注册工具、中间件等。
    teardown() 是可选的清理函数。

    示例（最简插件）：

        # my_plugin/__init__.py
        from wuwei.plugin import PluginContext

        def setup(ctx: PluginContext):
            @ctx.tool_registry.tool(name="hello", description="打招呼")
            async def hello(name: str = "World"):
                return f"Hello, {name}!"
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    path: str = ""
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # 内部回调
    _setup_fn: Callable[[PluginContext], None] | None = field(
        default=None, repr=False, compare=False
    )
    _teardown_fn: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def setup(self, ctx: PluginContext) -> None:
        """执行插件初始化，注册能力到上下文。"""
        if self._setup_fn is not None:
            self._setup_fn(ctx)

    def teardown(self) -> None:
        """执行插件清理。"""
        if self._teardown_fn is not None:
            self._teardown_fn()
