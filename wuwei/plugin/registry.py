"""插件注册表"""

from typing import Any, Callable
from wuwei.plugin.loader import Plugin


class PluginRegistry:
    """插件注册表

    管理已加载的插件，提供钩子和工具注册功能。
    """

    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._hooks: dict[str, list[Callable]] = {}
        self._tools: dict[str, dict] = {}

    def register(self, plugin: Plugin):
        """注册插件"""
        self._plugins[plugin.name] = plugin

        # 注册钩子
        for event, handler_name in plugin.hooks.items():
            if plugin.module and hasattr(plugin.module, handler_name):
                handler = getattr(plugin.module, handler_name)
                if event not in self._hooks:
                    self._hooks[event] = []
                self._hooks[event].append(handler)

        # 注册工具
        for tool_def in plugin.tools:
            tool_name = tool_def.get("name")
            if tool_name:
                self._tools[tool_name] = tool_def

    def get_hook(self, event: str) -> list[Callable]:
        """获取指定事件的钩子"""
        return self._hooks.get(event, [])

    def get_tool(self, name: str) -> dict | None:
        """获取指定工具"""
        return self._tools.get(name)

    def list_plugins(self) -> list[Plugin]:
        """列出所有插件"""
        return list(self._plugins.values())

    def list_hooks(self) -> dict[str, list[Callable]]:
        """列出所有钩子"""
        return self._hooks.copy()

    def list_tools(self) -> dict[str, dict]:
        """列出所有工具"""
        return self._tools.copy()
