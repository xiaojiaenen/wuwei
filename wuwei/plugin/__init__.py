"""插件系统 — wuwei 的统一扩展机制"""

from wuwei.plugin.plugin import Plugin, PluginContext
from wuwei.plugin.manager import PluginManager

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginManager",
]
