"""插件系统 — wuwei 的统一扩展机制"""

from wuwei.plugin.manager import PluginManager
from wuwei.plugin.plugin import Plugin, PluginContext

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginManager",
]
