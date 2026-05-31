"""插件系统"""

from wuwei.plugin.loader import PluginLoader, Plugin
from wuwei.plugin.registry import PluginRegistry

__all__ = [
    "PluginLoader",
    "Plugin",
    "PluginRegistry",
]
