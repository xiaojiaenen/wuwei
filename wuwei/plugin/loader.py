"""插件加载器"""

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
import yaml
import importlib.util


@dataclass
class Plugin:
    """插件数据模型"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    hooks: dict[str, str] = field(default_factory=dict)  # 事件 -> 处理函数名
    tools: list[dict] = field(default_factory=list)  # 工具定义
    path: str = ""  # 插件路径
    module: Any = None  # 加载的模块


class PluginLoader:
    """插件加载器

    借鉴 Hermes-Agent 的插件系统，支持：
    - 从目录发现插件
    - 加载插件配置
    - 导入插件模块
    - 注册插件钩子和工具

    示例：
        loader = PluginLoader("plugins/")
        plugins = loader.load_all()
    """

    def __init__(self, plugins_dir: str):
        self.plugins_dir = Path(plugins_dir)
        self._plugins: dict[str, Plugin] = {}

    def load_all(self) -> list[Plugin]:
        """加载所有插件"""
        if not self.plugins_dir.exists():
            return []

        plugins = []
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir():
                plugin = self._load_plugin(plugin_dir)
                if plugin:
                    plugins.append(plugin)
                    self._plugins[plugin.name] = plugin

        return plugins

    def load_plugin(self, name: str) -> Plugin | None:
        """加载指定插件"""
        if name in self._plugins:
            return self._plugins[name]

        plugin_dir = self.plugins_dir / name
        if plugin_dir.exists():
            plugin = self._load_plugin(plugin_dir)
            if plugin:
                self._plugins[name] = plugin
                return plugin

        return None

    def _load_plugin(self, plugin_dir: Path) -> Plugin | None:
        """加载单个插件"""
        config_file = plugin_dir / "plugin.yaml"
        if not config_file.exists():
            return None

        try:
            # 加载配置
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config:
                return None

            plugin = Plugin(
                name=config.get("name", plugin_dir.name),
                version=config.get("version", "1.0.0"),
                description=config.get("description", ""),
                author=config.get("author", ""),
                hooks=config.get("hooks", {}),
                tools=config.get("tools", []),
                path=str(plugin_dir),
            )

            # 加载模块
            init_file = plugin_dir / "__init__.py"
            if init_file.exists():
                plugin.module = self._import_module(plugin_dir)

            return plugin

        except Exception as e:
            print(f"加载插件 {plugin_dir.name} 失败: {e}")
            return None

    def _import_module(self, plugin_dir: Path):
        """导入插件模块"""
        init_file = plugin_dir / "__init__.py"
        module_name = f"wuwei_plugin_{plugin_dir.name}"

        spec = importlib.util.spec_from_file_location(module_name, init_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        return None
