"""PluginManager — 插件生命周期管理

统一管理插件的发现、加载、注册、卸载。
替代旧的 PluginLoader + PluginRegistry。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from .plugin import Plugin, PluginContext

_logger = logging.getLogger("wuwei.plugin")


class PluginManager:
    """插件管理器

    职责：
    1. 从目录发现并加载插件（plugin.yaml + __init__.py 或纯 __init__.py）
    2. 管理插件生命周期（注册 / 卸载）
    3. 查询已加载插件信息
    """

    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._plugins: dict[str, Plugin] = {}
        self._load_order: list[str] = []

    # ─── 查询 ───

    @property
    def plugins(self) -> dict[str, Plugin]:
        """返回已加载插件的副本。"""
        return self._plugins.copy()

    def get(self, name: str) -> Plugin | None:
        """按名称获取插件。"""
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        """列出所有已加载插件的摘要信息。"""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "tags": p.tags,
            }
            for p in self._plugins.values()
        ]

    # ─── 注册 / 卸载 ───

    def register(self, plugin: Plugin) -> None:
        """注册并初始化一个插件。

        1. 检查依赖是否已满足
        2. 调用 plugin.setup(ctx)
        3. 记录到内部状态
        """
        if plugin.name in self._plugins:
            raise ValueError(f"插件 '{plugin.name}' 已注册")

        for dep in plugin.dependencies:
            if dep not in self._plugins:
                raise ValueError(
                    f"插件 '{plugin.name}' 依赖未注册的插件 '{dep}'"
                )

        plugin.setup(self.ctx)
        self._plugins[plugin.name] = plugin
        self._load_order.append(plugin.name)
        _logger.info("✅ 插件已加载: %s v%s", plugin.name, plugin.version)

    def unregister(self, name: str) -> None:
        """卸载插件。

        先检查是否有其他插件依赖此插件，若有则拒绝卸载。
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            return

        dependents = [
            p.name for p in self._plugins.values() if name in p.dependencies
        ]
        if dependents:
            raise ValueError(
                f"无法卸载 '{name}': 被 {dependents} 依赖"
            )

        plugin.teardown()
        del self._plugins[name]
        self._load_order.remove(name)
        _logger.info("插件已卸载: %s", name)

    # ─── 目录发现 ───

    def load_directory(self, plugins_dir: str | Path) -> list[Plugin]:
        """从目录自动发现并加载插件。

        扫描 plugins_dir 下的子目录，每个子目录是一个插件：
        - 必须有 __init__.py
        - 可选 plugin.yaml（声明元数据）
        - __init__.py 必须有 setup(ctx) 函数

        跳过以 _ 开头的目录。
        """
        plugins_dir = Path(plugins_dir)
        if not plugins_dir.exists():
            return []

        loaded: list[Plugin] = []
        for plugin_path in sorted(plugins_dir.iterdir()):
            if not plugin_path.is_dir():
                continue
            if plugin_path.name.startswith("_"):
                continue

            plugin = self._discover(plugin_path)
            if plugin is None:
                continue

            try:
                self.register(plugin)
                loaded.append(plugin)
            except Exception:
                _logger.exception("加载插件 '%s' 失败", plugin_path.name)

        return loaded

    # ─── 内部 ───

    def _discover(self, path: Path) -> Plugin | None:
        """从单个目录发现插件。"""
        init_file = path / "__init__.py"
        if not init_file.exists():
            return None

        try:
            module = self._import_module(init_file)
        except Exception:
            _logger.exception("导入插件模块 '%s' 失败", path.name)
            return None

        if module is None:
            return None

        # 方式 1: 有 plugin.yaml
        yaml_file = path / "plugin.yaml"
        if yaml_file.exists():
            return self._from_yaml(path, module, yaml_file)

        # 方式 2: 约定式 — 模块必须有 setup 函数
        setup_fn = getattr(module, "setup", None)
        if callable(setup_fn):
            doc = (getattr(module, "__doc__", None) or "").strip()
            return Plugin(
                name=path.name,
                description=doc,
                path=str(path),
                _setup_fn=setup_fn,
                _teardown_fn=getattr(module, "teardown", None),
            )

        _logger.warning("插件 '%s' 无 plugin.yaml 且无 setup()", path.name)
        return None

    def _from_yaml(
        self, path: Path, module: Any, yaml_file: Path
    ) -> Plugin | None:
        """从 plugin.yaml + 模块构建 Plugin。"""
        try:
            import yaml
        except ImportError:
            _logger.error("需要安装 pyyaml 才能解析 plugin.yaml")
            return None

        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            _logger.exception("解析 %s 失败", yaml_file)
            return None

        setup_fn = getattr(module, "setup", None)
        if not callable(setup_fn):
            _logger.warning("插件 '%s' 的 __init__.py 缺少 setup(ctx) 函数", path.name)
            return None

        return Plugin(
            name=config.get("name", path.name),
            version=config.get("version", "1.0.0"),
            description=config.get("description", ""),
            author=config.get("author", ""),
            dependencies=config.get("dependencies", []),
            tags=config.get("tags", []),
            path=str(path),
            _setup_fn=setup_fn,
            _teardown_fn=getattr(module, "teardown", None),
        )

    @staticmethod
    def _import_module(init_file: Path) -> Any | None:
        """动态导入 __init__.py 模块。"""
        module_name = f"wuwei_plugin_{init_file.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, init_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
