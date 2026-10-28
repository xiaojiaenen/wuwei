"""内置插件 — wuwei 自带的插件集合"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from wuwei.plugin.plugin import Plugin, PluginContext

if TYPE_CHECKING:
    from wuwei.plugin.manager import PluginManager

_logger = logging.getLogger("wuwei.plugin.builtin")

# 内置插件分组（依赖顺序：基础工具在前，子系统插件在后）
BUILTIN_GROUPS: dict[str, list[str]] = {
    # 最小集：纯只读、零风险的基础工具
    "minimal": ["calc", "time", "json", "text"],
    # 全集：包含写文件 / git / npm / 执行脚本等有副作用的工具
    "all": [
        "calc", "time", "file", "git", "npm", "python",
        "decision", "json", "http", "text", "skill", "rag", "mcp", "vfs",
    ],
}


def _make_plugin(name: str, description: str, module) -> Plugin:
    """从模块创建 Plugin 对象。"""
    return Plugin(
        name=name,
        description=description,
        _setup_fn=getattr(module, "setup", None),
        _teardown_fn=getattr(module, "teardown", None),
    )


def _builtin_modules() -> dict[str, tuple[str, object]]:
    """惰性导入所有内置插件模块。"""
    from wuwei.plugin.builtin import (
        calc,
        decision,
        file,
        git,
        http,
        json_tools,
        mcp,
        npm,
        python,
        rag,
        skill,
        text,
        time_plugin,
        vfs,
    )
    return {
        "calc": ("数学计算", calc),
        "time": ("时间查询", time_plugin),
        "file": ("文件操作", file),
        "git": ("Git 操作", git),
        "npm": ("NPM 操作", npm),
        "python": ("Python 脚本执行", python),
        "decision": ("用户决策", decision),
        "json": ("JSON 处理", json_tools),
        "http": ("HTTP 请求", http),
        "text": ("文本处理", text),
        "skill": ("技能管理", skill),
        "rag": ("知识库检索", rag),
        "mcp": ("MCP 工具", mcp),
        "vfs": ("虚拟文件系统", vfs),
    }


def load_builtin(manager: PluginManager, names: list[str]) -> list[str]:
    """按名称加载指定内置插件，返回实际加载成功的名称。

    失败的插件仅记录日志，不中断（与旧行为一致）。
    """
    modules = _builtin_modules()
    loaded: list[str] = []
    for name in names:
        entry = modules.get(name)
        if entry is None:
            _logger.warning("未知内置插件: %s（可选: %s）", name, ", ".join(modules))
            continue
        desc, mod = entry
        try:
            manager.register(_make_plugin(name, desc, mod))
            loaded.append(name)
        except Exception:
            _logger.exception("加载内置插件 '%s' 失败", name)
    return loaded


def load_all_builtin(manager: PluginManager) -> list[str]:
    """加载所有内置插件（等效于 load_builtin(manager, BUILTIN_GROUPS["all"])）。"""
    return load_builtin(manager, BUILTIN_GROUPS["all"])


__all__ = ["load_all_builtin", "load_builtin", "BUILTIN_GROUPS"]
