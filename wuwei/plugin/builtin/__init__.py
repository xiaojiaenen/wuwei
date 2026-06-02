"""内置插件 — wuwei 自带的插件集合"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from wuwei.plugin.plugin import Plugin, PluginContext

if TYPE_CHECKING:
    from wuwei.plugin.manager import PluginManager

_logger = logging.getLogger("wuwei.plugin.builtin")


def _make_plugin(name: str, description: str, module) -> Plugin:
    """从模块创建 Plugin 对象。"""
    return Plugin(
        name=name,
        description=description,
        _setup_fn=getattr(module, "setup", None),
        _teardown_fn=getattr(module, "teardown", None),
    )


def load_all_builtin(manager: PluginManager) -> None:
    """加载所有内置插件。

    按依赖顺序加载：基础工具先加载，skill/rag/mcp 后加载。
    """
    from wuwei.plugin.builtin import calc as calc_mod
    from wuwei.plugin.builtin import time_plugin as time_mod
    from wuwei.plugin.builtin import file as file_mod
    from wuwei.plugin.builtin import git as git_mod
    from wuwei.plugin.builtin import npm as npm_mod
    from wuwei.plugin.builtin import python as python_mod
    from wuwei.plugin.builtin import decision as decision_mod
    from wuwei.plugin.builtin import json_tools as json_mod
    from wuwei.plugin.builtin import http as http_mod
    from wuwei.plugin.builtin import text as text_mod
    from wuwei.plugin.builtin import skill as skill_mod
    from wuwei.plugin.builtin import rag as rag_mod
    from wuwei.plugin.builtin import mcp as mcp_mod

    # 基础工具（无依赖）
    BUILTINS = [
        ("calc", "数学计算", calc_mod),
        ("time", "时间查询", time_mod),
        ("file", "文件操作", file_mod),
        ("git", "Git 操作", git_mod),
        ("npm", "NPM 操作", npm_mod),
        ("python", "Python 脚本执行", python_mod),
        ("decision", "用户决策", decision_mod),
        ("json", "JSON 处理", json_mod),
        ("http", "HTTP 请求", http_mod),
        ("text", "文本处理", text_mod),
        # 依赖其他子系统的插件
        ("skill", "技能管理", skill_mod),
        ("rag", "知识库检索", rag_mod),
        ("mcp", "MCP 工具", mcp_mod),
    ]

    for name, desc, mod in BUILTINS:
        try:
            plugin = _make_plugin(name, desc, mod)
            manager.register(plugin)
        except Exception:
            _logger.exception("加载内置插件 '%s' 失败", name)


__all__ = ["load_all_builtin"]
