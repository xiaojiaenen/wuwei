"""轻量 reducer — 节点返回 dict 时的字段归并策略

替代已删除的 channels 系统（照抄 LangGraph API 面但从未接入执行引擎）。
用法：
    graph.add_node("research", research_fn, writes={"findings": extend})

- 未声明 reducer 的字段默认 overwrite
- State 上不存在的字段写入 state.metadata
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def overwrite(current: Any, new: Any) -> Any:
    """直接覆盖（默认策略）。"""
    return new


def extend(current: Any, new: Any) -> Any:
    """列表拼接：current + new。"""
    base = list(current) if current else []
    addition = list(new) if new else []
    return base + addition


def merge_dict(current: Any, new: Any) -> Any:
    """字典浅合并：{**current, **new}。"""
    base = dict(current) if current else {}
    base.update(new or {})
    return base


def add(current: Any, new: Any) -> Any:
    """数值累加。"""
    return (current or 0) + (new or 0)


def first_non_empty(current: Any, new: Any) -> Any:
    """保留第一个非空值。"""
    return current if current not in (None, "", [], {}) else new


def reducer(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    """组合多个 reducer：从左到右链式归并。"""
    def _combined(current: Any, new: Any) -> Any:
        value = new
        for fn in funcs:
            value = fn(current, value)
            current = value
        return value

    return _combined
