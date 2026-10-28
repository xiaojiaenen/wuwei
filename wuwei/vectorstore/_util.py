"""VectorStore 内部工具，供各后端适配器复用。"""

from __future__ import annotations

from typing import Any


def metadata_matches(metadata: dict[str, Any] | None, filter: dict[str, Any] | None) -> bool:
    """判断 metadata 是否满足 filter 的键值相等匹配。"""
    if not filter:
        return True
    metadata = metadata or {}
    return all(metadata.get(k) == v for k, v in filter.items())


def flatten_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """只保留标量值，兼容 Chroma / Milvus 等只接受标量元数据的后端。"""
    if not metadata:
        return {}
    return {
        k: v
        for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
