"""VectorStore 基础定义

包含检索结果模型 ScoredChunk 和 VectorStore 协议。
所有后端（内存 / Chroma / Qdrant / pgvector / Milvus）都实现这个协议。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ScoredChunk(BaseModel):
    """带相似度分数的知识片段，是 VectorStore 的统一检索结果。"""

    text: str
    source: str = ""
    score: float = 0.0
    title: str | None = None
    namespace: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """向量存储协议，async-first。"""

    async def add(
        self,
        texts: list[str],
        source: str,
        *,
        namespace: str = "default",
        title: str | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """写入一批文本片段，返回写入条数。同一 (source, namespace) 会先删后写。"""
        ...

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """按相似度检索，返回按 score 降序排列的结果。

        Args:
            query: 查询文本
            namespace: 命名空间隔离
            limit: 最多返回条数
            filter: metadata 键值相等过滤，如 {"lang": "zh"}
        """
        ...

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """删除指定 source 下的全部片段。"""
        ...

    async def count(self, namespace: str | None = None) -> int:
        """统计片段数量，namespace 为 None 时统计全部。"""
        ...
