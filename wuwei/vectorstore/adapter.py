"""VectorStore -> KnowledgeStore 兼容适配器。

把 VectorStore 适配成 wuwei 现有 KnowledgeStore 协议（ingest/search/delete_by_source），
使现有 RagMiddleware 无需改动即可使用任意 VectorStore 后端。
"""

from __future__ import annotations

from typing import Any

from wuwei.memory.knowledge_store import _split_text
from wuwei.vectorstore.base import ScoredChunk, VectorStore


class KnowledgeStoreCompat:
    """把 VectorStore 包装成 KnowledgeStore 协议。

    示例：
        store = get_vector_store("chroma", embedder=embedder)
        knowledge_store = KnowledgeStoreCompat(store)
        middleware = RagMiddleware(knowledge_store=knowledge_store)
    """

    def __init__(self, store: VectorStore):
        self.store = store

    async def ingest(
        self,
        text: str,
        source: str,
        *,
        namespace: str = "default",
        title: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[ScoredChunk]:
        """切块后写入向量库，返回分块结果（score 置 0，仅供查看）。"""
        pieces = _split_text(text, chunk_size, chunk_overlap)
        await self.store.add(
            pieces, source, namespace=namespace, title=title, metadatas=metadatas
        )
        return [
            ScoredChunk(
                text=piece, source=source, score=0.0, title=title, namespace=namespace
            )
            for piece in pieces
        ]

    async def search(
        self, query: str, *, namespace: str = "default", limit: int = 4
    ) -> list[ScoredChunk]:
        """检索，返回带 .text/.source 的 ScoredChunk，供 RagMiddleware 消费。"""
        return await self.store.search(query, namespace=namespace, limit=limit)

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """删除指定 source 的全部片段。"""
        await self.store.delete_by_source(source, namespace=namespace)

    async def count(self, namespace: str | None = None) -> int:
        """透传统计。"""
        return await self.store.count(namespace)
