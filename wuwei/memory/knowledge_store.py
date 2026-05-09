from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from wuwei.memory.embedder import Embedder
from wuwei.memory.memory_store import _cosine_similarity
from wuwei.memory.memory_types import KnowledgeChunk


def _split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """按字符数切块，保留 overlap 重叠。"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


class KnowledgeStore(Protocol):
    """RAG 知识库存储协议。"""

    async def ingest(
        self,
        text: str,
        source: str,
        *,
        namespace: str = "default",
        title: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> list[KnowledgeChunk]: ...

    async def search(
        self, query: str, *, namespace: str = "default", limit: int = 4
    ) -> list[KnowledgeChunk]: ...

    async def delete_by_source(
        self, source: str, *, namespace: str = "default"
    ) -> None: ...


class InMemoryKnowledgeStore:
    """内存版知识库，零外部依赖。"""

    def __init__(self, embedder: Embedder | None = None):
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._embedder = embedder

    async def ingest(
        self,
        text: str,
        source: str,
        *,
        namespace: str = "default",
        title: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> list[KnowledgeChunk]:
        await self.delete_by_source(source, namespace=namespace)

        pieces = _split_text(text, chunk_size, chunk_overlap)
        chunks: list[KnowledgeChunk] = []

        embeddings = None
        if self._embedder and pieces:
            embeddings = await self._embedder.embed_texts(pieces)

        for i, piece in enumerate(pieces):
            chunk = KnowledgeChunk(
                id=uuid4().hex,
                text=piece,
                source=source,
                namespace=namespace,
                title=title,
                embedding=embeddings[i] if embeddings else None,
            )
            self._chunks[chunk.id] = chunk
            chunks.append(chunk)

        return chunks

    async def search(
        self, query: str, *, namespace: str = "default", limit: int = 4
    ) -> list[KnowledgeChunk]:
        candidates = [c for c in self._chunks.values() if c.namespace == namespace]
        if not candidates:
            return []

        if self._embedder:
            query_vec = await self._embedder.embed_query(query)
            scored = []
            for c in candidates:
                if c.embedding is not None:
                    sim = _cosine_similarity(query_vec, c.embedding)
                    scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]
        else:
            query_lower = query.lower()
            scored = []
            for c in candidates:
                overlap = sum(1 for w in query_lower.split() if w in c.text.lower())
                scored.append((overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]

    async def delete_by_source(
        self, source: str, *, namespace: str = "default"
    ) -> None:
        to_delete = [
            cid
            for cid, c in self._chunks.items()
            if c.source == source and c.namespace == namespace
        ]
        for cid in to_delete:
            del self._chunks[cid]
