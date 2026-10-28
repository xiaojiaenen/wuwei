"""内存版 VectorStore，零外部依赖。

- 有 embedder 时用余弦相似度打分；
- 无 embedder 时退化为查询词与文本的词重叠比例打分（归一到 0-1），仅供测试和演示。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from wuwei.memory.embedder import Embedder
from wuwei.memory.memory_store import _cosine_similarity
from wuwei.vectorstore.base import ScoredChunk


@dataclass
class _Record:
    """内存中的一条向量记录。"""

    text: str
    source: str
    namespace: str
    title: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


class InMemoryVectorStore:
    """内存向量存储，实现 VectorStore 协议。

    示例：
        store = InMemoryVectorStore(embedder=SimpleEmbedder())
        await store.add(["无为而治"], source="dao", title="道德经")
        results = await store.search("无为")
    """

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder
        self._records: list[_Record] = []

    # ------------------------------------------------------------------
    # VectorStore 协议
    # ------------------------------------------------------------------
    async def add(
        self,
        texts: list[str],
        source: str,
        *,
        namespace: str = "default",
        title: str | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """写入一批文本，同 (source, namespace) 先删后写。"""
        await self.delete_by_source(source, namespace=namespace)

        embeddings: list[list[float]] | None = None
        if self._embedder and texts:
            embeddings = await self._embedder.embed_texts(list(texts))

        metadatas = metadatas or [None] * len(texts)
        for i, text in enumerate(texts):
            self._records.append(
                _Record(
                    text=text,
                    source=source,
                    namespace=namespace,
                    title=title,
                    metadata=dict(metadatas[i]) if metadatas and metadatas[i] else {},
                    embedding=embeddings[i] if embeddings else None,
                )
            )
        return len(texts)

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """按相似度检索，score 降序。filter 为 metadata 键值相等匹配。"""
        candidates = [
            r for r in self._records if r.namespace == namespace and self._match(r, filter)
        ]
        if not candidates or not query.strip():
            return []

        if self._embedder:
            query_vec = await self._embedder.embed_query(query)
            scored = [
                (self._cosine(query_vec, r.embedding), r)
                for r in candidates
            ]
        else:
            scored = [(self._word_overlap(query, r.text), r) for r in candidates]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ScoredChunk(
                text=r.text,
                source=r.source,
                score=score,
                title=r.title,
                namespace=r.namespace,
                metadata=dict(r.metadata),
            )
            for score, r in scored[: max(limit, 0)]
        ]

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """删除指定 source 的全部片段。"""
        self._records = [
            r
            for r in self._records
            if not (r.source == source and r.namespace == namespace)
        ]

    async def count(self, namespace: str | None = None) -> int:
        """统计片段数量。"""
        if namespace is None:
            return len(self._records)
        return sum(1 for r in self._records if r.namespace == namespace)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _match(record: _Record, filter: dict[str, Any] | None) -> bool:
        """metadata 键值相等过滤。"""
        if not filter:
            return True
        return all(record.metadata.get(k) == v for k, v in filter.items())

    @staticmethod
    def _cosine(query_vec: list[float], embedding: list[float] | None) -> float:
        """余弦相似度，记录缺 embedding 时按 0 分处理。"""
        if embedding is None:
            return 0.0
        return _cosine_similarity(query_vec, embedding)

    @staticmethod
    def _word_overlap(query: str, text: str) -> float:
        """无 embedder 时的兜底打分：词重叠比例，归一到 0-1。"""
        words = re.findall(r"\w+", query.lower())
        if not words:
            return 0.0
        text_lower = text.lower()
        overlap = sum(1 for w in words if w in text_lower)
        return overlap / len(words)
