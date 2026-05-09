from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from wuwei.memory.embedder import Embedder
from wuwei.memory.memory_types import MemoryRecord


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def decay_score(record: MemoryRecord, now: datetime | None = None) -> float:
    """计算记忆的衰减分数。越低越应该被淘汰。

    公式: importance * 0.9^天数 * log2(2 + access_count)
    - 最近访问过的保持活跃
    - 重要 + 被频繁访问的记忆不容易衰减
    - 新创建但从未被访问的记忆按 created_at 计算天数
    """
    if now is None:
        now = datetime.now(timezone.utc)
    ref_time = record.last_accessed or record.created_at
    if ref_time is None:
        return record.importance
    days = max(0, (now - ref_time).days)
    return record.importance * (0.9 ** days) * math.log2(2 + record.access_count)


class MemoryStore(Protocol):
    """长期记忆存储协议。"""

    async def add(
        self,
        content: str,
        *,
        namespace: str = "default",
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord: ...

    async def search(
        self, query: str, *, namespace: str = "default", limit: int = 5
    ) -> list[MemoryRecord]: ...

    async def delete(self, memory_id: str) -> None: ...

    async def list_all(self, *, namespace: str = "default") -> list[MemoryRecord]: ...

    async def cleanup(
        self, *, namespace: str = "default", threshold: float = 0.1
    ) -> list[MemoryRecord]: ...


class InMemoryMemoryStore:
    """内存版长期记忆存储，零外部依赖。"""

    def __init__(self, embedder: Embedder | None = None):
        self._records: dict[str, MemoryRecord] = {}
        self._embedder = embedder

    async def add(
        self,
        content: str,
        *,
        namespace: str = "default",
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        embedding = None
        if self._embedder:
            embedding = (await self._embedder.embed_texts([content]))[0]

        record = MemoryRecord(
            id=uuid4().hex,
            content=content,
            memory_type=memory_type,
            namespace=namespace,
            importance=importance,
            confidence=confidence,
            embedding=embedding,
            tags=tags or [],
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
        )
        self._records[record.id] = record
        return record

    async def search(
        self, query: str, *, namespace: str = "default", limit: int = 5
    ) -> list[MemoryRecord]:
        candidates = [r for r in self._records.values() if r.namespace == namespace]
        if not candidates:
            return []

        if self._embedder:
            query_vec = await self._embedder.embed_query(query)
            scored = []
            for r in candidates:
                if r.embedding is not None:
                    sim = _cosine_similarity(query_vec, r.embedding)
                    score = sim * 0.6 + r.importance * 0.3 + r.confidence * 0.1
                    scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [r for _, r in scored[:limit]]
        else:
            query_lower = query.lower()
            scored = []
            for r in candidates:
                overlap = sum(1 for word in query_lower.split() if word in r.content.lower())
                score = overlap * 0.5 + r.importance * 0.3 + r.confidence * 0.2
                scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [r for _, r in scored[:limit]]

        now = datetime.now(timezone.utc)
        for r in results:
            r.last_accessed = now
            r.access_count += 1

        return results

    async def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    async def list_all(self, *, namespace: str = "default") -> list[MemoryRecord]:
        return [r for r in self._records.values() if r.namespace == namespace]

    async def cleanup(
        self, *, namespace: str = "default", threshold: float = 0.1
    ) -> list[MemoryRecord]:
        """清理衰减分数低于阈值的记忆，返回被删除的记录列表。"""
        now = datetime.now(timezone.utc)
        to_delete = [
            r
            for r in self._records.values()
            if r.namespace == namespace and decay_score(r, now) < threshold
        ]
        for r in to_delete:
            del self._records[r.id]
        return to_delete
