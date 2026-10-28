"""Memory Curator — 记忆写入策展层

社区共识：主流方案把记忆当"检索问题"做，真正难的是**写入侧策展**
（什么该记、怎么合并、怎么过期、防投毒）。本模块补齐写入侧：

- **写入闸门**：相似记忆去重合并（不是追加，保留 revision 计数）
- **来源信任分级**：user > inferred > tool；untrusted 进隔离区，
  检索时默认排除——对记忆投毒的第一道防线
- **遗忘曲线调度**：周期性触发底层 store 的 decay cleanup

用法：
    store = CuratedMemoryStore(InMemoryMemoryStore(embedder=...))
    await store.add("用户偏好简洁回复", trust="user")
    await store.add("用户似乎喜欢蓝色", trust="inferred")
    # 检索自动过滤隔离区与低分记忆
    memories = await store.search("用户喜欢什么")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from wuwei.memory.memory_store import MemoryRecord, MemoryStore

logger = logging.getLogger("wuwei.memory.curator")

TrustLevel = Literal["user", "inferred", "tool", "untrusted"]

# 信任权重：检索时低信任记忆打折扣
_TRUST_WEIGHT = {"user": 1.0, "inferred": 0.7, "tool": 0.5, "untrusted": 0.0}


class CuratedMemoryStore:
    """带写入策展的记忆存储（包装任意 MemoryStore 实现，duck-type 兼容协议）。"""

    def __init__(
        self,
        inner: MemoryStore,
        *,
        dedupe_similarity: float = 0.92,
        namespace: str = "default",
    ):
        self.inner = inner
        self.dedupe_similarity = dedupe_similarity
        self.namespace = namespace

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
        trust: TrustLevel = "inferred",
    ) -> MemoryRecord:
        """写入记忆：先查重合并，再按信任分级落库。"""
        # 1. 与现有记忆查重（有 embedding 时用相似度，否则前缀/包含匹配）
        duplicate = await self._find_duplicate(content, namespace)
        if duplicate is not None:
            # 合并：保留更早的记忆 id，重要性取 max，累加 revision
            merged_meta = dict(duplicate.metadata or {})
            merged_meta["revisions"] = (merged_meta.get("revisions", 1)) + 1
            merged_meta["last_content"] = content
            # 信任只升不降
            old_trust = merged_meta.get("trust", "inferred")
            if _TRUST_WEIGHT.get(trust, 0) > _TRUST_WEIGHT.get(old_trust, 0):
                merged_meta["trust"] = trust
            importance = max(duplicate.importance, importance)
            # 用底层 delete+add 实现"更新"（MemoryStore 协议无 update）
            await self.inner.delete(duplicate.id)
            return await self.inner.add(
                duplicate.content,
                namespace=namespace,
                memory_type=duplicate.memory_type,
                importance=importance,
                confidence=max(duplicate.confidence, confidence),
                tags=list(set((duplicate.tags or []) + (tags or []))),
                metadata=merged_meta,
            )

        merged_meta = dict(metadata or {})
        merged_meta["trust"] = trust
        return await self.inner.add(
            content,
            namespace=namespace,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            tags=tags,
            metadata=merged_meta,
        )

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 5,
        include_untrusted: bool = False,
    ) -> list[MemoryRecord]:
        """检索：默认排除隔离区（untrusted），按信任加权排序。"""
        results = await self.inner.search(query, namespace=namespace, limit=limit * 2)
        scored = []
        for record in results:
            trust = (record.metadata or {}).get("trust", "inferred")
            if trust == "untrusted" and not include_untrusted:
                continue
            weight = _TRUST_WEIGHT.get(trust, 0.5)
            scored.append((weight, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:limit]]

    async def delete(self, memory_id: str) -> None:
        await self.inner.delete(memory_id)

    async def list_all(self, *, namespace: str = "default") -> list[MemoryRecord]:
        return await self.inner.list_all(namespace=namespace)

    async def cleanup(self, *, namespace: str = "default", threshold: float = 0.1):
        return await self.inner.cleanup(namespace=namespace, threshold=threshold)

    async def _find_duplicate(self, content: str, namespace: str) -> MemoryRecord | None:
        existing = await self.inner.list_all(namespace=namespace)
        embedder = getattr(self.inner, "_embedder", None)
        if embedder is not None:
            try:
                query_vec = await embedder.embed_query(content)
                for record in existing:
                    if record.embedding is not None:
                        from wuwei.memory.memory_store import _cosine_similarity

                        if _cosine_similarity(query_vec, record.embedding) >= self.dedupe_similarity:
                            return record
            except Exception:
                logger.debug("查重 embedding 失败，降级文本匹配", exc_info=True)
        lowered = content.strip().lower()
        for record in existing:
            existing_text = (record.content or "").strip().lower()
            if not existing_text:
                continue
            if existing_text in lowered or lowered in existing_text:
                return record
        return None


async def schedule_cleanup(
    store,
    *,
    interval_seconds: float = 3600,
    threshold: float = 0.1,
    namespace: str = "default",
):
    """周期性触发记忆衰减清理（遗忘曲线调度器）。作为后台任务运行。"""
    while True:
        try:
            removed = await store.cleanup(namespace=namespace, threshold=threshold)
            if removed:
                logger.info("[curator] 记忆衰减清理 %d 条", len(removed))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("记忆清理失败")
        await asyncio.sleep(interval_seconds)
