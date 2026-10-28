"""Qdrant 向量库适配器。

使用官方 qdrant-client SDK（lazy import）。url 不传时使用本地内存模式，便于测试。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from wuwei.memory.embedder import Embedder
from wuwei.vectorstore._util import metadata_matches
from wuwei.vectorstore.base import ScoredChunk

INSTALL_HINT = "pip install 'wuwei[qdrant]' 或 pip install qdrant-client"


class QdrantVectorStore:
    """基于 Qdrant 的 VectorStore 适配器。

    url 不传时使用 qdrant-client 的内存模式（":memory:"），适合本地测试。
    embedding 由构造传入的 Embedder 生成后写入。
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        url: str | None = None,
        collection_name: str = "wuwei",
        **kwargs: Any,
    ):
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as e:
            raise ImportError(
                f"使用 QdrantVectorStore 需要安装 qdrant-client：`{INSTALL_HINT}`"
            ) from e
        if embedder is None:
            raise ValueError(
                "QdrantVectorStore 需要传入 embedder，"
                "参考 wuwei.memory.embedder.OpenAIEmbedder"
            )

        self._client = AsyncQdrantClient(url) if url else AsyncQdrantClient(":memory:")
        self._collection = collection_name
        self._embedder = embedder
        self._dim: int | None = None

    async def _ensure_collection(self, dim: int) -> None:
        """首次写入时按向量维度建表。"""
        from qdrant_client import models

        if self._dim == dim:
            return
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
        self._dim = dim

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
        embeddings = await self._embedder.embed_texts(list(texts))
        await self._ensure_collection(len(embeddings[0]) if embeddings else 0)

        from qdrant_client import models

        metadatas = metadatas or [None] * len(texts)
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=emb,
                payload={
                    "text": text,
                    "source": source,
                    "namespace": namespace,
                    "title": title,
                    "metadata": dict(md) if md else {},
                },
            )
            for text, emb, md in zip(texts, embeddings, metadatas, strict=False)
        ]
        await self._client.upsert(self._collection, points=points)
        return len(points)

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """余弦相似度检索，score 降序；filter 为 metadata 键值相等匹配。"""
        from qdrant_client import models

        if not query.strip():
            return []
        vec = await self._embedder.embed_query(query)
        res = await self._client.query_points(
            self._collection,
            query=vec,
            limit=max(limit, 1) * 4,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace", match=models.MatchValue(value=namespace)
                    )
                ]
            ),
        )
        results = []
        for point in res.points:
            payload = point.payload or {}
            if not metadata_matches(payload.get("metadata"), filter):
                continue
            results.append(
                ScoredChunk(
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    score=float(point.score),
                    title=payload.get("title"),
                    namespace=payload.get("namespace", namespace),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[: max(limit, 0)]

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """按过滤器删除指定 source 的全部片段。"""
        from qdrant_client import models

        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchValue(value=source)
                        ),
                        models.FieldCondition(
                            key="namespace", match=models.MatchValue(value=namespace)
                        ),
                    ]
                )
            ),
        )

    async def count(self, namespace: str | None = None) -> int:
        """统计片段数量。"""
        from qdrant_client import models

        count_filter = None
        if namespace is not None:
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace", match=models.MatchValue(value=namespace)
                    )
                ]
            )
        res = await self._client.count(self._collection, count_filter=count_filter)
        return res.count
