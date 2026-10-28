"""Milvus 向量库适配器。

使用官方 pymilvus 的 MilvusClient（lazy import，同步 SDK，此处直接包装为 async 接口）。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from wuwei.memory.embedder import Embedder
from wuwei.vectorstore._util import metadata_matches
from wuwei.vectorstore.base import ScoredChunk

INSTALL_HINT = "pip install 'wuwei[milvus]' 或 pip install pymilvus"

# 建集合 SQL/DSL 无法用纯 SQL 表达，Milvus 通过 SDK 的 schema API 建集合，
# 这里仅给出推荐配置常量供参考。
COLLECTION_CONFIG = {
    "metric_type": "COSINE",
    "index_type": "FLAT",  # 小规模够用，大规模可换 HNSW
    "vector_field": "vector",
}


class MilvusVectorStore:
    """基于 Milvus 的 VectorStore 适配器。

    示例：
        store = MilvusVectorStore(embedder=OpenAIEmbedder(api_key="..."),
                                  uri="http://localhost:19530")
    embedding 由构造传入的 Embedder 生成后写入。
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        uri: str = "http://localhost:19530",
        collection_name: str = "wuwei",
        **kwargs: Any,
    ):
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError(
                f"使用 MilvusVectorStore 需要安装 pymilvus：`{INSTALL_HINT}`"
            ) from e
        if embedder is None:
            raise ValueError(
                "MilvusVectorStore 需要传入 embedder，"
                "参考 wuwei.memory.embedder.OpenAIEmbedder"
            )

        self._client = MilvusClient(uri=uri, **kwargs)
        self._collection = collection_name
        self._embedder = embedder
        self._dim: int | None = None

    @staticmethod
    def _escape(value: str) -> str:
        """转义表达式字符串中的引号与反斜杠。"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _ensure_collection(self, dim: int) -> None:
        """首次写入时按向量维度建集合（COSINE 度量）。"""
        from pymilvus import DataType

        if self._dim == dim:
            return
        if not self._client.has_collection(self._collection):
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("source", DataType.VARCHAR, max_length=512)
            schema.add_field("namespace", DataType.VARCHAR, max_length=256)
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("metadata", DataType.JSON)
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=COLLECTION_CONFIG["index_type"],
                metric_type=COLLECTION_CONFIG["metric_type"],
            )
            self._client.create_collection(
                self._collection, schema=schema, index_params=index_params
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

        metadatas = metadatas or [None] * len(texts)
        rows = [
            {
                "id": uuid4().hex,
                "vector": emb,
                "text": text,
                "source": source,
                "namespace": namespace,
                "title": title or "",
                "metadata": dict(md) if md else {},
            }
            for text, emb, md in zip(texts, embeddings, metadatas, strict=False)
        ]
        self._client.insert(self._collection, rows)
        return len(rows)

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """COSINE 相似度检索，score 降序；filter 为 metadata 键值相等匹配。"""
        if not query.strip():
            return []
        vec = await self._embedder.embed_query(query)
        hits = self._client.search(
            self._collection,
            data=[vec],
            limit=max(limit, 1) * 4,
            filter=f'namespace == "{self._escape(namespace)}"',
            output_fields=["text", "source", "namespace", "title", "metadata"],
        )[0]

        results = []
        for hit in hits:
            entity = hit.get("entity", {})
            if not metadata_matches(entity.get("metadata"), filter):
                continue
            results.append(
                ScoredChunk(
                    text=entity.get("text", ""),
                    source=entity.get("source", ""),
                    score=float(hit.get("distance", 0.0)),
                    title=entity.get("title") or None,
                    namespace=entity.get("namespace", namespace),
                    metadata=dict(entity.get("metadata") or {}),
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[: max(limit, 0)]

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """删除指定 source 的全部片段。"""
        self._client.delete(
            self._collection,
            filter=(
                f'source == "{self._escape(source)}" and '
                f'namespace == "{self._escape(namespace)}"'
            ),
        )

    async def count(self, namespace: str | None = None) -> int:
        """统计片段数量。"""
        if namespace is None:
            stats = self._client.get_collection_stats(self._collection)
            return int(stats.get("row_count", 0))
        rows = self._client.query(
            self._collection,
            filter=f'namespace == "{self._escape(namespace)}"',
            output_fields=["count(*)"],
        )
        return int(rows[0]["count(*)"]) if rows else 0
