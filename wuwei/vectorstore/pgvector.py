"""pgvector（PostgreSQL）向量库适配器。

通过 asyncpg（lazy import）连接 PostgreSQL，embedding 存 DOUBLE PRECISION[]，
相似度在 Python 侧按余弦计算。不依赖 pgvector 扩展即可使用。
"""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from wuwei.memory.embedder import Embedder
from wuwei.memory.memory_store import _cosine_similarity
from wuwei.vectorstore._util import metadata_matches
from wuwei.vectorstore.base import ScoredChunk

INSTALL_HINT = "pip install 'wuwei[pgvector]' 或 pip install asyncpg"

# 建表 SQL。{table} 为表名占位符，构造时会校验表名合法性后填充。
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    title TEXT,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding DOUBLE PRECISION[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_ns_source_idx ON {table} (namespace, source);
"""

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PgVectorStore:
    """基于 PostgreSQL 的 VectorStore 适配器（asyncpg 驱动，连接池懒建）。

    示例：
        store = PgVectorStore(embedder=OpenAIEmbedder(api_key="..."),
                              dsn="postgresql://user:pass@localhost/db")
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        dsn: str,
        table: str = "wuwei_vectors",
        **kwargs: Any,
    ):
        try:
            import asyncpg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"使用 PgVectorStore 需要安装 asyncpg（或 psycopg）：`{INSTALL_HINT}`"
            ) from e
        if embedder is None:
            raise ValueError(
                "PgVectorStore 需要传入 embedder，参考 wuwei.memory.embedder.OpenAIEmbedder"
            )
        if not _TABLE_NAME_RE.match(table):
            raise ValueError(f"非法表名：{table!r}")

        self._dsn = dsn
        self._table = table
        self._embedder = embedder
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        """懒创建连接池并执行建表 SQL。"""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._dsn)
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL.format(table=self._table))
        return self._pool

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
        metadatas = metadatas or [None] * len(texts)
        pool = await self._get_pool()

        rows = [
            (
                uuid4().hex,
                text,
                source,
                namespace,
                title,
                json.dumps(md or {}, ensure_ascii=False),
                emb,
            )
            for text, emb, md in zip(texts, embeddings, metadatas, strict=False)
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                f"INSERT INTO {self._table} "
                "(id, text, source, namespace, title, metadata, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)",
                rows,
            )
        return len(rows)

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """取回命名空间内全部候选后按余弦相似度排序（小数据量够用）。"""
        if not query.strip():
            return []
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT text, source, namespace, title, metadata, embedding "
                f"FROM {self._table} WHERE namespace = $1",
                namespace,
            )
        if not rows:
            return []

        query_vec = await self._embedder.embed_query(query)
        results = []
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            if not metadata_matches(metadata, filter):
                continue
            emb = row["embedding"]
            score = _cosine_similarity(query_vec, list(emb)) if emb else 0.0
            results.append(
                ScoredChunk(
                    text=row["text"],
                    source=row["source"],
                    score=score,
                    title=row["title"],
                    namespace=row["namespace"],
                    metadata=metadata,
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[: max(limit, 0)]

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """删除指定 source 的全部片段。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE source = $1 AND namespace = $2",
                source,
                namespace,
            )

    async def count(self, namespace: str | None = None) -> int:
        """统计片段数量。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if namespace is None:
                return int(await conn.fetchval(f"SELECT count(*) FROM {self._table}"))
            return int(
                await conn.fetchval(
                    f"SELECT count(*) FROM {self._table} WHERE namespace = $1",
                    namespace,
                )
            )
