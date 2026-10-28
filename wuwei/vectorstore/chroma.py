"""Chroma 向量库适配器。

使用官方 chromadb SDK（lazy import），采用 COSINE 空间，score = 1 - distance。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from wuwei.memory.embedder import Embedder
from wuwei.vectorstore._util import flatten_metadata, metadata_matches
from wuwei.vectorstore.base import ScoredChunk

INSTALL_HINT = "pip install 'wuwei[chroma]' 或 pip install chromadb"


class ChromaVectorStore:
    """基于 Chroma 的 VectorStore 适配器。

    - host 传入时连接远程服务；
    - persist_dir 传入时使用本地持久化；
    - 都不传时使用内存模式（进程退出即丢弃）。

    embedding 由构造传入的 Embedder 生成后写入。
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection_name: str = "wuwei",
        host: str | None = None,
        port: int | None = None,
        persist_dir: str | None = None,
        **kwargs: Any,
    ):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                f"使用 ChromaVectorStore 需要安装 chromadb：`{INSTALL_HINT}`"
            ) from e
        if embedder is None:
            raise ValueError(
                "ChromaVectorStore 需要传入 embedder，"
                "参考 wuwei.memory.embedder.OpenAIEmbedder"
            )

        if host:
            self._client = chromadb.HttpClient(host=host, port=port or 8000, **kwargs)
        elif persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder

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

        ids, metas = [], []
        for md in metadatas:
            meta = flatten_metadata(md)
            meta.update(
                {
                    "wuwei_source": source,
                    "wuwei_namespace": namespace,
                    "wuwei_title": title or "",
                }
            )
            ids.append(uuid4().hex)
            metas.append(meta)

        self._collection.add(
            ids=ids,
            documents=list(texts),
            embeddings=embeddings,
            metadatas=metas,
        )
        return len(ids)

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """余弦相似度检索，score 降序；filter 为 metadata 键值相等匹配。"""
        total = self._collection.count()
        if total == 0 or not query.strip():
            return []
        vec = await self._embedder.embed_query(query)
        # 多取一批以便客户端过滤后仍够 limit 条
        n_results = min(max(limit, 1) * 4, total)
        res = self._collection.query(
            query_embeddings=[vec],
            n_results=n_results,
            where={"wuwei_namespace": namespace},
        )
        docs = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        results = []
        for doc, dist, meta in zip(docs, distances, metas, strict=False):
            meta = meta or {}
            if not metadata_matches(
                {k: v for k, v in meta.items() if not k.startswith("wuwei_")}, filter
            ):
                continue
            results.append(
                ScoredChunk(
                    text=doc,
                    source=meta.get("wuwei_source", ""),
                    score=1.0 - float(dist),
                    title=meta.get("wuwei_title") or None,
                    namespace=meta.get("wuwei_namespace", namespace),
                    metadata={k: v for k, v in meta.items() if not k.startswith("wuwei_")},
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[: max(limit, 0)]

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """删除指定 source 的全部片段。"""
        self._collection.delete(
            where={"$and": [{"wuwei_source": source}, {"wuwei_namespace": namespace}]}
        )

    async def count(self, namespace: str | None = None) -> int:
        """统计片段数量。"""
        if namespace is None:
            return self._collection.count()
        got = self._collection.get(where={"wuwei_namespace": namespace})
        return len(got.get("ids", []))
