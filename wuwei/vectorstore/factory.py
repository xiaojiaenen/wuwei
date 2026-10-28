"""VectorStore 工厂函数。"""

from __future__ import annotations

from typing import Any

from wuwei.memory.embedder import Embedder
from wuwei.vectorstore.base import VectorStore

_BACKENDS = {
    "memory": ("wuwei.vectorstore.inmemory", "InMemoryVectorStore"),
    "inmemory": ("wuwei.vectorstore.inmemory", "InMemoryVectorStore"),
    "chroma": ("wuwei.vectorstore.chroma", "ChromaVectorStore"),
    "qdrant": ("wuwei.vectorstore.qdrant", "QdrantVectorStore"),
    "pgvector": ("wuwei.vectorstore.pgvector", "PgVectorStore"),
    "postgres": ("wuwei.vectorstore.pgvector", "PgVectorStore"),
    "milvus": ("wuwei.vectorstore.milvus", "MilvusVectorStore"),
}


def get_vector_store(name: str, *, embedder: Embedder | None = None, **kwargs: Any) -> VectorStore:
    """按名称构建 VectorStore 实例。

    外部后端模块内 lazy import 官方 SDK，缺依赖时抛出带安装提示的 ImportError。

    Args:
        name: 后端名称，支持 memory/inmemory/chroma/qdrant/pgvector/postgres/milvus
        embedder: Embedding 生成器（内存后端可省略，其余后端必填）
        **kwargs: 透传给具体后端的构造参数（如 dsn、url、host 等）

    示例：
        store = get_vector_store("memory")
        store = get_vector_store("pgvector", embedder=embedder,
                                 dsn="postgresql://user:pass@localhost/db")
    """
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"未知的向量库后端：{name!r}，可选：{', '.join(sorted(_BACKENDS))}"
        )

    import importlib

    module_path, class_name = _BACKENDS[key]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(embedder=embedder, **kwargs)  # type: ignore[no-any-return]
