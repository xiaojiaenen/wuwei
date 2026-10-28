"""VectorStore 抽象与多后端适配器。

统一协议见 base.VectorStore，检索结果见 base.ScoredChunk。
后端：内存（InMemoryVectorStore）、Chroma、Qdrant、pgvector（PostgreSQL）、Milvus。

示例：
    from wuwei.vectorstore import get_vector_store, KnowledgeStoreCompat
    from wuwei.middleware.rag import RagMiddleware

    store = get_vector_store("memory")
    middleware = RagMiddleware(knowledge_store=KnowledgeStoreCompat(store))
"""

from wuwei.vectorstore.adapter import KnowledgeStoreCompat
from wuwei.vectorstore.base import ScoredChunk, VectorStore
from wuwei.vectorstore.chroma import ChromaVectorStore
from wuwei.vectorstore.factory import get_vector_store
from wuwei.vectorstore.inmemory import InMemoryVectorStore
from wuwei.vectorstore.milvus import MilvusVectorStore
from wuwei.vectorstore.pgvector import SCHEMA_SQL, PgVectorStore
from wuwei.vectorstore.qdrant import QdrantVectorStore

__all__ = [
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "KnowledgeStoreCompat",
    "MilvusVectorStore",
    "PgVectorStore",
    "QdrantVectorStore",
    "SCHEMA_SQL",
    "ScoredChunk",
    "VectorStore",
    "get_vector_store",
]
