"""VectorStore 模块测试

覆盖：
- ScoredChunk 模型与 VectorStore 协议
- InMemoryVectorStore 全功能（有/无 embedder、filter、namespace、删除、计数）
- KnowledgeStoreCompat 与 RagMiddleware 组合
- 工厂函数
- 外部后端适配器的 lazy import 报错信息（chroma/qdrant/pgvector/milvus）
"""

from __future__ import annotations

import importlib.util

import pytest

from wuwei.core.message import HumanMessage, SystemMessage
from wuwei.graph.state import State
from wuwei.memory.embedder import SimpleEmbedder
from wuwei.middleware.base import MiddlewareContext
from wuwei.middleware.rag import RagMiddleware
from wuwei.vectorstore import (
    ChromaVectorStore,
    InMemoryVectorStore,
    KnowledgeStoreCompat,
    MilvusVectorStore,
    PgVectorStore,
    QdrantVectorStore,
    ScoredChunk,
    VectorStore,
    get_vector_store,
)
from wuwei.vectorstore.factory import _BACKENDS


def _make_ctx(user_text: str, system_text: str = "你是助手") -> MiddlewareContext:
    """构造带 system + user 消息的中间件上下文。"""
    state = State()
    state.messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    return MiddlewareContext(state=state)


class TestScoredChunk:
    """ScoredChunk 模型测试"""

    def test_defaults(self):
        """默认字段值正确"""
        chunk = ScoredChunk(text="无为而治")
        assert chunk.source == ""
        assert chunk.score == 0.0
        assert chunk.title is None
        assert chunk.namespace == "default"
        assert chunk.metadata == {}

    def test_full_fields(self):
        """全字段构造与 metadata 保留"""
        chunk = ScoredChunk(
            text="上善若水",
            source="daodejing",
            score=0.9,
            title="道德经",
            namespace="dao",
            metadata={"chapter": 8},
        )
        assert chunk.metadata == {"chapter": 8}
        assert chunk.title == "道德经"


class TestInMemoryVectorStore:
    """InMemoryVectorStore 全功能测试"""

    async def test_isinstance_protocol(self):
        """InMemoryVectorStore 满足 VectorStore 协议"""
        store = InMemoryVectorStore()
        assert isinstance(store, VectorStore)

    async def test_add_and_count(self):
        """add 返回写入条数，count 支持按 namespace 统计"""
        store = InMemoryVectorStore()
        assert await store.add(["a", "b"], source="s1") == 2
        assert await store.add(["c"], source="s2", namespace="other") == 1
        assert await store.count() == 3
        assert await store.count("default") == 2
        assert await store.count("other") == 1

    async def test_search_word_overlap_without_embedder(self):
        """无 embedder 时按词重叠打分，score 归一到 0-1"""
        store = InMemoryVectorStore()
        await store.add(["无为而治 道法自然"], source="dao")
        await store.add(["仁者爱人 克己复礼"], source="ru")

        results = await store.search("无为 道法", limit=2)
        assert results[0].source == "dao"
        assert 0.0 <= results[0].score <= 1.0
        assert results[0].score == 1.0  # 两个查询词全部命中

    async def test_search_with_embedder_cosine(self):
        """有 embedder 时按余弦相似度打分"""
        store = InMemoryVectorStore(embedder=SimpleEmbedder())
        await store.add(["今天天气晴朗，适合出门散步"], source="weather")
        await store.add(["向量数据库用于存储高维向量"], source="db")

        results = await store.search("天气晴朗 出门", limit=2)
        assert results[0].source == "weather"
        assert results[0].score > 0.0

    async def test_search_limit_and_order(self):
        """limit 生效且按 score 降序"""
        store = InMemoryVectorStore()
        await store.add(["目标 在这里出现"], source="hit1")
        await store.add(["完全无关的内容"], source="miss")
        await store.add(["目标 关键词 都在这里"], source="hit2")

        results = await store.search("目标 关键词", limit=2)
        assert len(results) == 2
        assert results[0].source == "hit2"
        assert results[0].score > results[1].score > 0.0

    async def test_search_metadata_filter(self):
        """filter 支持 metadata 键值相等匹配"""
        store = InMemoryVectorStore()
        await store.add(
            ["目标 文本甲", "目标 文本乙"],
            source="s",
            metadatas=[{"lang": "zh", "v": 1}, {"lang": "en", "v": 1}],
        )
        results = await store.search("目标", limit=10, filter={"lang": "zh"})
        assert len(results) == 1
        assert results[0].metadata == {"lang": "zh", "v": 1}
        # 不匹配的 filter 应返回空
        assert await store.search("目标", limit=10, filter={"lang": "fr"}) == []

    async def test_namespace_isolation(self):
        """namespace 之间相互隔离"""
        store = InMemoryVectorStore()
        await store.add(["共享文本"], source="s", namespace="a")
        await store.add(["共享文本"], source="s", namespace="b")

        assert len(await store.search("共享", namespace="a")) == 1
        assert len(await store.search("共享", namespace="b")) == 1
        assert await store.search("共享", namespace="c") == []

    async def test_delete_by_source(self):
        """删除指定 source，且只影响对应 namespace"""
        store = InMemoryVectorStore()
        await store.add(["x1", "x2"], source="s1")
        await store.add(["y"], source="s2", namespace="other")
        await store.add(["z"], source="s1", namespace="other")

        await store.delete_by_source("s1")
        assert await store.count() == 2
        assert await store.count("other") == 2  # other 里 s1+s2 各一条

    async def test_add_same_source_replaces(self):
        """同 (source, namespace) 重复 add 先删后写"""
        store = InMemoryVectorStore()
        await store.add(["旧文本"], source="doc")
        await store.add(["新文本"], source="doc")
        assert await store.count() == 1
        results = await store.search("新文本")
        assert results[0].text == "新文本"

    async def test_search_empty_store(self):
        """空库检索返回空列表"""
        store = InMemoryVectorStore()
        assert await store.search("任意") == []

    async def test_title_and_metadata_roundtrip(self):
        """title 与 metadata 随写入/检索往返保留"""
        store = InMemoryVectorStore()
        await store.add(
            ["带元数据的文本"], source="s", title="标题", metadatas=[{"page": 3}]
        )
        results = await store.search("元数据")
        assert results[0].title == "标题"
        assert results[0].metadata == {"page": 3}


class TestKnowledgeStoreCompat:
    """KnowledgeStoreCompat 适配器测试"""

    async def test_ingest_splits_and_counts(self):
        """ingest 切块写入并返回分块列表"""
        store = InMemoryVectorStore()
        compat = KnowledgeStoreCompat(store)
        chunks = await compat.ingest(
            "甲" * 30 + "乙" * 30, source="doc", chunk_size=20, chunk_overlap=5
        )
        assert len(chunks) > 1
        assert all(c.source == "doc" for c in chunks)
        assert await store.count() == len(chunks)

    async def test_search_returns_text_source(self):
        """search 返回带 .text/.source 的对象"""
        store = InMemoryVectorStore()
        compat = KnowledgeStoreCompat(store)
        await compat.ingest("无为而治，道法自然", source="dao")
        results = await compat.search("无为 道法", limit=1)
        assert results[0].text == "无为而治，道法自然"
        assert results[0].source == "dao"

    async def test_delete_by_source(self):
        """透传 delete_by_source"""
        store = InMemoryVectorStore()
        compat = KnowledgeStoreCompat(store)
        await compat.ingest("文本", source="doc")
        await compat.delete_by_source("doc")
        assert await store.count() == 0


class TestRagIntegration:
    """KnowledgeStoreCompat + RagMiddleware 组合测试"""

    async def test_before_llm_injects_at_correct_position(self):
        """before_llm 检索并注入到开头 system 消息之后"""
        store = InMemoryVectorStore()
        compat = KnowledgeStoreCompat(store)
        await compat.ingest("无为而治是道家的核心思想", source="daodejing", title="道德经")

        middleware = RagMiddleware(knowledge_store=compat, top_k=3)
        ctx = await middleware.before_llm(_make_ctx("什么是无为而治？"))

        # 消息数 2 -> 3，注入位置在开头 system 之后（index 1）
        assert len(ctx.state.messages) == 3
        injected = ctx.state.messages[1]
        assert injected.role == "system"
        assert "daodejing" in injected.content
        assert "道德经" in injected.content
        assert "无为而治是道家的核心思想" in injected.content
        # 原有消息未被移动
        assert ctx.state.messages[0].content == "你是助手"
        assert ctx.state.messages[2].role == "user"

    async def test_before_llm_min_score_filters(self):
        """min_score 阈值过滤生效"""
        store = InMemoryVectorStore()
        compat = KnowledgeStoreCompat(store)
        await compat.ingest("完全无关的文本", source="noise")

        # 阈值高到过滤掉一切时不注入
        middleware = RagMiddleware(knowledge_store=compat, top_k=3, min_score=0.99)
        ctx = await middleware.before_llm(_make_ctx("量子力学是什么"))
        assert len(ctx.state.messages) == 2

    async def test_before_llm_empty_store_noop(self):
        """空知识库时 before_llm 不注入、不报错"""
        compat = KnowledgeStoreCompat(InMemoryVectorStore())
        middleware = RagMiddleware(knowledge_store=compat)
        ctx = await middleware.before_llm(_make_ctx("任意问题"))
        assert len(ctx.state.messages) == 2

    async def test_works_with_embedder(self):
        """组合 SimpleEmbedder 后同样正常注入"""
        store = InMemoryVectorStore(embedder=SimpleEmbedder())
        compat = KnowledgeStoreCompat(store)
        await compat.ingest("知识库中的参考资料内容", source="kb")

        middleware = RagMiddleware(knowledge_store=compat)
        ctx = await middleware.before_llm(_make_ctx("参考资料内容"))
        assert len(ctx.state.messages) == 3


class TestFactory:
    """工厂函数测试"""

    def test_memory_backend(self):
        """memory/inmemory 返回内存后端"""
        assert isinstance(get_vector_store("memory"), InMemoryVectorStore)
        assert isinstance(get_vector_store("InMemory", embedder=None), InMemoryVectorStore)

    def test_unknown_backend_raises(self):
        """未知后端报 ValueError 并列出可用项"""
        with pytest.raises(ValueError, match="unknown-db"):
            get_vector_store("unknown-db")
        with pytest.raises(ValueError, match="chroma"):
            get_vector_store("nope")


class TestExternalAdapterImports:
    """外部后端适配器 lazy import 测试

    SDK 未安装时必须抛出带包名和安装提示的 ImportError；
    已安装时对能本地运行的后端（chroma 内存模式 / qdrant 内存模式）做一次全流程验证。
    """

    @staticmethod
    def _assert_install_hint(exc_value: Exception, package: str):
        """校验 ImportError 信息包含包名与安装提示。"""
        msg = str(exc_value)
        assert package in msg
        assert "pip install" in msg

    def test_chroma_import_error(self):
        if importlib.util.find_spec("chromadb") is not None:
            pytest.skip("chromadb 已安装，跳过 ImportError 路径")
        with pytest.raises(ImportError) as exc:
            ChromaVectorStore(embedder=SimpleEmbedder())
        self._assert_install_hint(exc.value, "chromadb")

    def test_qdrant_import_error(self):
        if importlib.util.find_spec("qdrant_client") is not None:
            pytest.skip("qdrant_client 已安装，跳过 ImportError 路径")
        with pytest.raises(ImportError) as exc:
            QdrantVectorStore(embedder=SimpleEmbedder())
        self._assert_install_hint(exc.value, "qdrant-client")

    def test_pgvector_import_error(self):
        if importlib.util.find_spec("asyncpg") is not None or importlib.util.find_spec(
            "psycopg"
        ) is not None:
            pytest.skip("pg 驱动已安装，跳过 ImportError 路径")
        with pytest.raises(ImportError) as exc:
            PgVectorStore(embedder=SimpleEmbedder(), dsn="postgresql://localhost/db")
        self._assert_install_hint(exc.value, "asyncpg")

    def test_milvus_import_error(self):
        if importlib.util.find_spec("pymilvus") is not None:
            pytest.skip("pymilvus 已安装，跳过 ImportError 路径")
        with pytest.raises(ImportError) as exc:
            MilvusVectorStore(embedder=SimpleEmbedder())
        self._assert_install_hint(exc.value, "pymilvus")

    def test_all_backends_registered(self):
        """工厂注册了全部声明过的后端"""
        registered = set(_BACKENDS)
        for name in ["chroma", "qdrant", "pgvector", "postgres", "milvus", "memory"]:
            assert name in registered

    @pytest.mark.asyncio
    async def test_chroma_roundtrip_local(self):
        """chromadb 已安装时用内存模式跑一次全流程"""
        if importlib.util.find_spec("chromadb") is None:
            pytest.importorskip("chromadb")
        store = ChromaVectorStore(embedder=SimpleEmbedder())
        await store.add(["无为而治 道法自然"], source="dao", title="道德经")
        results = await store.search("无为 道法")
        assert results and results[0].source == "dao"
        await store.delete_by_source("dao")
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_qdrant_roundtrip_local(self):
        """qdrant_client 已安装时用内存模式跑一次全流程"""
        if importlib.util.find_spec("qdrant_client") is None:
            pytest.importorskip("qdrant_client")
        store = QdrantVectorStore(embedder=SimpleEmbedder())
        await store.add(["无为而治 道法自然"], source="dao", title="道德经")
        results = await store.search("无为 道法")
        assert results and results[0].source == "dao"
        await store.delete_by_source("dao")
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_pgvector_connect_skip(self):
        """pgvector 后端连接失败时跳过"""
        if importlib.util.find_spec("asyncpg") is None:
            pytest.importorskip("asyncpg")
        store = PgVectorStore(
            embedder=SimpleEmbedder(), dsn="postgresql://invalid:invalid@localhost:1/db"
        )
        try:
            await store.count()
        except OSError:
            pytest.skip("无法连接 PostgreSQL")
        except Exception:
            pytest.skip("无法连接 PostgreSQL")

    @pytest.mark.asyncio
    async def test_milvus_connect_skip(self):
        """milvus 后端连接失败时跳过"""
        if importlib.util.find_spec("pymilvus") is None:
            pytest.importorskip("pymilvus")
        store = MilvusVectorStore(
            embedder=SimpleEmbedder(), uri="http://localhost:19530"
        )
        try:
            await store.count()
        except Exception:
            pytest.skip("无法连接 Milvus")
