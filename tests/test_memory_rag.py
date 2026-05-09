"""长期记忆 & RAG 功能测试。"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from wuwei.memory.embedder import SimpleEmbedder
from wuwei.memory.knowledge_store import InMemoryKnowledgeStore, _split_text
from wuwei.memory.memory_store import InMemoryMemoryStore, _cosine_similarity
from wuwei.memory.memory_types import KnowledgeChunk, MemoryRecord
from wuwei.runtime.hooks import HookManager, RuntimeHook
from wuwei.runtime.memory_hook import MemoryExtractionHook, MemoryRetrievalHook
from wuwei.runtime.rag_hook import RagRetrievalHook


# ── 数据模型 ──────────────────────────────────────────


def test_memory_record_defaults():
    r = MemoryRecord(id="1", content="hello")
    assert r.memory_type == "fact"
    assert r.namespace == "default"
    assert r.importance == 0.5
    assert r.confidence == 0.8
    assert r.embedding is None
    assert r.access_count == 0


def test_knowledge_chunk_defaults():
    c = KnowledgeChunk(id="1", text="hello", source="test.md")
    assert c.namespace == "default"
    assert c.embedding is None


# ── Embedder ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_embedder_produces_vectors():
    emb = SimpleEmbedder(dim=128)
    vecs = await emb.embed_texts(["hello world", "foo bar"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128
    # 归一化后模长应为 1
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_simple_embedder_query():
    emb = SimpleEmbedder(dim=64)
    vec = await emb.embed_query("test")
    assert len(vec) == 64


# ── MemoryStore ───────────────────────────────────────


@pytest.mark.asyncio
async def test_in_memory_store_add_and_list():
    store = InMemoryMemoryStore()
    r = await store.add("用户偏好简洁回答", memory_type="preference")
    assert r.content == "用户偏好简洁回答"
    assert r.memory_type == "preference"

    all_records = await store.list_all()
    assert len(all_records) == 1


@pytest.mark.asyncio
async def test_in_memory_store_search_by_keywords():
    store = InMemoryMemoryStore()
    await store.add("Python 是项目的主语言")
    await store.add("用户不喜欢冗长的回答")
    await store.add("项目使用 PostgreSQL 数据库")

    results = await store.search("Python 语言", limit=2)
    assert len(results) >= 1
    assert any("Python" in r.content for r in results)


@pytest.mark.asyncio
async def test_in_memory_store_search_with_embedder():
    emb = SimpleEmbedder(dim=64)
    store = InMemoryMemoryStore(embedder=emb)
    await store.add("用户偏好简洁回答")
    await store.add("项目使用 Python 3.12")

    results = await store.search("编程语言", limit=2)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_in_memory_store_delete():
    store = InMemoryMemoryStore()
    r = await store.add("test")
    assert len(await store.list_all()) == 1
    await store.delete(r.id)
    assert len(await store.list_all()) == 0


@pytest.mark.asyncio
async def test_in_memory_store_namespace_isolation():
    store = InMemoryMemoryStore()
    await store.add("ns1 记忆", namespace="ns1")
    await store.add("ns2 记忆", namespace="ns2")

    assert len(await store.list_all(namespace="ns1")) == 1
    assert len(await store.list_all(namespace="ns2")) == 1
    assert len(await store.list_all(namespace="default")) == 0


@pytest.mark.asyncio
async def test_in_memory_store_access_tracking():
    store = InMemoryMemoryStore()
    await store.add("test memory")
    results = await store.search("test", limit=1)
    assert len(results) == 1
    assert results[0].access_count == 1
    assert results[0].last_accessed is not None


# ── KnowledgeStore ────────────────────────────────────


def test_split_text_basic():
    text = "a" * 1000
    chunks = _split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # 每个 chunk 不超过 chunk_size
    for c in chunks:
        assert len(c) <= 300


def test_split_text_short():
    text = "hello world"
    chunks = _split_text(text, chunk_size=300, overlap=50)
    assert chunks == ["hello world"]


@pytest.mark.asyncio
async def test_knowledge_store_ingest_and_search():
    store = InMemoryKnowledgeStore()
    text = "Wuwei 是一个轻量 Python Agent 框架。" * 10
    chunks = await store.ingest(text, source="README.md", chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 0
    assert all(c.source == "README.md" for c in chunks)

    results = await store.search("Python 框架", limit=2)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_knowledge_store_ingest_replaces_old():
    store = InMemoryKnowledgeStore()
    await store.ingest("old content", source="doc.md")
    await store.ingest("new content", source="doc.md")

    all_chunks = [c for c in store._chunks.values()]
    assert all(c.text == "new content" for c in all_chunks)


@pytest.mark.asyncio
async def test_knowledge_store_delete_by_source():
    store = InMemoryKnowledgeStore()
    await store.ingest("content a", source="a.md")
    await store.ingest("content b", source="b.md")
    assert len(store._chunks) == 2

    await store.delete_by_source("a.md")
    assert len(store._chunks) == 1
    remaining = list(store._chunks.values())
    assert remaining[0].source == "b.md"


@pytest.mark.asyncio
async def test_knowledge_store_with_embedder():
    emb = SimpleEmbedder(dim=64)
    store = InMemoryKnowledgeStore(embedder=emb)
    await store.ingest("Wuwei Agent 框架介绍", source="readme.md", chunk_size=100, chunk_overlap=10)

    results = await store.search("什么是 wuwei", limit=2)
    assert len(results) >= 1


# ── cosine similarity ─────────────────────────────────


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0, 0], [1, 0]) == 0.0


# ── MemoryRetrievalHook ───────────────────────────────


@pytest.mark.asyncio
async def test_memory_retrieval_hook_injects_into_messages():
    store = InMemoryMemoryStore()
    await store.add("用户偏好简洁回答", memory_type="preference")

    hook = MemoryRetrievalHook(store, top_k=3)

    # 模拟 session 和 messages
    session = MagicMock()
    messages = [
        MagicMock(role="system", content="你是一个助手"),
        MagicMock(role="user", content="你好"),
    ]
    tools = []

    new_messages, _ = await hook.before_llm(session, messages, tools, step=0)
    # 应该多了一条 system message
    assert len(new_messages) == 3
    assert new_messages[1].role == "system"
    assert "长期记忆" in new_messages[1].content
    assert "偏好" in new_messages[1].content


@pytest.mark.asyncio
async def test_memory_retrieval_hook_no_results():
    store = InMemoryMemoryStore()
    hook = MemoryRetrievalHook(store, top_k=3)

    session = MagicMock()
    messages = [MagicMock(role="user", content="完全无关的内容")]
    tools = []

    new_messages, _ = await hook.before_llm(session, messages, tools, step=0)
    # 没有检索结果时不注入
    assert len(new_messages) == 1


# ── RagRetrievalHook ──────────────────────────────────


@pytest.mark.asyncio
async def test_rag_retrieval_hook_injects():
    store = InMemoryKnowledgeStore()
    await store.ingest("Wuwei 是一个 Python Agent 框架", source="readme.md")

    hook = RagRetrievalHook(store, top_k=2)

    session = MagicMock()
    messages = [
        MagicMock(role="system", content="你是助手"),
        MagicMock(role="user", content="介绍一下 wuwei"),
    ]
    tools = []

    new_messages, _ = await hook.before_llm(session, messages, tools, step=0)
    assert len(new_messages) == 3
    assert "参考资料" in new_messages[1].content
    assert "readme.md" in new_messages[1].content


# ── HookManager on_run_end ────────────────────────────


@pytest.mark.asyncio
async def test_hook_manager_on_run_end():
    class TestHook(RuntimeHook):
        def __init__(self):
            self.called = False

        async def on_run_end(self, session, result, *, task=None):
            self.called = True

    hook = TestHook()
    manager = HookManager([hook])
    await manager.on_run_end(MagicMock(), None)
    assert hook.called


# ── MemoryExtractionHook ──────────────────────────────


@pytest.mark.asyncio
async def test_memory_extraction_hook_extracts():
    store = InMemoryMemoryStore()

    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '[{"type": "preference", "content": "用户喜欢简洁回答"}]'
    mock_llm.generate = AsyncMock(return_value=mock_response)

    hook = MemoryExtractionHook(mock_llm, store)

    # Mock session with context
    session = MagicMock()
    msg1 = MagicMock(role="user", content="请简洁回答我的问题，我不喜欢冗长的解释，这是我的个人偏好")
    msg2 = MagicMock(role="assistant", content="好的，我会尽量简洁地回答你的问题")
    session.context.get_messages.return_value = [msg1, msg2]

    await hook.on_run_end(session, None)

    records = await store.list_all()
    assert len(records) == 1
    assert "简洁" in records[0].content
    assert records[0].memory_type == "preference"


@pytest.mark.asyncio
async def test_memory_extraction_hook_skips_short_conversation():
    store = InMemoryMemoryStore()
    mock_llm = MagicMock()
    hook = MemoryExtractionHook(mock_llm, store)

    session = MagicMock()
    msg = MagicMock(role="user", content="hi")
    session.context.get_messages.return_value = [msg]

    await hook.on_run_end(session, None)
    assert len(await store.list_all()) == 0
    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_memory_extraction_hook_deduplicates():
    store = InMemoryMemoryStore()
    await store.add("用户喜欢简洁回答", memory_type="preference")

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '[{"type": "preference", "content": "用户喜欢简洁回答"}]'
    mock_llm.generate = AsyncMock(return_value=mock_response)

    hook = MemoryExtractionHook(mock_llm, store)

    session = MagicMock()
    msg1 = MagicMock(role="user", content="请简洁回答我的问题，我不喜欢冗长的解释，这是我的个人偏好")
    msg2 = MagicMock(role="assistant", content="好的，我会尽量简洁地回答你的问题")
    session.context.get_messages.return_value = [msg1, msg2]

    await hook.on_run_end(session, None)

    # 去重后应该还是只有 1 条
    records = await store.list_all()
    assert len(records) == 1
