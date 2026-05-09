# 长期记忆 & RAG 开发指南

本文档告诉你怎么一步步把长期记忆和 RAG 做出来。不搞复杂设计，直接上代码。

## 目标

两个功能：

1. **长期记忆** — Agent 能记住重要信息（用户偏好、关键决策），下次对话自动用上
2. **RAG** — Agent 能从文档中检索相关内容，注入到当前对话

两者都通过 Hook 接入，不改 AgentRunner 主循环。

## 新增文件

```
wuwei/
  memory/
    memory_types.py       # 数据模型：MemoryRecord、KnowledgeChunk
    memory_store.py       # 长期记忆存储协议 + 内存实现
    knowledge_store.py    # 知识库存储协议 + 内存实现
    embedder.py           # Embedding 协议 + 内置实现
  runtime/
    memory_hook.py        # MemoryRetrievalHook + MemoryExtractionHook
    rag_hook.py           # RagRetrievalHook
```

---

## 第一步：数据模型

新建 `wuwei/memory/memory_types.py`：

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryRecord:
    """一条长期记忆。"""
    id: str                                    # 唯一 ID（uuid4.hex）
    content: str                               # 记忆正文，一句话或一小段
    memory_type: str = "fact"                  # fact / preference / constraint / summary
    namespace: str = "default"                 # 用于隔离不同用户/项目
    importance: float = 0.5                    # 0~1，越高越重要
    confidence: float = 0.8                    # 0~1，越高越确定
    embedding: list[float] | None = None       # 向量，可选
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    last_accessed: datetime | None = None      # 上次被检索到的时间
    access_count: int = 0                      # 被检索到的次数


@dataclass
class KnowledgeChunk:
    """知识库中的一个片段。"""
    id: str
    text: str                                  # 片段正文
    source: str                                # 来源文件路径或 URL
    namespace: str = "default"
    title: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

要点：
- `MemoryRecord` 有 `importance` 和 `confidence`，检索时可以加权排序
- `KnowledgeChunk` 有 `source`，方便按来源批量删除/更新
- 两者都有 `embedding` 字段，但允许为 None（无 Embedder 时退化为关键词匹配）

---

## 第二步：Embedding 协议 + 内置实现

新建 `wuwei/memory/embedder.py`：

```python
from typing import Protocol
import math


class Embedder(Protocol):
    """Embedding 协议。任何 Embedding 服务都实现这个接口。"""
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量。"""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """单条查询转向量。默认调 embed_texts 取第一条。"""
        results = await self.embed_texts([text])
        return results[0]


class OpenAIEmbedder:
    """OpenAI Embedding 适配器。"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


class SimpleEmbedder:
    """零依赖的简易 Embedder，基于字符 n-gram 哈希。仅供测试和演示。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _text_to_vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = text.lower().strip()
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            h = hash(bigram) % self.dim
            vec[h] += 1.0
        # 归一化
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._text_to_vec(text)
```

要点：
- `Embedder` 是 Protocol，你接 OpenAI、Cohere、本地模型都行
- `SimpleEmbedder` 零依赖，用字符 bigram 哈希生成向量，效果一般但能跑通整个流程
- 生产环境用 `OpenAIEmbedder` 或接入 sentence-transformers

---

## 第三步：长期记忆存储

新建 `wuwei/memory/memory_store.py`：

```python
import math
from typing import Protocol
from uuid import uuid4

from wuwei.memory.memory_types import MemoryRecord
from wuwei.memory.embedder import Embedder


class MemoryStore(Protocol):
    """长期记忆存储协议。"""

    async def add(self, content: str, *, namespace: str = "default",
                  memory_type: str = "fact", importance: float = 0.5,
                  confidence: float = 0.8, tags: list[str] | None = None,
                  metadata: dict | None = None) -> MemoryRecord:
        """存一条记忆，返回 MemoryRecord。"""
        ...

    async def search(self, query: str, *, namespace: str = "default",
                     limit: int = 5) -> list[MemoryRecord]:
        """语义检索相关记忆。"""
        ...

    async def delete(self, memory_id: str) -> None:
        """删除一条记忆。"""
        ...

    async def list_all(self, *, namespace: str = "default") -> list[MemoryRecord]:
        """列出所有记忆（调试用）。"""
        ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryMemoryStore:
    """内存版长期记忆存储，零外部依赖。"""

    def __init__(self, embedder: Embedder | None = None):
        self._records: dict[str, MemoryRecord] = {}
        self._embedder = embedder

    async def add(self, content: str, *, namespace: str = "default",
                  memory_type: str = "fact", importance: float = 0.5,
                  confidence: float = 0.8, tags: list[str] | None = None,
                  metadata: dict | None = None) -> MemoryRecord:
        from datetime import datetime, timezone
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

    async def search(self, query: str, *, namespace: str = "default",
                     limit: int = 5) -> list[MemoryRecord]:
        candidates = [r for r in self._records.values() if r.namespace == namespace]
        if not candidates:
            return []

        if self._embedder:
            # 向量检索
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
            # 关键词匹配退化
            query_lower = query.lower()
            scored = []
            for r in candidates:
                overlap = sum(1 for word in query_lower.split() if word in r.content.lower())
                score = overlap * 0.5 + r.importance * 0.3 + r.confidence * 0.2
                scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [r for _, r in scored[:limit]]

        # 更新访问统计
        from datetime import datetime, timezone
        for r in results:
            r.last_accessed = datetime.now(timezone.utc)
            r.access_count += 1

        return results

    async def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    async def list_all(self, *, namespace: str = "default") -> list[MemoryRecord]:
        return [r for r in self._records.values() if r.namespace == namespace]
```

要点：
- 有 Embedder 时用余弦相似度 + importance/confidence 加权排序
- 没有 Embedder 时退化为关键词匹配，照样能用
- `access_count` 和 `last_accessed` 会自动更新，方便后续做衰减

---

## 第四步：知识库存储（RAG）

新建 `wuwei/memory/knowledge_store.py`：

```python
from typing import Protocol
from uuid import uuid4

from wuwei.memory.memory_types import KnowledgeChunk
from wuwei.memory.embedder import Embedder
from wuwei.memory.memory_store import _cosine_similarity


class KnowledgeStore(Protocol):
    """RAG 知识库存储协议。"""

    async def ingest(self, text: str, source: str, *, namespace: str = "default",
                     title: str | None = None, chunk_size: int = 800,
                     chunk_overlap: int = 100) -> list[KnowledgeChunk]:
        """将文本切块并存入知识库。返回所有生成的 chunk。"""
        ...

    async def search(self, query: str, *, namespace: str = "default",
                     limit: int = 4) -> list[KnowledgeChunk]:
        """语义检索相关片段。"""
        ...

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        """按来源文件删除所有相关 chunk。"""
        ...


def _split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """按字符数切块，保留 overlap 重叠。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


class InMemoryKnowledgeStore:
    """内存版知识库，零外部依赖。"""

    def __init__(self, embedder: Embedder | None = None):
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._embedder = embedder

    async def ingest(self, text: str, source: str, *, namespace: str = "default",
                     title: str | None = None, chunk_size: int = 800,
                     chunk_overlap: int = 100) -> list[KnowledgeChunk]:
        # 先删除同一来源的旧数据
        await self.delete_by_source(source, namespace=namespace)

        pieces = _split_text(text, chunk_size, chunk_overlap)
        chunks = []

        embeddings = None
        if self._embedder and pieces:
            embeddings = await self._embedder.embed_texts(pieces)

        for i, piece in enumerate(pieces):
            chunk = KnowledgeChunk(
                id=uuid4().hex,
                text=piece,
                source=source,
                namespace=namespace,
                title=title,
                embedding=embeddings[i] if embeddings else None,
            )
            self._chunks[chunk.id] = chunk
            chunks.append(chunk)

        return chunks

    async def search(self, query: str, *, namespace: str = "default",
                     limit: int = 4) -> list[KnowledgeChunk]:
        candidates = [c for c in self._chunks.values() if c.namespace == namespace]
        if not candidates:
            return []

        if self._embedder:
            query_vec = await self._embedder.embed_query(query)
            scored = []
            for c in candidates:
                if c.embedding is not None:
                    sim = _cosine_similarity(query_vec, c.embedding)
                    scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]
        else:
            # 关键词匹配退化
            query_lower = query.lower()
            scored = []
            for c in candidates:
                overlap = sum(1 for w in query_lower.split() if w in c.text.lower())
                scored.append((overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]

    async def delete_by_source(self, source: str, *, namespace: str = "default") -> None:
        to_delete = [cid for cid, c in self._chunks.items()
                     if c.source == source and c.namespace == namespace]
        for cid in to_delete:
            del self._chunks[cid]
```

要点：
- `ingest()` 会自动切块、自动 embed、自动删除同来源旧数据
- chunk_size 默认 800 字符，overlap 100 字符，适合大多数文档
- 没有 Embedder 时退化为关键词匹配

---

## 第五步：Hook — 记忆检索

在 `wuwei/runtime/memory_hook.py` 中实现：

```python
from __future__ import annotations
from typing import TYPE_CHECKING

from wuwei.runtime.hooks import RuntimeHook

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.llm import Message
    from wuwei.memory.memory_store import MemoryStore
    from wuwei.tools import Tool


class MemoryRetrievalHook(RuntimeHook):
    """在每次 LLM 调用前，检索相关长期记忆并注入 system prompt。"""

    def __init__(self, store: "MemoryStore", *, top_k: int = 3, namespace: str = "default"):
        self.store = store
        self.top_k = top_k
        self.namespace = namespace

    async def before_llm(self, session: "AgentSession", messages: list["Message"],
                         tools: list["Tool"], *, step: int, task=None):
        # 从最近几条消息拼出查询词
        recent_texts = []
        for msg in messages[-6:]:
            if msg.role in ("user", "assistant") and msg.content:
                recent_texts.append(msg.content[:200])
        query = " ".join(recent_texts)
        if not query.strip():
            return messages, tools

        # 检索
        records = await self.store.search(query, namespace=self.namespace, limit=self.top_k)
        if not records:
            return messages, tools

        # 格式化注入
        lines = ["以下是可能相关的长期记忆，仅在与当前任务相关时使用："]
        for r in records:
            tag = f"[{r.memory_type}]"
            lines.append(f"- {tag} {r.content}")
        injection = "\n".join(lines)

        # 插入到 system message 之后
        from wuwei.llm.types import Message as LLMMessage
        memory_msg = LLMMessage(role="system", content=injection)

        new_messages = []
        system_inserted = False
        for msg in messages:
            new_messages.append(msg)
            if msg.role == "system" and not system_inserted:
                new_messages.append(memory_msg)
                system_inserted = True
        if not system_inserted:
            new_messages.insert(0, memory_msg)

        return new_messages, tools
```

工作原理：
1. 拿最近 6 条消息拼成查询词
2. 去 MemoryStore 语义检索 top_k 条
3. 格式化为 `[fact] xxx` 的列表，插入 system message 后面
4. 对 LLM 来说，就像 system prompt 里多了一段"记忆上下文"

---

## 第六步：Hook — 记忆抽取

在同一个文件中继续：

```python
import json
from wuwei.runtime.hooks import RuntimeHook
from wuwei.memory.memory_store import MemoryStore


EXTRACTION_PROMPT = """分析以下对话，提取值得长期记住的信息。

只提取以下类型的信息：
- preference: 用户明确表达的偏好（如"我喜欢简洁的回答"）
- fact: 已确认的稳定事实（如"项目用 Python 3.12"）
- constraint: 长期约束（如"代码不要用第三方库"）
- summary: 对话中达成的重要结论

不要提取：
- 临时任务、一次性问答
- 模型的猜测或不确定的信息
- 已经是常识的信息

输出 JSON 数组，每条包含 type 和 content。最多 3 条。如果没有值得记住的信息，输出空数组 []。

对话内容：
{conversation}

输出（纯 JSON，不要 markdown）："""


class MemoryExtractionHook(RuntimeHook):
    """在一轮运行结束后，用 LLM 从对话中抽取值得记住的记忆。"""

    def __init__(self, llm, store: MemoryStore, *, namespace: str = "default",
                 max_memories_per_run: int = 3):
        self.llm = llm
        self.store = store
        self.namespace = namespace
        self.max_memories = max_memories_per_run

    async def after_ai_message(self, session, message, *, step, task=None):
        # 只在最后一步（没有更多 tool_calls）时抽取
        # 简化处理：每轮都尝试抽取，靠 prompt 控制输出数量

        # 收集最近对话
        messages = session.context.get_messages()
        recent = messages[-10:]  # 最近 10 条
        conversation = "\n".join(
            f"{m.role}: {m.content[:300]}" for m in recent if m.content
        )
        if len(conversation) < 50:
            return  # 太短的对话不抽取

        # 调用 LLM 抽取
        prompt = EXTRACTION_PROMPT.format(conversation=conversation)
        from wuwei.llm.types import Message as LLMMessage
        response = await self.llm.generate([LLMMessage(role="user", content=prompt)])

        # 解析结果
        try:
            text = response.content.strip()
            # 去掉可能的 markdown 代码块标记
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            items = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            return  # 解析失败就跳过

        if not isinstance(items, list):
            return

        # 去重后写入
        for item in items[:self.max_memories]:
            if not isinstance(item, dict) or "content" not in item:
                continue
            content = item["content"].strip()
            if not content:
                continue

            # 简单去重：搜索已有记忆，如果高度相似就跳过
            existing = await self.store.search(content, namespace=self.namespace, limit=1)
            if existing and existing[0].content.lower() == content.lower():
                continue

            await self.store.add(
                content=content,
                namespace=self.namespace,
                memory_type=item.get("type", "fact"),
                importance=item.get("importance", 0.5),
                confidence=0.7,  # LLM 抽取的记忆 confidence 稍低
            )
```

工作原理：
1. 每轮 AI 回复后，收集最近 10 条消息
2. 用 LLM 分析对话，提取值得记住的信息（输出 JSON）
3. 和已有记忆去重后写入 MemoryStore
4. 每次最多存 3 条，避免记忆膨胀

---

## 第七步：Hook — RAG 检索

新建 `wuwei/runtime/rag_hook.py`：

```python
from __future__ import annotations
from typing import TYPE_CHECKING

from wuwei.runtime.hooks import RuntimeHook

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.llm import Message
    from wuwei.memory.knowledge_store import KnowledgeStore
    from wuwei.tools import Tool


class RagRetrievalHook(RuntimeHook):
    """在每次 LLM 调用前，从知识库检索相关片段并注入 system prompt。"""

    def __init__(self, store: "KnowledgeStore", *, top_k: int = 4,
                 max_chars_per_chunk: int = 500, namespace: str = "default"):
        self.store = store
        self.top_k = top_k
        self.max_chars = max_chars_per_chunk
        self.namespace = namespace

    async def before_llm(self, session: "AgentSession", messages: list["Message"],
                         tools: list["Tool"], *, step: int, task=None):
        # 取最近的用户消息作为查询
        user_texts = [m.content[:300] for m in messages[-4:]
                      if m.role == "user" and m.content]
        query = " ".join(user_texts)
        if not query.strip():
            return messages, tools

        chunks = await self.store.search(query, namespace=self.namespace, limit=self.top_k)
        if not chunks:
            return messages, tools

        # 格式化
        lines = ["以下是检索到的参考资料，如不相关可忽略：", ""]
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text[:self.max_chars]
            source = chunk.source
            title = chunk.title or ""
            header = f"[{i}] source={source}"
            if title:
                header += f" title={title}"
            lines.append(header)
            lines.append(text)
            lines.append("")

        injection = "\n".join(lines)

        from wuwei.llm.types import Message as LLMMessage
        rag_msg = LLMMessage(role="system", content=injection)

        new_messages = []
        system_inserted = False
        for msg in messages:
            new_messages.append(msg)
            if msg.role == "system" and not system_inserted:
                new_messages.append(rag_msg)
                system_inserted = True
        if not system_inserted:
            new_messages.insert(0, rag_msg)

        return new_messages, tools
```

---

## 第八步：给 RuntimeHook 加 on_run_end

目前 `RuntimeHook` 没有 `on_run_end` 回调。`MemoryExtractionHook` 放在 `after_ai_message` 也能用，但语义上放在 run 结束后更合适。

在 `wuwei/runtime/hooks.py` 的 `RuntimeHook` 类中加：

```python
async def on_run_end(self, session, result, *, task=None) -> None:
    """一轮 run 完全结束后调用。"""
    pass
```

然后在 `HookManager` 中加：

```python
async def on_run_end(self, session, result, *, task=None) -> None:
    for hook in self._hooks:
        await hook.on_run_end(session, result, task=task)
```

最后在 `AgentRunner._run_non_stream` 和 `_run_stream` 的末尾调用：

```python
await self.hooks.on_run_end(self.session, result)
```

这样 `MemoryExtractionHook` 就可以把 `after_ai_message` 改为 `on_run_end`，只在整轮结束后抽取一次。

---

## 第九步：文档导入工具（RAG）

RAG 还需要一个导入文档的入口。最简单的方式是提供一个工具：

在 `wuwei/tools/builtin/` 下新建 `rag_tools.py`：

```python
from pathlib import Path
from wuwei.memory.knowledge_store import KnowledgeStore


def register_rag_tools(registry, *, knowledge_store: KnowledgeStore):
    @registry.tool(
        name="ingest_document",
        description="将文档导入知识库。支持 txt/md 文件。导入后可用于 RAG 检索。",
    )
    async def ingest_document(file_path: str) -> dict:
        path = Path(file_path).resolve()
        if not path.exists():
            return {"ok": False, "error": f"文件不存在: {file_path}"}
        if path.suffix not in (".txt", ".md", ".markdown"):
            return {"ok": False, "error": "仅支持 .txt / .md 文件"}

        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = await knowledge_store.ingest(text, source=str(path))
        return {"ok": True, "chunks": len(chunks), "source": str(path)}

    @registry.tool(
        name="search_knowledge",
        description="从知识库中检索相关文档片段。",
    )
    async def search_knowledge(query: str, limit: int = 4) -> dict:
        chunks = await knowledge_store.search(query, limit=limit)
        results = []
        for c in chunks:
            results.append({
                "source": c.source,
                "text": c.text[:500],
            })
        return {"ok": True, "results": results}
```

然后在 `__init__.py` 的 `BUILTIN_TOOL_REGISTRARS` 中注册：

```python
"rag": register_rag_tools,
```

使用时：

```python
knowledge_store = InMemoryKnowledgeStore(embedder)
registry = ToolRegistry.from_builtin(["rag"], knowledge_store=knowledge_store)
```

---

## 完整使用示例

```python
import asyncio
from wuwei import Agent, LLMGateway, ToolRegistry
from wuwei.memory.memory_store import InMemoryMemoryStore
from wuwei.memory.knowledge_store import InMemoryKnowledgeStore
from wuwei.memory.embedder import OpenAIEmbedder
from wuwei.runtime.memory_hook import MemoryRetrievalHook, MemoryExtractionHook
from wuwei.runtime.rag_hook import RagRetrievalHook


async def main():
    # 1. 创建 Embedder（可选，不传则退化为关键词匹配）
    llm = LLMGateway.from_env()
    embedder = OpenAIEmbedder(api_key="sk-xxx")

    # 2. 创建存储
    memory_store = InMemoryMemoryStore(embedder)
    knowledge_store = InMemoryKnowledgeStore(embedder)

    # 3. 导入文档到知识库
    await knowledge_store.ingest(
        "Wuwei 是一个轻量 Python Agent 框架...",
        source="README.md",
    )

    # 4. 创建 Agent，挂载 Hook
    agent = Agent(
        llm=llm,
        tools=ToolRegistry.from_builtin(["time", "rag"], knowledge_store=knowledge_store),
        hooks=[
            MemoryRetrievalHook(memory_store),        # 每次 LLM 调用前注入记忆
            RagRetrievalHook(knowledge_store),         # 每次 LLM 调用前注入知识片段
            MemoryExtractionHook(llm, memory_store),   # 每轮结束后抽取新记忆
        ],
    )

    # 5. 正常使用
    session = agent.create_session()
    result = await agent.run("介绍一下 wuwei 的架构", session=session)
    print(result.content)

    # 第二轮对话，Agent 会自动记得上一轮的信息
    result = await agent.run("刚才我们聊了什么？", session=session)
    print(result.content)


asyncio.run(main())
```

---

## Hook 注册顺序

推荐顺序：

```
1. MemoryRetrievalHook    # 先注入记忆
2. RagRetrievalHook       # 再注入知识
3. ContextCompressionHook # 然后压缩裁剪
4. StorageHook            # 持久化
5. MemoryExtractionHook   # 最后抽取新记忆
```

原因：记忆和知识要在上下文压缩之前注入，否则可能被裁掉。

---

## 后续扩展（现在不做）

- 接入 ChromaDB / Qdrant 等向量数据库（实现 `MemoryStore` / `KnowledgeStore` 即可）
- 记忆衰减：根据 `last_accessed` 和 `access_count` 自动降权
- 记忆合并：定期用 LLM 合并相似记忆
- RAG rerank：检索后用 rerank 模型重新排序
- 混合检索：向量 + BM25 关键词检索结合
