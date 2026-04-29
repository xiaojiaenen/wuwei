# Wuwei 长期记忆与 RAG 最小方案

本文给出一份适合 Wuwei 的长期记忆与 RAG 设计方案，目标是：

- 保持框架边界清晰
- 先做最小可用版本
- 不把 MySQL、向量库、Embedding 服务写死到 core
- 尽量复用现有 `session / runtime hook / storage / tools` 结构

这份方案偏框架设计，不是业务产品设计。

## 1. 目标

希望框架原生支持两类能力：

1. 长期记忆
   - 记住用户偏好、稳定事实、项目约束、重要结论
   - 在后续对话中按需取回

2. RAG
   - 从知识库、文档、代码片段中检索相关内容
   - 把检索结果注入当前 prompt

最小版本不追求：

- 多租户后台
- 复杂权限系统
- 自动知识同步平台
- 重排序模型、混合检索、图谱检索
- 强绑定某个向量数据库

## 2. 设计原则

### 2.1 记忆和 RAG 分开

不要把“长期记忆”和“RAG”混成一个 store。

- 长期记忆更像“少量、高价值、可持续复用的信息”
- RAG 更像“外部知识片段检索”

两者虽然都可以在 `before_llm` 阶段注入上下文，但来源、生命周期、写入方式都不同。

建议在框架里分成两套协议：

- `MemoryStore`
- `KnowledgeStore`

### 2.2 Core 只定义协议

Wuwei core 只负责定义：

- 数据模型
- 存取协议
- Hook 扩展点
- 默认轻量实现

不要在 core 中直接依赖：

- Qdrant
- Milvus
- pgvector
- Elasticsearch
- OpenSearch
- 指定 Embedding 供应商

这些都应该作为 adapter 或 example。

### 2.3 默认先做简单检索

第一版不要追求“最强检索”，而要追求“最稳接口”。

建议默认分两层：

1. 抽象协议
2. 一个非常轻量的默认实现

默认实现可以是：

- 长期记忆：`InMemoryMemoryStore`
- RAG：`InMemoryKnowledgeStore`

这样框架测试、文档示例、接口稳定性都可以先跑起来。

## 3. 建议的模块边界

建议新增这些文件：

```text
wuwei/
  memory/
    memory_store.py       # 长期记忆协议 + 默认实现
    knowledge_store.py    # RAG 知识库协议 + 默认实现
    embedder.py           # Embedding 协议
    memory_types.py       # MemoryRecord / KnowledgeChunk 等数据模型
  runtime/
    memory_hook.py        # 记忆检索 / 记忆抽取
    rag_hook.py           # RAG 检索注入
```

如果你想更少文件，也可以先合并成：

```text
wuwei/
  memory/
    stores.py
    types.py
  runtime/
    memory_hook.py
```

但从长期维护看，分开会更清楚。

## 4. 数据模型

## 4.1 长期记忆

长期记忆建议是结构化记录，而不是单纯字符串。

```python
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class MemoryRecord:
    id: str
    content: str
    memory_type: str = "fact"
    namespace: str = "default"
    confidence: float = 0.8
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
```

建议支持这些 `memory_type`：

- `fact`
- `preference`
- `constraint`
- `summary`
- `warning`

最小版本先不要做太多类型，够用即可。

## 4.2 RAG 知识块

RAG 建议单独定义 chunk 结构：

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeChunk:
    id: str
    text: str
    source: str
    namespace: str = "default"
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

这里不要过早绑定：

- embedding 向量字段
- 分数计算方式
- 存储引擎字段

这些应该留给具体 store 实现决定。

## 5. 协议设计

## 5.1 Embedding 协议

RAG 迟早会需要 embedding，但不要把 embedding 逻辑塞进 store 里。

```python
from typing import Protocol


class Embedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...
```

这样以后接 OpenAI、本地模型、第三方服务都方便。

## 5.2 长期记忆协议

```python
from typing import Protocol, Any


class MemoryStore(Protocol):
    async def add(
        self,
        content: str,
        *,
        namespace: str = "default",
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        ...

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        ...

    async def delete(self, memory_id: str) -> None:
        ...
```

最小版本先保留：

- `add`
- `search`
- `delete`

不要一开始就塞进：

- update
- batch upsert
- TTL
- soft delete
- 审计日志

这些以后再扩。

## 5.3 RAG 知识库协议

```python
from typing import Protocol, Any


class KnowledgeStore(Protocol):
    async def upsert_chunks(
        self,
        chunks: list[KnowledgeChunk],
        *,
        namespace: str = "default",
    ) -> None:
        ...

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[KnowledgeChunk]:
        ...

    async def delete_by_source(
        self,
        source: str,
        *,
        namespace: str = "default",
    ) -> None:
        ...
```

`upsert_chunks()` 比 “只 add 不 upsert” 更适合 RAG，因为文档重建索引是常见操作。

## 6. Hook 接入方案

Wuwei 现在最合适的接入点仍然是 runtime hook。

## 6.1 MemoryRetrievalHook

职责：

- 在 `before_llm` 阶段检索相关长期记忆
- 注入一条 system message

行为建议：

- 从最近几条 user/assistant 消息拼查询词
- 用 `session.metadata` 作为过滤条件的一部分
- 最多注入 3 到 5 条记忆

注入格式建议保持简单：

```text
以下是可能相关的长期记忆，仅在与当前任务相关时使用：
- [preference] 用户偏好简洁回答
- [constraint] 当前项目 Python 最低版本为 3.10
- [fact] 上次已经确认采用文件存储会话历史
```

不要把完整 JSON 原样塞给模型。

## 6.2 MemoryExtractionHook

职责：

- 在一轮运行结束后，从最近对话中抽取值得长期保存的记忆

最小版本建议：

- 先用 LLM 抽取
- 严格限制只提取“稳定信息”
- 输出 JSON
- 抽取条数限制为 0 到 3 条

建议只保存：

- 用户明确偏好
- 稳定事实
- 项目长期约束
- 已确认的重要决策

不要保存：

- 临时任务
- 一次性问答
- 短期状态
- 模型猜测

### 是否需要新 Hook 生命周期

建议加一个新的运行结束钩子：

```python
async def on_run_end(self, session, result, *, task=None) -> None:
    ...
```

原因是长期记忆抽取更适合在一次 run 真正结束后做，而不是绑在 `after_ai_message` 上。

如果你暂时不想改 Hook 生命周期，也可以先在：

- `after_ai_message`

里做一个简化版本，但长期看不如 `on_run_end` 清晰。

## 6.3 RagRetrievalHook

职责：

- 在 `before_llm` 阶段从知识库检索文档片段
- 将片段压缩后注入 prompt

建议注入格式：

```text
以下是检索到的参考资料，如不相关可忽略：

[1] source=docs/framework_flow.md
AgentRunner 负责执行循环，按 hook -> llm -> tool 的顺序推进...

[2] source=README.md
Wuwei 是一个轻量、可扩展的 Python Agent 框架...
```

建议限制：

- chunk 数量最多 3 到 6 条
- 每条 chunk 限制字符数
- 总注入内容限制在一个较小范围内

第一版不做 rerank 也完全可以。

## 7. 检索策略

## 7.1 长期记忆检索

长期记忆适合“轻量语义检索 + metadata 过滤”。

最小策略：

1. 用 query embedding 做相似度召回
2. 按 `namespace` 过滤
3. 可选按 `session.metadata` 过滤
4. 简单按 `importance + confidence + similarity` 排序

如果暂时不做 embedding，也可以先用：

- 关键词匹配
- tags 匹配
- 简单 BM25 或朴素打分

## 7.2 RAG 检索

RAG 比长期记忆更偏“文档找片段”。

最小策略：

1. 文档切块
2. 建立向量索引
3. query embedding 检索 top-k
4. 拼成参考上下文

第一版参数建议：

- chunk size: 500 到 1000 字符
- overlap: 50 到 150 字符
- top-k: 4

先用字符切块就够了，不必一开始就做语义分段。

## 8. 写入策略

## 8.1 长期记忆写入

长期记忆一定要“克制写入”。

建议规则：

- 每次 run 最多写 3 条
- 相似内容优先覆盖或跳过，不要无限增长
- 低 confidence 不写入

推荐最小去重策略：

1. 先检索相似记忆
2. 如果高度相似，则跳过或替换
3. 如果是新信息，则新增

## 8.2 RAG 写入

RAG 的写入不在运行时自动发生，建议明确分离：

- 运行时只负责检索
- 文档导入单独通过 ingestion 流程完成

也就是说，RAG 不建议在 agent 对话过程中随手写知识库。

这能避免：

- 数据污染
- 索引反复重建
- 用户对话内容和知识库内容边界混乱

## 9. 推荐的最小落地顺序

建议分三步做。

### 第一步：先做长期记忆协议和默认实现

交付内容：

- `MemoryRecord`
- `MemoryStore`
- `InMemoryMemoryStore`
- `MemoryRetrievalHook`

先做到“能取回”，不急着自动抽取。

### 第二步：再做 MemoryExtractionHook

交付内容：

- `MemoryExtractionHook`
- 最小 JSON 抽取 prompt
- 基础去重策略

到这一步，长期记忆就闭环了。

### 第三步：做最小 RAG

交付内容：

- `KnowledgeChunk`
- `KnowledgeStore`
- `Embedder`
- `RagRetrievalHook`
- 一个简单的文档切块导入示例

先不做：

- rerank
- hybrid search
- agent 自动写知识库

## 10. 推荐的默认实现

为了让框架示例能跑，建议内置两个极简实现。

## 10.1 InMemoryMemoryStore

特点：

- 用 Python list 或 dict 存记录
- 支持最简单的 search
- 可用于单元测试和文档示例

## 10.2 InMemoryKnowledgeStore

特点：

- 持有 chunk 列表
- 如果没有 embedding，就先用朴素关键词召回
- 如果有 embedder，再升级成向量检索

这样可以保证框架使用者不配置外部基础设施也能先跑通。

## 11. 对现有框架的改动建议

最小改动建议如下：

### 必做

- 新增 `MemoryStore`
- 新增 `KnowledgeStore`
- 新增 `MemoryRetrievalHook`
- 新增 `RagRetrievalHook`

### 推荐做

- 给 `RuntimeHook` 增加 `on_run_end`
- 在 `AgentRunner` 中调用 `on_run_end`

### 暂时不做

- Web 知识库管理后台
- 异步索引任务系统
- 复杂多路召回
- 自动 rerank
- 记忆图谱

## 12. API 使用示例

长期记忆：

```python
memory_store = InMemoryMemoryStore()

agent = Agent.from_env(
    hooks=[
        MemoryRetrievalHook(memory_store),
        MemoryExtractionHook(llm, memory_store),
    ],
)
```

RAG：

```python
knowledge_store = InMemoryKnowledgeStore()

agent = Agent.from_env(
    hooks=[
        RagRetrievalHook(knowledge_store),
    ],
)
```

二者一起使用：

```python
agent = Agent.from_env(
    hooks=[
        MemoryRetrievalHook(memory_store),
        RagRetrievalHook(knowledge_store),
        MemoryExtractionHook(llm, memory_store),
    ],
)
```

推荐注入顺序：

1. MemoryRetrievalHook
2. RagRetrievalHook
3. 其他上下文裁剪或压缩 Hook
4. MemoryExtractionHook

## 13. 最终建议

如果目标是“框架”，那长期记忆和 RAG 最重要的不是一次做全，而是把边界做对。

最适合 Wuwei 的路径是：

1. 把长期记忆和 RAG 分成两套协议
2. 都通过 hook 接入 runtime
3. Core 只提供协议和轻量默认实现
4. 外部向量库、数据库、Embedding 服务通过 adapter 扩展

这样做的好处是：

- 框架保持轻
- API 容易稳定
- 示例容易写
- 后续扩展到向量库也不会推翻当前设计

一句话总结：

长期记忆做“少量高价值信息的可持续召回”，RAG 做“外部知识片段检索”，两者共享接入方式，但不要共享同一个抽象。
