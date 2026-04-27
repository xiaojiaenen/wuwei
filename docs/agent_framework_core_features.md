# Wuwei Agent 框架核心能力接口设计指南

本文基于当前 Wuwei 代码结构，说明一个 **Agent 框架** 应如何设计上下文滑动、上下文压缩、对话历史持久化、长期记忆、HITL（Human-in-the-loop）等核心能力。

重点原则：**Wuwei 是框架，不是业务产品**。因此核心包应该提供抽象接口、默认轻量实现、Hook 扩展点和示例适配器，而不应该强绑定 `user_id`、`tenant_id`、MySQL、某个向量数据库或某套审批系统。

当前框架关键入口：

- `wuwei/agent/base.py`：`BaseSessionAgent` 负责 `llm/tools/hooks/session` 装配。
- `wuwei/agent/session.py`：`AgentSession` 保存会话配置、运行统计和 `Context`。
- `wuwei/memory/context.py`：`Context` 保存当前内存态消息历史。
- `wuwei/runtime/agent_runner.py`：`AgentRunner` 在每轮 LLM 前复制 `Context`，并通过 `before_llm` hook 改写 `messages/tools`。
- `wuwei/runtime/hooks.py`：`RuntimeHook` 是上下文治理、持久化、记忆检索、审批、预算控制的主要扩展点。
- `wuwei/llm/types.py`：`Message / ToolCall / LLMResponse / AgentEvent` 是跨模块数据契约。

## 1. 框架边界

### 1.1 不应该放进核心框架的东西

核心框架不应该假设：

- 使用者一定有用户系统，所以不应该固定 `user_id`。
- 使用者一定有租户系统，所以不应该固定 `tenant_id`。
- 使用者一定用 MySQL，所以不应该让核心依赖 `aiomysql`。
- 使用者一定用某个向量库，所以不应该在核心写死 Qdrant、Milvus、pgvector。
- 使用者一定用 Web 审批，所以不应该在核心绑定 HTTP/API/前端状态。

这些都应该通过 `metadata`、`filters`、`namespace`、接口适配器交给使用者决定。

### 1.2 核心框架应该提供的东西

核心框架应该提供：

1. 抽象协议：`HistoryStore`、`MemoryStore`、`ApprovalProvider`。
2. 默认实现：`InMemoryHistoryStore`、`SimpleContextWindow`。
3. Hook 封装：`ContextHook`、`HistoryHook`、`MemoryHook`、`HitlHook`。
4. 示例适配器：MySQL、Redis、向量库可以放在 `examples/` 或可选 `integrations/`。
5. 框架数据字段：只保留通用字段，例如 `session_id`、`metadata`、`namespace`、`tags`。

推荐整体结构：

```text
wuwei/
  memory/
    context.py                 # 已有：内存消息容器
    types.py                   # ContextState / MemoryRecord 等通用类型
    context_window.py          # 简单上下文窗口
    context_compressor.py      # 上下文压缩协议 + LLM 实现
    history_store.py           # HistoryStore 协议 + InMemory 实现
    memory_store.py            # MemoryStore 协议 + InMemory 实现
  runtime/
    hooks.py                   # 已有：RuntimeHook / HookManager
    context_hook.py            # ContextWindowHook / ContextCompressionHook
    history_hook.py            # 持久化 Hook
    memory_hook.py             # 长期记忆检索 / 抽取 Hook
    hitl.py                    # HITL 协议和 Hook
    budget_hook.py             # 预算控制 Hook
  integrations/                # 可选，不作为核心强依赖
    mysql/
      history_store.py
      schema.sql
examples/
  mysql_history_store_example.py
```

如果你想保持包极简，可以先不建 `integrations/`，只在 `examples/` 提供 MySQL 实现。

## 2. AgentSession 的框架化设计

当前 `AgentSession` 已经很轻量。建议只补通用字段，不补业务字段。

### 2.1 推荐改造

文件：`wuwei/agent/session.py`

```python
from dataclasses import dataclass, field
from typing import Any

from wuwei.memory import Context


@dataclass
class AgentSession:
    """保存一次会话的配置、状态和短期上下文。

    这是框架对象，不应该固定 user_id / tenant_id 等业务字段。
    业务系统需要的标识可以放进 metadata。
    """

    session_id: str
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 10
    parallel_tool_calls: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    last_usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    last_latency_ms: int = 0
    last_llm_calls: int = 0
    context: Context = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.context = Context()
        self.context.add_system_message(self.system_prompt)
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_latency_ms = 0
        self.last_llm_calls = 0
```

使用者可以这样传业务信息：

```python
session = agent.create_session(
    session_id="chat_001",
    system_prompt="你是一个代码助手",
)
session.metadata["user_id"] = "u_123"
session.metadata["workspace_id"] = "repo_wuwei"
session.metadata["channel"] = "cli"
```

框架内部不理解这些字段，只负责透传给 store、hook、policy。

## 3. 上下文滑动窗口

### 3.1 目标

当前 `Context.get_messages()` 返回完整历史，`AgentRunner._copy_messages()` 每轮都会把完整历史交给模型。滑动窗口要解决：

- 保留最近对话；
- 保留 `assistant(tool_calls) -> tool` 的合法结构；
- 截断过长工具返回；
- 不修改 `session.context` 里的完整历史。

### 3.2 第一版先简单做

第一版不需要做复杂 token 预算，也不需要引入 tokenizer。默认策略就三步：

```text
保留 system -> 按 turn 保留最近 N 轮 -> 截断过长 tool 输出
```

这样已经能解决大部分问题：

- 不会无限把完整历史塞给模型；
- 不会把 `tool` 消息和对应的 `assistant.tool_calls` 拆散；
- 不需要框架绑定某个 tokenizer；
- 行为非常容易解释和调参。

不推荐直接按 message 数裁剪：

```python
messages = messages[-20:]
```

因为它可能破坏工具调用结构。

推荐默认参数只保留这几个：

```python
max_recent_turns = 8   # 保留最近 8 轮
max_tool_chars = 8000  # 单条 tool 输出最多保留 8000 字符
```

`max_prompt_tokens`、精确 `TokenCounter`、按模型上下文窗口自动推导预算，都可以放到后续增强版。第一版先不要把核心路径做复杂。

### 3.3 简单配置

上下文窗口、压缩器和 runtime hook 不建议放在一个文件里。推荐拆成：

```text
wuwei/memory/context_window.py       # split_turns / SimpleContextWindow
wuwei/memory/context_compressor.py   # ContextCompressor / LLMContextCompressor
wuwei/runtime/context_hook.py        # ContextWindowHook / ContextCompressionHook
```

这样 `memory` 只处理消息和压缩能力，`runtime` 只负责接入 Agent 生命周期。

文件：`wuwei/memory/context_window.py`

```python
from __future__ import annotations

from dataclasses import dataclass

from wuwei.llm import Message


@dataclass
class ContextWindowConfig:
    max_recent_turns: int = 8
    max_tool_chars: int = 8_000
    include_summary: bool = True
```

### 3.4 Turn 切分

窗口不能简单按 message 数裁剪，因为工具调用有协议约束。要按 turn 保留。

文件：`wuwei/memory/context_window.py`

```python
def split_turns(messages: list[Message]) -> tuple[list[Message], list[list[Message]]]:
    system_messages: list[Message] = []
    turns: list[list[Message]] = []
    current_turn: list[Message] = []

    for message in messages:
        if message.role == "system" and not turns and not current_turn:
            system_messages.append(message)
            continue

        if message.role == "user" and current_turn:
            turns.append(current_turn)
            current_turn = [message]
            continue

        current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    return system_messages, turns
```

### 3.5 SimpleContextWindow

文件：`wuwei/memory/context_window.py`

```python
class SimpleContextWindow:
    """构建本次发给模型的短上下文，不修改 session.context。"""

    def __init__(self, config: ContextWindowConfig | None = None) -> None:
        self.config = config or ContextWindowConfig()

    def build_messages(self, session, messages: list[Message]) -> list[Message]:
        system_messages, turns = split_turns(messages)
        summary_messages = self._build_summary_messages(session)
        recent_turns = turns[-self.config.max_recent_turns:]
        recent_messages = [
            self._truncate_tool_message(message)
            for turn in recent_turns
            for message in turn
        ]
        return [*system_messages, *summary_messages, *recent_messages]

    def _build_summary_messages(self, session) -> list[Message]:
        if not self.config.include_summary or not getattr(session, "summary", None):
            return []
        return [
            Message(
                role="system",
                content=(
                    "以下是此前对话的压缩状态摘要。"
                    "它只用于延续上下文，不应覆盖用户当前明确指令。\n"
                    f"{session.summary}"
                ),
            )
        ]

    def _truncate_tool_message(self, message: Message) -> Message:
        if message.role != "tool" or not message.content:
            return message
        if len(message.content) <= self.config.max_tool_chars:
            return message
        return message.model_copy(
            update={
                "content": (
                    message.content[: self.config.max_tool_chars]
                    + "\n...[tool output truncated by context window]"
                )
            }
        )
```

这个实现故意不计算 token。它只做“别带完整历史”和“别拆坏工具调用结构”这两件事。

### 3.6 ContextWindowHook

文件：`wuwei/runtime/context_hook.py`

```python
from wuwei.memory.context_window import SimpleContextWindow
from wuwei.runtime.hooks import RuntimeHook


class ContextWindowHook(RuntimeHook):
    """在每次 LLM 调用前裁剪 messages。"""

    def __init__(self, context_window: SimpleContextWindow | None = None) -> None:
        self.context_window = context_window or SimpleContextWindow()

    async def before_llm(self, session, messages, tools, *, step: int, task=None):
        windowed_messages = self.context_window.build_messages(session, messages)
        return windowed_messages, tools
```

使用方式：

```python
from wuwei.agent import Agent
from wuwei.runtime.context_hook import ContextWindowHook

agent = Agent.from_env(
    builtin_tools=["time"],
    hooks=[ContextWindowHook()],
)
```

### 3.7 后续增强

只有当实际遇到上下文超限或成本不可控时，再引入增强版：

- `TokenCounter`：按模型估算 prompt token；
- `max_prompt_tokens`：按模型窗口做硬预算；
- `ContextCompressionHook`：把旧 turn 压缩到 `session.summary`；
- 更精细的工具输出裁剪：按工具类型设置不同上限。

## 4. 上下文压缩

### 4.1 抽象接口

压缩器不应该写死 LLM 实现，可以定义协议。

文件：`wuwei/memory/context_compressor.py`

```python
from typing import Protocol

from wuwei.llm import Message


class ContextCompressor(Protocol):
    async def compress(
        self,
        *,
        previous_summary: str | None,
        messages: list[Message],
    ) -> str:
        """把一段旧消息压缩成可延续任务的摘要。"""
        ...
```

### 4.2 LLM 摘要压缩器

文件：`wuwei/memory/context_compressor.py`

```python
from wuwei.llm import LLMGateway, Message


class LLMContextCompressor:
    def __init__(self, llm: LLMGateway, system_prompt: str | None = None) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or (
            "你是一个 Agent 上下文压缩器。"
            "你的任务是把旧对话压缩成后续可继续工作的状态摘要。"
            "不要编造历史中没有的信息。"
        )

    async def compress(
        self,
        *,
        previous_summary: str | None,
        messages: list[Message],
    ) -> str:
        history_text = self._format_messages(messages)
        prompt = f"""
请压缩以下 Agent 历史，输出简洁、结构化中文摘要。

必须保留：
- 用户目标
- 已确认事实和约束
- 用户偏好
- 已执行工具及关键结果
- 当前进度
- 待办事项
- 风险和阻塞

已有摘要：
{previous_summary or "无"}

待压缩历史：
{history_text}
""".strip()

        response = await self.llm.generate(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt),
            ]
        )
        return response.message.content or ""

    def _format_messages(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.content or ""
            if message.tool_calls:
                content += f"\n工具调用: {message.model_dump_json()}"
            lines.append(f"[{message.role}] {content}")
        return "\n\n".join(lines)
```

### 4.3 压缩 Hook

压缩 Hook 也可以先按 turn 数触发，不必第一版就做 token 统计。它只压缩旧 turn，最近几轮原样保留。

文件：`wuwei/runtime/context_hook.py`

```python
from wuwei.llm import Message
from wuwei.memory.context_compressor import ContextCompressor
from wuwei.memory.context_window import (
    ContextWindowConfig,
    SimpleContextWindow,
    split_turns,
)
from wuwei.runtime.hooks import RuntimeHook


class ContextCompressionHook(RuntimeHook):
    """历史 turn 过多时生成滚动摘要。"""

    METADATA_KEY = "_context_compressed_until_turn"

    def __init__(
        self,
        compressor: ContextCompressor,
        context_window: SimpleContextWindow | None = None,
        *,
        compress_after_turns: int = 16,
        keep_recent_turns: int = 4,
    ) -> None:
        if keep_recent_turns >= compress_after_turns:
            raise ValueError("keep_recent_turns must be smaller than compress_after_turns")

        self.compressor = compressor
        self.compress_after_turns = compress_after_turns
        self.keep_recent_turns = keep_recent_turns
        self.context_window = context_window or SimpleContextWindow(
            ContextWindowConfig(max_recent_turns=keep_recent_turns)
        )

    async def before_llm(self, session, messages, tools, *, step: int, task=None):
        _, turns = split_turns(messages)

        if len(turns) > self.compress_after_turns:
            await self._compress_old_turns(session, turns)

        return self._build_window(session, messages, tools)

    async def _compress_old_turns(self, session, turns: list[list[Message]]) -> None:
        metadata = getattr(session, "metadata", None)
        if metadata is None:
            metadata = {}
            session.metadata = metadata

        compress_until = len(turns) - self.keep_recent_turns
        compressed_until = metadata.get(self.METADATA_KEY, 0)

        if compressed_until >= compress_until:
            return

        turns_to_compress = turns[compressed_until:compress_until]
        session.summary = await self.compressor.compress(
            previous_summary=getattr(session, "summary", None),
            messages=self._flatten(turns_to_compress),
        )
        metadata[self.METADATA_KEY] = compress_until

    def _build_window(self, session, messages, tools):
        windowed_messages = self.context_window.build_messages(session, messages)
        return windowed_messages, tools

    def _flatten(self, turns: list[list[Message]]) -> list[Message]:
        return [message for turn in turns for message in turn]
```

这个 Hook 内部已经调用 `SimpleContextWindow`，所以启用它时通常不需要再额外挂 `ContextWindowHook`。两者二选一即可：

- 只想裁剪窗口：使用 `ContextWindowHook`；
- 想自动压缩旧上下文并裁剪窗口：使用 `ContextCompressionHook`。

使用方式：

```python
from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.memory.context_compressor import LLMContextCompressor
from wuwei.runtime.context_hook import ContextCompressionHook

llm = LLMGateway.from_env()
agent = Agent(
    llm=llm,
    hooks=[
        ContextCompressionHook(
            compressor=LLMContextCompressor(llm),
            compress_after_turns=16,
            keep_recent_turns=4,
        )
    ],
)
```

## 5. 对话历史持久化接口

### 5.1 为什么只提供接口

持久化属于业务基础设施。框架应该提供 `HistoryStore` 协议，但不应该规定使用者必须用 MySQL。

使用者可能选择：

- MySQL：传统业务系统，审计和报表方便；
- PostgreSQL：JSON/全文/pgvector 生态；
- Redis：短期 session 缓存；
- SQLite：本地 CLI；
- S3/OSS：低成本归档；
- 自研存储：企业内部系统。

### 5.2 通用数据类型

文件：`wuwei/memory/history_store.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from wuwei.agent.session import AgentSession
from wuwei.llm import Message


@dataclass
class StoredSession:
    session_id: str
    system_prompt: str
    max_steps: int = 10
    parallel_tool_calls: bool = False
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class StoredMessage:
    id: str | int | None
    session_id: str
    message: Message
    step: int | None = None
    task_id: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class StoredRun:
    run_id: str
    session_id: str
    status: str
    input: str | None = None
    output: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    llm_calls: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.3 HistoryStore 协议

文件：`wuwei/memory/history_store.py`

```python
class HistoryStore(Protocol):
    """对话历史持久化协议。

    框架只依赖这个协议，不依赖具体数据库。
    """

    async def save_session(self, session: AgentSession) -> None:
        """创建或更新 session 元数据。"""
        ...

    async def load_session(self, session_id: str) -> StoredSession | None:
        """读取 session 元数据。"""
        ...

    async def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        step: int | None = None,
        task_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        """追加一条消息。"""
        ...

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        before_id: str | int | None = None,
    ) -> list[StoredMessage]:
        """按时间顺序读取消息。"""
        ...

    async def update_summary(self, session_id: str, summary: str) -> None:
        """更新滚动摘要。"""
        ...

    async def record_run_start(
        self,
        session_id: str,
        *,
        input: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """记录一次 run 开始，返回 run_id。"""
        ...

    async def record_run_end(
        self,
        run_id: str,
        *,
        status: str,
        output: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: int = 0,
        llm_calls: int = 0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录一次 run 结束。"""
        ...
```

### 5.4 InMemoryHistoryStore 默认实现

文件：`wuwei/memory/history_store.py`

```python
from datetime import datetime
from uuid import uuid4


class InMemoryHistoryStore:
    """内存版 HistoryStore。

    适合测试、示例、CLI 原型。不适合生产持久化。
    """

    def __init__(self) -> None:
        self.sessions: dict[str, StoredSession] = {}
        self.messages: dict[str, list[StoredMessage]] = {}
        self.runs: dict[str, StoredRun] = {}
        self._message_id = 0

    async def save_session(self, session: AgentSession) -> None:
        now = datetime.utcnow()
        existing = self.sessions.get(session.session_id)
        self.sessions[session.session_id] = StoredSession(
            session_id=session.session_id,
            system_prompt=session.system_prompt,
            max_steps=session.max_steps,
            parallel_tool_calls=session.parallel_tool_calls,
            summary=session.summary,
            metadata=dict(session.metadata),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

    async def load_session(self, session_id: str) -> StoredSession | None:
        return self.sessions.get(session_id)

    async def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        step: int | None = None,
        task_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        self._message_id += 1
        stored_message = StoredMessage(
            id=self._message_id,
            session_id=session_id,
            message=message.model_copy(deep=True),
            step=step,
            task_id=task_id,
            metadata=dict(metadata or {}),
            created_at=datetime.utcnow(),
        )
        self.messages.setdefault(session_id, []).append(stored_message)
        return stored_message

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        before_id: str | int | None = None,
    ) -> list[StoredMessage]:
        items = list(self.messages.get(session_id, []))
        if before_id is not None:
            items = [item for item in items if item.id is not None and item.id < before_id]
        if limit is not None:
            items = items[-limit:]
        return items

    async def update_summary(self, session_id: str, summary: str) -> None:
        session = self.sessions.get(session_id)
        if session:
            session.summary = summary
            session.updated_at = datetime.utcnow()

    async def record_run_start(
        self,
        session_id: str,
        *,
        input: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = uuid4().hex
        self.runs[run_id] = StoredRun(
            run_id=run_id,
            session_id=session_id,
            status="running",
            input=input,
            metadata=dict(metadata or {}),
        )
        return run_id

    async def record_run_end(
        self,
        run_id: str,
        *,
        status: str,
        output: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: int = 0,
        llm_calls: int = 0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        run = self.runs[run_id]
        run.status = status
        run.output = output
        run.usage = dict(usage or {})
        run.latency_ms = latency_ms
        run.llm_calls = llm_calls
        run.error = error
        run.metadata.update(metadata or {})
```

### 5.5 HistoryPersistenceHook

当前 `RuntimeHook` 缺少 `on_run_start/on_run_end/after_user_message`。建议先扩展 hook 协议。

文件：`wuwei/runtime/hooks.py`

```python
class RuntimeHook:
    async def on_run_start(self, session, user_input: str, *, task=None) -> None:
        pass

    async def after_user_message(self, session, message, *, task=None) -> None:
        pass

    async def on_run_end(self, session, result, *, task=None) -> None:
        pass
```

`HookManager` 增加对应分发：

```python
class HookManager:
    async def on_run_start(self, session, user_input: str, *, task=None) -> None:
        for hook in self._hooks:
            await hook.on_run_start(session, user_input, task=task)

    async def after_user_message(self, session, message, *, task=None) -> None:
        for hook in self._hooks:
            await hook.after_user_message(session, message, task=task)

    async def on_run_end(self, session, result, *, task=None) -> None:
        for hook in self._hooks:
            await hook.on_run_end(session, result, task=task)
```

然后在 `AgentRunner._run_non_stream()` 中：

```python
await self.hooks.on_run_start(self.session, user_input, task=task)
context.add_user_message(user_input)
await self.hooks.after_user_message(
    self.session,
    context.get_last_message(),
    task=task,
)
```

在返回 result 前：

```python
result = self._build_run_result(...)
await self.hooks.on_run_end(self.session, result, task=task)
return result
```

Hook 实现：

文件：`wuwei/runtime/history_hook.py`

```python
from wuwei.llm import AgentRunResult, LLMResponse, Message
from wuwei.memory.history_store import HistoryStore
from wuwei.runtime.hooks import RuntimeHook


class HistoryPersistenceHook(RuntimeHook):
    """把会话、消息和运行结果写入 HistoryStore。"""

    def __init__(self, store: HistoryStore) -> None:
        self.store = store
        self._run_ids: dict[str, str] = {}

    async def on_run_start(self, session, user_input: str, *, task=None) -> None:
        await self.store.save_session(session)
        run_id = await self.store.record_run_start(
            session.session_id,
            input=user_input,
            metadata=self._metadata(session, task=task),
        )
        self._run_ids[session.session_id] = run_id

    async def after_user_message(self, session, message: Message, *, task=None) -> None:
        await self.store.append_message(
            session.session_id,
            message,
            metadata=self._metadata(session, task=task),
        )

    async def after_llm(self, session, response: LLMResponse, *, step: int, task=None) -> None:
        await self.store.append_message(
            session.session_id,
            response.message,
            step=step,
            metadata={
                **self._metadata(session, task=task),
                "finish_reason": response.finish_reason,
                "model": response.model,
                "usage": response.usage,
                "latency_ms": response.latency_ms,
            },
        )

    async def after_tool(self, session, tool_call, tool_message, *, step: int, task=None) -> None:
        await self.store.append_message(
            session.session_id,
            tool_message,
            step=step,
            metadata={
                **self._metadata(session, task=task),
                "tool_name": tool_call.function.name,
                "tool_arguments": tool_call.function.arguments,
            },
        )

    async def on_run_end(self, session, result: AgentRunResult, *, task=None) -> None:
        run_id = self._run_ids.pop(session.session_id, None)
        if not run_id:
            return
        await self.store.record_run_end(
            run_id,
            status="completed",
            output=result.content,
            usage=result.usage,
            latency_ms=result.latency_ms,
            llm_calls=result.llm_calls,
            metadata=self._metadata(session, task=task),
        )
        await self.store.save_session(session)

    def _metadata(self, session, *, task=None) -> dict:
        metadata = dict(getattr(session, "metadata", {}) or {})
        if task is not None:
            metadata["task_id"] = getattr(task, "id", None)
            metadata["task_name"] = getattr(task, "name", None)
        return metadata
```

注意：当前 `AgentRunner._append_tool_messages()` 会直接追加 tool message，但不会触发 `after_llm` 的 stream 分支。流式路径需要在完整 assistant message 拼接完成后补一个类似 `after_llm_stream_message` 或直接调用新的 `after_ai_message` hook。建议最终统一成更细的消息级 hook：

```python
async def after_ai_message(self, session, message: Message, *, step: int, task=None) -> None: ...
async def after_tool_message(self, session, message: Message, *, step: int, task=None) -> None: ...
```

## 6. MySQL 作为可选适配器示例

### 6.1 放置位置

MySQL 不建议放进核心依赖。可选方案：

```text
examples/mysql_history_store.py
```

或者：

```text
wuwei/integrations/mysql/history_store.py
```

`pyproject.toml` 中配置可选依赖：

```toml
[project.optional-dependencies]
mysql = ["aiomysql>=0.2.0"]
```

### 6.2 通用 MySQL 表结构

表结构不放 `user_id` 固定列，而是使用 `metadata JSON`。业务方需要高频查询某个字段时，可以自己增加索引列。

```sql
CREATE TABLE agent_sessions (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id VARCHAR(128) NOT NULL UNIQUE,
  system_prompt TEXT NOT NULL,
  max_steps INT NOT NULL DEFAULT 10,
  parallel_tool_calls BOOLEAN NOT NULL DEFAULT FALSE,
  summary MEDIUMTEXT NULL,
  metadata JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
);

CREATE TABLE agent_messages (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id VARCHAR(128) NOT NULL,
  role VARCHAR(32) NOT NULL,
  content MEDIUMTEXT NULL,
  tool_call_id VARCHAR(128) NULL,
  tool_calls JSON NULL,
  step INT NULL,
  task_id INT NULL,
  token_count INT NULL,
  metadata JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  INDEX idx_agent_messages_session_id_id (session_id, id),
  INDEX idx_agent_messages_session_created (session_id, created_at)
);

CREATE TABLE agent_runs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  run_id VARCHAR(128) NOT NULL UNIQUE,
  session_id VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL,
  input MEDIUMTEXT NULL,
  output MEDIUMTEXT NULL,
  prompt_tokens INT NOT NULL DEFAULT 0,
  completion_tokens INT NOT NULL DEFAULT 0,
  total_tokens INT NOT NULL DEFAULT 0,
  latency_ms INT NOT NULL DEFAULT 0,
  llm_calls INT NOT NULL DEFAULT 0,
  error TEXT NULL,
  metadata JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  finished_at DATETIME(6) NULL,
  INDEX idx_agent_runs_session_id (session_id)
);
```

业务方如果需要用户维度查询，可以自行扩展：

```sql
ALTER TABLE agent_sessions ADD COLUMN app_user_id VARCHAR(128) NULL;
CREATE INDEX idx_agent_sessions_app_user_id ON agent_sessions(app_user_id);
```

框架文档只建议，不强制。

### 6.3 MySQLHistoryStore 示例骨架

文件：`examples/mysql_history_store.py`

```python
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import aiomysql

from wuwei.agent.session import AgentSession
from wuwei.llm import Message, ToolCall
from wuwei.memory.history_store import StoredMessage, StoredSession


class MySQLHistoryStore:
    """HistoryStore 的 MySQL 示例实现。

    这是 adapter，不是核心框架必需组件。
    """

    def __init__(self, pool: aiomysql.Pool) -> None:
        self.pool = pool

    @classmethod
    async def create(
        cls,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        db: str,
        minsize: int = 1,
        maxsize: int = 10,
    ) -> "MySQLHistoryStore":
        pool = await aiomysql.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            db=db,
            minsize=minsize,
            maxsize=maxsize,
            autocommit=True,
        )
        return cls(pool)

    async def save_session(self, session: AgentSession) -> None:
        sql = """
        INSERT INTO agent_sessions (
          session_id, system_prompt, max_steps, parallel_tool_calls, summary, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          system_prompt = VALUES(system_prompt),
          max_steps = VALUES(max_steps),
          parallel_tool_calls = VALUES(parallel_tool_calls),
          summary = VALUES(summary),
          metadata = VALUES(metadata)
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql,
                    (
                        session.session_id,
                        session.system_prompt,
                        session.max_steps,
                        session.parallel_tool_calls,
                        session.summary,
                        json.dumps(session.metadata, ensure_ascii=False),
                    ),
                )

    async def load_session(self, session_id: str) -> StoredSession | None:
        sql = """
        SELECT session_id, system_prompt, max_steps, parallel_tool_calls,
               summary, metadata, created_at, updated_at
        FROM agent_sessions
        WHERE session_id = %s
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, (session_id,))
                row = await cursor.fetchone()

        if not row:
            return None

        return StoredSession(
            session_id=row["session_id"],
            system_prompt=row["system_prompt"],
            max_steps=row["max_steps"],
            parallel_tool_calls=bool(row["parallel_tool_calls"]),
            summary=row["summary"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        step: int | None = None,
        task_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        sql = """
        INSERT INTO agent_messages (
          session_id, role, content, tool_call_id, tool_calls,
          step, task_id, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql,
                    (
                        session_id,
                        message.role,
                        message.content,
                        message.tool_call_id,
                        self._dump_tool_calls(message.tool_calls),
                        step,
                        task_id,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
                message_id = cursor.lastrowid

        return StoredMessage(
            id=message_id,
            session_id=session_id,
            message=message.model_copy(deep=True),
            step=step,
            task_id=task_id,
            metadata=dict(metadata or {}),
            created_at=datetime.utcnow(),
        )

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        before_id: str | int | None = None,
    ) -> list[StoredMessage]:
        params: list[Any] = [session_id]
        where = "session_id = %s"
        if before_id is not None:
            where += " AND id < %s"
            params.append(before_id)

        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT %s"
            params.append(limit)

        sql = f"""
        SELECT id, session_id, role, content, tool_call_id, tool_calls,
               step, task_id, token_count, metadata, created_at
        FROM (
          SELECT * FROM agent_messages
          WHERE {where}
          ORDER BY id DESC
          {limit_sql}
        ) AS recent
        ORDER BY id ASC
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, tuple(params))
                rows = await cursor.fetchall()

        return [self._row_to_stored_message(row) for row in rows]

    async def update_summary(self, session_id: str, summary: str) -> None:
        sql = "UPDATE agent_sessions SET summary = %s WHERE session_id = %s"
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, (summary, session_id))

    async def record_run_start(
        self,
        session_id: str,
        *,
        input: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = uuid4().hex
        sql = """
        INSERT INTO agent_runs (run_id, session_id, status, input, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql,
                    (
                        run_id,
                        session_id,
                        "running",
                        input,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
        return run_id

    async def record_run_end(
        self,
        run_id: str,
        *,
        status: str,
        output: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: int = 0,
        llm_calls: int = 0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        usage = usage or {}
        sql = """
        UPDATE agent_runs
        SET status = %s,
            output = %s,
            prompt_tokens = %s,
            completion_tokens = %s,
            total_tokens = %s,
            latency_ms = %s,
            llm_calls = %s,
            error = %s,
            metadata = %s,
            finished_at = CURRENT_TIMESTAMP(6)
        WHERE run_id = %s
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql,
                    (
                        status,
                        output,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                        latency_ms,
                        llm_calls,
                        error,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        run_id,
                    ),
                )

    def _dump_tool_calls(self, tool_calls: list[ToolCall] | None) -> str | None:
        if not tool_calls:
            return None
        return json.dumps([tool_call.model_dump() for tool_call in tool_calls], ensure_ascii=False)

    def _load_tool_calls(self, raw: str | None) -> list[ToolCall] | None:
        if not raw:
            return None
        return [ToolCall.model_validate(item) for item in json.loads(raw)]

    def _row_to_stored_message(self, row: dict[str, Any]) -> StoredMessage:
        return StoredMessage(
            id=row["id"],
            session_id=row["session_id"],
            message=Message(
                role=row["role"],
                content=row["content"],
                tool_call_id=row["tool_call_id"],
                tool_calls=self._load_tool_calls(row["tool_calls"]),
            ),
            step=row["step"],
            task_id=row["task_id"],
            token_count=row["token_count"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
        )
```

### 6.4 从 Store 恢复 Session

建议新增一个小工具，而不是把数据库逻辑塞进 `AgentRunner`。

文件：`wuwei/agent/session_repository.py`

```python
from wuwei.agent.session import AgentSession
from wuwei.memory.history_store import HistoryStore


class SessionRepository:
    """负责从 HistoryStore 创建/恢复 AgentSession。"""

    def __init__(self, history_store: HistoryStore, recent_message_limit: int = 50) -> None:
        self.history_store = history_store
        self.recent_message_limit = recent_message_limit

    async def load(self, session_id: str) -> AgentSession | None:
        stored_session = await self.history_store.load_session(session_id)
        if not stored_session:
            return None

        session = AgentSession(
            session_id=stored_session.session_id,
            system_prompt=stored_session.system_prompt,
            max_steps=stored_session.max_steps,
            parallel_tool_calls=stored_session.parallel_tool_calls,
            metadata=dict(stored_session.metadata),
            summary=stored_session.summary,
        )

        session.context.reset()
        session.context.add_system_message(session.system_prompt)

        stored_messages = await self.history_store.list_messages(
            session_id,
            limit=self.recent_message_limit,
        )
        for stored_message in stored_messages:
            message = stored_message.message
            if message.role == "system":
                continue
            if message.role == "user":
                session.context.add_user_message(message.content or "")
            elif message.role == "assistant":
                session.context.add_ai_message(message.content, message.tool_calls)
            elif message.role == "tool":
                session.context.add_tool_message(message.content or "", message.tool_call_id)

        return session

    async def save(self, session: AgentSession) -> None:
        await self.history_store.save_session(session)
```

`BaseSessionAgent` 可以后续接收可选 repository：

```python
class BaseSessionAgent(BaseAgent):
    def __init__(..., session_repository: SessionRepository | None = None):
        self.session_repository = session_repository
```

由于 `create_or_get_session` 当前是同步方法，如果要从数据库异步恢复，可以新增异步方法：

```python
async def aload_session(self, session_id: str) -> AgentSession:
    if session_id in self._sessions:
        return self._sessions[session_id]
    if self.session_repository:
        session = await self.session_repository.load(session_id)
        if session:
            self._sessions[session.session_id] = session
            return session
    return self.create_session(session_id=session_id)
```

避免把现有同步 API 硬改成 async，减少破坏性。

## 7. 长期记忆接口

### 7.1 设计原则

长期记忆也不能固定 `user_id`。框架只提供：

- `namespace`：记忆命名空间，例如 `default`、`project:wuwei`。
- `filters`：使用者自定义过滤条件，例如 `{"user_id": "u_123"}`。
- `metadata`：使用者自定义字段。

### 7.2 MemoryRecord 和 MemoryStore

文件：`wuwei/memory/memory_store.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class MemoryRecord:
    id: str | None
    content: str
    namespace: str = "default"
    memory_type: str = "fact"
    importance: float = 0.5
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryStore(Protocol):
    """长期记忆协议。

    框架不关心用户体系和具体向量库。
    """

    async def add(
        self,
        content: str,
        *,
        namespace: str = "default",
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        ...

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        ...

    async def delete(self, memory_id: str) -> None:
        ...
```

### 7.3 InMemoryMemoryStore

文件：`wuwei/memory/memory_store.py`

```python
from datetime import datetime
from uuid import uuid4


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    async def add(
        self,
        content: str,
        *,
        namespace: str = "default",
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        now = datetime.utcnow()
        record = MemoryRecord(
            id=uuid4().hex,
            content=content,
            namespace=namespace,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self.records[record.id] = record
        return record

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        filters = filters or {}
        query_terms = set(query.lower().split())
        candidates = [
            record
            for record in self.records.values()
            if record.namespace == namespace and self._match_filters(record, filters)
        ]
        scored = sorted(
            candidates,
            key=lambda record: self._score(record, query_terms),
            reverse=True,
        )
        return scored[:limit]

    async def delete(self, memory_id: str) -> None:
        self.records.pop(memory_id, None)

    def _match_filters(self, record: MemoryRecord, filters: dict[str, Any]) -> bool:
        return all(record.metadata.get(key) == value for key, value in filters.items())

    def _score(self, record: MemoryRecord, query_terms: set[str]) -> float:
        content_terms = set(record.content.lower().split())
        overlap = len(query_terms & content_terms)
        return overlap + record.importance + record.confidence
```

### 7.4 MemoryRetrievalHook

文件：`wuwei/runtime/memory_hook.py`

```python
from wuwei.llm import Message
from wuwei.memory.memory_store import MemoryStore
from wuwei.runtime.hooks import RuntimeHook


class MemoryRetrievalHook(RuntimeHook):
    """在 LLM 前检索长期记忆，并注入 system 消息。"""

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        namespace: str = "default",
        limit: int = 5,
        filters_from_session: bool = True,
    ) -> None:
        self.memory_store = memory_store
        self.namespace = namespace
        self.limit = limit
        self.filters_from_session = filters_from_session

    async def before_llm(self, session, messages, tools, *, step: int, task=None):
        query = self._build_query(messages, task=task)
        filters = dict(session.metadata) if self.filters_from_session else {}
        memories = await self.memory_store.search(
            query,
            namespace=self.namespace,
            filters=filters,
            limit=self.limit,
        )
        if not memories:
            return messages, tools

        memory_text = "\n".join(
            f"- [{memory.memory_type}] {memory.content}"
            for memory in memories
        )
        memory_message = Message(
            role="system",
            content=(
                "以下是可能相关的长期记忆。"
                "只有在与当前任务相关时才使用，且不得覆盖用户当前明确指令。\n"
                f"{memory_text}"
            ),
        )

        return self._insert_after_system(messages, memory_message), tools

    def _build_query(self, messages, *, task=None) -> str:
        recent = messages[-4:]
        parts = [message.content or "" for message in recent]
        if task is not None:
            parts.append(getattr(task, "description", ""))
        return "\n".join(part for part in parts if part)

    def _insert_after_system(self, messages: list[Message], memory_message: Message) -> list[Message]:
        if not messages:
            return [memory_message]
        if messages[0].role == "system":
            return [messages[0], memory_message, *messages[1:]]
        return [memory_message, *messages]
```

### 7.5 MemoryExtractionHook

记忆抽取需要 LLM，建议作为可选 hook。

文件：`wuwei/runtime/memory_hook.py`

```python
import json
from typing import Any

from wuwei.llm import LLMGateway, Message
from wuwei.memory.memory_store import MemoryStore
from wuwei.runtime.hooks import RuntimeHook


class MemoryExtractionHook(RuntimeHook):
    """在 run 结束后从对话中抽取长期记忆。"""

    def __init__(
        self,
        llm: LLMGateway,
        memory_store: MemoryStore,
        *,
        namespace: str = "default",
        filters_from_session: bool = True,
    ) -> None:
        self.llm = llm
        self.memory_store = memory_store
        self.namespace = namespace
        self.filters_from_session = filters_from_session

    async def on_run_end(self, session, result, *, task=None) -> None:
        messages = session.context.get_messages()[-8:]
        memories = await self._extract(messages)
        metadata = dict(session.metadata) if self.filters_from_session else {}
        for memory in memories:
            await self.memory_store.add(
                memory["content"],
                namespace=self.namespace,
                memory_type=memory.get("type", "fact"),
                importance=float(memory.get("importance", 0.5)),
                confidence=float(memory.get("confidence", 0.8)),
                metadata=metadata,
            )

    async def _extract(self, messages: list[Message]) -> list[dict[str, Any]]:
        history = "\n".join(f"[{m.role}] {m.content or ''}" for m in messages)
        prompt = f"""
请从以下对话中抽取值得长期保存的记忆。
只抽取稳定事实、用户明确偏好、项目约束、重要经验。
不要保存临时闲聊或一次性问题。

返回严格 JSON：
{{
  "memories": [
    {{
      "type": "preference|fact|skill|warning|summary",
      "content": "...",
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}

对话：
{history}
""".strip()
        response = await self.llm.generate(
            [
                Message(role="system", content="你是长期记忆抽取器，只输出 JSON。"),
                Message(role="user", content=prompt),
            ]
        )
        try:
            payload = json.loads(response.message.content or "{}")
        except json.JSONDecodeError:
            return []
        memories = payload.get("memories", [])
        return memories if isinstance(memories, list) else []
```

## 8. HITL 接口设计

### 8.1 核心原则

HITL 是框架能力，不是产品审批系统。核心框架应该定义：

- 什么时候需要审批：`ApprovalPolicy`；
- 如何发起审批：`ApprovalProvider`；
- 审批结果是什么：`ApprovalDecision`；
- 如何接入工具调用：`HitlHook`。

不应该绑定 Web、数据库、企业 IM、前端页面。

### 8.2 数据类型和协议

文件：`wuwei/runtime/hitl.py`

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from wuwei.llm import ToolCall

ApprovalStatus = Literal["approved", "rejected", "pending"]


@dataclass
class ApprovalRequest:
    session_id: str
    action_type: str
    payload: dict[str, Any]
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalDecision:
    status: ApprovalStatus
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ApprovalProvider(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class ApprovalPolicy:
    """默认审批策略。

    业务方可以继承或替换。
    """

    def __init__(
        self,
        *,
        require_approval_tools: set[str] | None = None,
        auto_approve_tools: set[str] | None = None,
    ) -> None:
        self.require_approval_tools = require_approval_tools or set()
        self.auto_approve_tools = auto_approve_tools or set()

    def requires_tool_approval(self, tool_call: ToolCall, *, session, task=None) -> bool:
        tool_name = tool_call.function.name
        if tool_name in self.auto_approve_tools:
            return False
        if tool_name in self.require_approval_tools:
            return True
        return False
```

### 8.3 ConsoleApprovalProvider

文件：`wuwei/runtime/hitl.py`

```python
class ConsoleApprovalProvider:
    """命令行审批 provider。

    适合本地开发和示例。生产环境可以实现 Web/API/消息队列版 provider。
    """

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        tool_name = request.tool_call.function.name if request.tool_call else request.action_type
        print("\n[HITL] 需要人工审批")
        print(f"session_id: {request.session_id}")
        print(f"action: {tool_name}")
        print(f"payload: {request.payload}")
        answer = input("是否批准？输入 y/N: ").strip().lower()
        if answer == "y":
            return ApprovalDecision(status="approved", reason="approved from console")
        return ApprovalDecision(status="rejected", reason="rejected from console")
```

### 8.4 HitlHook

文件：`wuwei/runtime/hitl.py`

```python
from wuwei.runtime.hooks import RuntimeHook


class ToolApprovalRejected(Exception):
    pass


class HitlHook(RuntimeHook):
    def __init__(
        self,
        provider: ApprovalProvider,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy or ApprovalPolicy()

    async def before_tool(self, session, tool_call, *, step: int, task=None) -> None:
        if not self.policy.requires_tool_approval(tool_call, session=session, task=task):
            return

        request = ApprovalRequest(
            session_id=session.session_id,
            action_type="tool_call",
            tool_call=tool_call,
            payload={
                "tool_name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
                "step": step,
            },
            metadata=dict(getattr(session, "metadata", {}) or {}),
        )
        decision = await self.provider.request_approval(request)
        if decision.status != "approved":
            raise ToolApprovalRejected(decision.reason or "tool call rejected")
```

第一版可以抛异常。更完善的做法是让 `ToolExecutor` 返回一条 tool error message，这样模型可以继续改计划：

```json
{"ok": false, "error": "tool call rejected by human"}
```

### 8.5 事件流扩展

当前 `AgentEvent.type` 只有：

```python
Literal["text_delta", "tool_start", "tool_end", "done", "error"]
```

如果要做 Web HITL，建议扩展：

```python
Literal[
    "text_delta",
    "tool_start",
    "tool_end",
    "approval_required",
    "approval_resolved",
    "done",
    "error",
]
```

异步审批需要把 run 暂停、持久化 pending 状态、用户审批后 resume。这属于第二阶段，不建议第一版就做复杂。

## 9. 预算控制接口

文件：`wuwei/runtime/budget_hook.py`

```python
from dataclasses import dataclass

from wuwei.runtime.hooks import RuntimeHook


@dataclass
class BudgetConfig:
    max_prompt_tokens_per_call: int | None = None
    max_total_tokens_per_run: int | None = None
    max_llm_calls_per_run: int | None = None
    max_tool_calls_per_run: int | None = None


class BudgetExceeded(Exception):
    pass


class BudgetHook(RuntimeHook):
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self._llm_calls: dict[str, int] = {}
        self._tool_calls: dict[str, int] = {}

    async def before_llm(self, session, messages, tools, *, step: int, task=None):
        session_id = session.session_id
        self._llm_calls[session_id] = self._llm_calls.get(session_id, 0) + 1
        if (
            self.config.max_llm_calls_per_run is not None
            and self._llm_calls[session_id] > self.config.max_llm_calls_per_run
        ):
            raise BudgetExceeded("LLM call budget exceeded")
        return messages, tools

    async def before_tool(self, session, tool_call, *, step: int, task=None) -> None:
        session_id = session.session_id
        self._tool_calls[session_id] = self._tool_calls.get(session_id, 0) + 1
        if (
            self.config.max_tool_calls_per_run is not None
            and self._tool_calls[session_id] > self.config.max_tool_calls_per_run
        ):
            raise BudgetExceeded("tool call budget exceeded")

    async def on_run_end(self, session, result, *, task=None) -> None:
        self._llm_calls.pop(session.session_id, None)
        self._tool_calls.pop(session.session_id, None)
```

如果后续确实要按 token 控制，可以再引入 `TokenCounter`，在 `before_llm` 估算 prompt tokens，并在 `after_llm/on_run_end` 读取真实 usage。

## 10. 推荐落地顺序

### 第 1 阶段：上下文窗口

新增：

- `wuwei/memory/context_window.py`
- `wuwei/runtime/context_hook.py`

先只做：

- system 保留；
- 最近 turn 保留；
- tool output 截断；
- tool call 成组保留；
- 不修改 `session.context`，只修改本次 LLM 的 messages。

### 第 2 阶段：Hook 生命周期补齐

扩展 `RuntimeHook`：

- `on_run_start`
- `after_user_message`
- `after_ai_message`
- `after_tool_message`
- `on_run_end`

这样持久化、记忆抽取、审计、预算都不用侵入 `AgentRunner` 太多。

### 第 3 阶段：HistoryStore 抽象

新增：

- `wuwei/memory/history_store.py`
- `InMemoryHistoryStore`
- `HistoryPersistenceHook`
- `SessionRepository`

MySQL 只作为 `examples/mysql_history_store.py` 或 `integrations/mysql`。

### 第 4 阶段：上下文压缩

新增：

- `wuwei/memory/context_compressor.py`；
- `ContextCompressor` 协议；
- `LLMContextCompressor`；
- `ContextCompressionHook`，放在 `wuwei/runtime/context_hook.py`；
- `session.summary`；
- `history_store.update_summary()`。

### 第 5 阶段：长期记忆

新增：

- `MemoryStore` 协议；
- `InMemoryMemoryStore`；
- `MemoryRetrievalHook`；
- `MemoryExtractionHook`；
- 向量库作为 adapter，而不是核心依赖。

### 第 6 阶段：HITL

新增：

- `ApprovalPolicy`；
- `ApprovalProvider`；
- `ConsoleApprovalProvider`；
- `HitlHook`；
- 后续再扩展 `AgentEvent` 和异步 resume。

## 11. 最小 API 使用示例

### 11.1 只启用滑动窗口

```python
from wuwei.agent import Agent
from wuwei.runtime.context_hook import ContextWindowHook

agent = Agent.from_env(
    builtin_tools=["time"],
    hooks=[ContextWindowHook()],
)

session = agent.create_session(session_id="demo")
result = await agent.run("你好，记住我喜欢简洁回答", session=session)
```

### 11.2 启用持久化，但不绑定数据库

```python
from wuwei.agent import Agent
from wuwei.memory.history_store import InMemoryHistoryStore
from wuwei.runtime.history_hook import HistoryPersistenceHook

history_store = InMemoryHistoryStore()

agent = Agent.from_env(
    hooks=[HistoryPersistenceHook(history_store)],
)

session = agent.create_session(session_id="chat_001")
session.metadata["app_user_id"] = "u_123"
session.metadata["workspace"] = "wuwei"

await agent.run("帮我解释这个框架", session=session)
```

### 11.3 使用业务方 MySQL adapter

```python
from examples.mysql_history_store import MySQLHistoryStore
from wuwei.agent import Agent
from wuwei.runtime.history_hook import HistoryPersistenceHook

store = await MySQLHistoryStore.create(
    host="127.0.0.1",
    user="root",
    password="password",
    db="agent_history",
)

agent = Agent.from_env(
    hooks=[HistoryPersistenceHook(store)],
)

session = agent.create_session(session_id="chat_001")
session.metadata.update({
    "app_user_id": "u_123",
    "project_id": "p_wuwei",
})

await agent.run("继续上次的话题", session=session)
```

### 11.4 启用长期记忆

```python
from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.memory.memory_store import InMemoryMemoryStore
from wuwei.runtime.memory_hook import MemoryExtractionHook, MemoryRetrievalHook

llm = LLMGateway.from_env()
memory_store = InMemoryMemoryStore()

agent = Agent(
    llm=llm,
    hooks=[
        MemoryRetrievalHook(memory_store, namespace="project:wuwei"),
        MemoryExtractionHook(llm, memory_store, namespace="project:wuwei"),
    ],
)

session = agent.create_session(session_id="chat_001")
session.metadata["app_user_id"] = "u_123"

await agent.run("以后回答默认简洁一点", session=session)
```

### 11.5 启用 HITL

```python
from wuwei.agent import Agent
from wuwei.runtime.hitl import ApprovalPolicy, ConsoleApprovalProvider, HitlHook

agent = Agent.from_env(
    builtin_tools=["file"],
    hooks=[
        HitlHook(
            provider=ConsoleApprovalProvider(),
            policy=ApprovalPolicy(require_approval_tools={"write_file", "delete_file"}),
        )
    ],
)

await agent.run("删除临时文件")
```

## 12. 关键结论

Wuwei 作为 Agent 框架，建议坚持以下边界：

- 核心框架只认 `session_id` 和 `metadata`，不固定 `user_id/tenant_id`。
- 核心框架只提供 `HistoryStore/MemoryStore/ApprovalProvider` 协议，不强绑定 MySQL 或向量库。
- MySQL、Redis、向量库、Web 审批都作为 adapter/example/integration。
- 上下文滑动和压缩通过 `before_llm` hook 接入，尽量不污染 `AgentRunner` 主循环。
- 对话历史保存完整事实，短期上下文只保存“本次发给模型的窗口”。
- 长期记忆使用 `namespace + filters + metadata`，让业务方自己定义用户、项目、租户维度。
- HITL 先做同步 provider，后续再扩展事件流和异步恢复。
