# Wuwei 后续演进方案

更新时间：2026-04-09

这份文档不是要把 Wuwei 做成一个“功能最全”的生产框架，而是要给当前项目一条**适合学习、容易实现、边界清晰**的演进路线。重点关注五件事：

1. Memory
2. Hook
3. 可观测
4. 上下文压缩
5. Skill

当前项目已经有这些基础：

- `wuwei.agent`
  - `Agent`
  - `PlanAgent`
  - `AgentSession`
- `wuwei.runtime`
  - `AgentRunner`
  - `PlannerExecutorRunner`
- `wuwei.planning`
  - `Planner`
  - `Task`
- `wuwei.memory`
  - `Context`
- `wuwei.tools`
  - `ToolRegistry`
  - `ToolExecutor`
- `wuwei.llm`
  - `LLMGateway`
  - 统一的消息与响应类型

这说明 Wuwei 已经有了一个不错的“最小内核”。后面不要乱加抽象，应该遵循一句话：

**保留一个小核心，把高级能力做成可插拔模块。**

---

## 1. 市面上主流 Agent 框架在做什么

这一节只看对当前项目最有参考价值的框架，不追求全。

### 1.1 OpenAI Agents SDK

官方设计特点：

- 很少的核心原语：Agent、Tools、Handoffs、Guardrails、Sessions、Tracing
- 明确强调“少抽象、易学习、可定制”
- 内置 Session 记忆层
- 内置 Tracing
- 支持 handoff / agents as tools

对 Wuwei 最值得借鉴的点：

- **少原语**
  - 不要搞太多 manager、controller、service
  - 核心对象维持在几个就够：Agent / Runner / Session / Tool / Planner
- **Session 是工作记忆，不是万能记忆**
  - 多轮对话历史由 session 持有
  - 长期记忆不必直接塞进 session
- **Tracing 是运行时附加层，不是业务主逻辑**
  - 可观测应该围绕“运行事件”构建
- **Handoff / specialized agent 的思路适合未来 Skill**
  - 专长不一定要做成插件，也可以先做成“带独立提示词和工具集的小能力包”

不建议直接照搬的点：

- 现在不要上 guardrail/handoff 的全套复杂度
- 不要立刻对接 OpenAI 的完整 tracing 生态
- 先别做大量 hosted tools 或 server-managed state

适合借鉴程度：很高

---

### 1.2 LangGraph

官方设计特点：

- 强调图执行、状态、checkpoint、threads、store
- 把“会话内状态”和“跨会话记忆”区分得很清楚
- persistence / replay / time-travel / human-in-the-loop 是重要能力

对 Wuwei 最值得借鉴的点：

- **短期状态和长期记忆分离**
  - 线程内状态：对你这里就是 `AgentSession.context`
  - 跨线程记忆：应该是一个独立 `MemoryStore`
- **长期记忆不要直接混进上下文对象**
  - 用“查询后注入”的方式更清晰
- **Namespace 思想**
  - 比如 `(user_id, "profile")`
  - `(user_id, "preferences")`
  - `(project_id, "facts")`
- **检查点和图状态是 runtime 的责任**
  - 这和你已经拆出来的 `runtime/` 很契合

不建议直接照搬的点：

- 现在不要为了学习框架引入完整 graph builder DSL
- 先别做复杂 checkpoint backend
- 也不用一开始就做人机中断恢复

适合借鉴程度：高

---

### 1.3 PydanticAI

官方设计特点：

- Python-first
- 类型化 agent / tools / dependencies
- 提供 `history_processors`
- 强调消息历史在发给模型前可以被统一加工

对 Wuwei 最值得借鉴的点：

- **上文压缩不要散落在各处**
  - 应该做成统一的“消息历史处理链”
- **压缩逻辑是“模型调用前处理”，不是事后补丁**
  - 也就是在 `llm.generate(...)` 前做
- **处理器链**
  - 例如：
    - 保留系统消息
    - 保留最近 N 轮
    - 保留成对的 tool call / tool result
    - 必要时插入 summary

不建议直接照搬的点：

- 当前项目没必要重度类型驱动到每一步
- 不用为了“类型很严谨”而把接口做得很重

适合借鉴程度：很高

---

### 1.4 AutoGen

官方设计特点：

- 强调多 agent、消息通信、runtime、memory protocol、observability
- memory 是协议对象，可以负责更新模型上下文
- 支持 OpenTelemetry / structured logging

对 Wuwei 最值得借鉴的点：

- **Memory 应该是协议，不是写死实现**
  - 可以定义 `MemoryStore` / `MemoryProvider`
- **Structured logging / events**
  - 可观测最好是结构化事件，而不是零散 print
- **Runtime 才是可观测和 memory 注入的关键边界**
  - 这和你现在的 `runtime/` 目录很契合

不建议直接照搬的点：

- 当前项目不需要做 AutoGen 那么完整的消息总线
- 也不需要立刻接 OpenTelemetry

适合借鉴程度：中高

---

### 1.5 CrewAI

官方设计特点：

- 任务、角色、事件总线、memory、tracing 比较完整
- 事件监听器和 prompt tracing 做得很明显

对 Wuwei 最值得借鉴的点：

- **Hook / Event Bus 的形态很清晰**
  - 发事件
  - 注册监听器
  - 做 tracing / logging / metrics
- **Prompt tracing 不是核心逻辑，而是建立在事件系统上的能力**

不建议直接照搬的点：

- 当前项目不要做太重的 crew / role / flow 概念
- 不要让 Hook 系统反过来绑架核心运行逻辑

适合借鉴程度：中高

---

### 1.6 smolagents

官方设计特点：

- 强调简单、直接
- 有 step callbacks、memory replay、telemetry
- 非常适合学习多步 agent 基本原理

对 Wuwei 最值得借鉴的点：

- **保持简单**
- **提供 replay/debug 能力**
  - 即使先只是把运行步骤记录下来，也很有帮助
- **step 级回调比“神秘中间件”更容易理解**

不建议直接照搬的点：

- 当前项目不需要走 code agent 路线
- 不需要引入 Python 代码执行工具模式

适合借鉴程度：中

---

## 2. 调研结论：最适合当前项目的总体方案

对于 Wuwei 这种“学习型框架”，最合适的路线不是“大而全”，而是：

### 结论一：保留小核心

核心层只保留：

- `agent`
- `runtime`
- `planning`
- `memory`
- `tools`
- `llm`

其中：

- `agent` 负责对外门面
- `runtime` 负责真实执行
- `planning` 负责任务图
- `memory` 负责上下文与长期记忆
- `tools` 负责工具
- `llm` 负责模型调用

### 结论二：Hook 是下一步最重要的基础设施

因为：

- 可观测依赖 Hook
- Memory 注入和保存依赖 Hook
- 上下文压缩触发点依赖 Hook
- Skill 激活也依赖 Hook 或运行上下文

所以不要先做“可观测页面”，先做 **Hook/Event 系统**。

### 结论三：Memory 要分层

至少分两层：

- 短期记忆：当前 session 的上下文
- 长期记忆：独立 store，查询后注入

不要把长期记忆直接塞进 `Context`。

### 结论四：上下文压缩应该是“处理链”

不要在 `AgentRunner` 里到处写 `if messages too long`

应该统一做成：

- message processors
- 或 context compressor chain

### 结论五：Skill 不要做成复杂插件系统

第一版 Skill 应该是：

- 一组额外 instructions
- 一组默认工具白名单
- 一组 memory namespace
- 可选的激活条件

也就是说：

**Skill = 轻量能力包，不是插件市场。**

---

## 3. 最适合 Wuwei 的目标架构

建议最终收敛为：

```text
wuwei/
├─ agent/
│  ├─ base.py
│  ├─ agent.py
│  ├─ plan_agent.py
│  ├─ session.py
│  └─ __init__.py
├─ runtime/
│  ├─ agent_runner.py
│  ├─ planner_executor_runner.py
│  ├─ hooks.py
│  ├─ events.py
│  └─ __init__.py
├─ planning/
│  ├─ planner.py
│  ├─ task.py
│  └─ __init__.py
├─ memory/
│  ├─ context.py
│  ├─ store.py
│  ├─ manager.py
│  ├─ compression.py
│  ├─ summary.py
│  └─ __init__.py
├─ observability/
│  ├─ console.py
│  ├─ jsonl.py
│  ├─ trace.py
│  └─ __init__.py
├─ skills/
│  ├─ skill.py
│  ├─ registry.py
│  ├─ resolver.py
│  └─ __init__.py
├─ llm/
└─ tools/
```

其中最关键的约束是：

- `observability` 不直接驱动业务逻辑
- `observability` 建立在 `hooks/events` 之上
- `memory` 通过 runtime 注入
- `skills` 通过 runtime 注入
- `compression` 在模型调用前统一处理

---

## 4. Memory 怎么做

## 4.1 目标

Memory 不要一步做成“向量数据库 + 智能抽取 + 自动人格成长”。

第一版只做三件事：

1. 保存长期记忆
2. 检索相关记忆
3. 在运行前注入上下文

## 4.2 分层设计

### A. 短期记忆

现有：

- `AgentSession.context`

职责：

- 当前会话中的 system / user / assistant / tool 消息
- 只负责本 session 内的工作记忆

这一层先不要改太多。

### B. 长期记忆

新增：

- `memory/store.py`

建议接口：

```python
from typing import Protocol, Any


class MemoryStore(Protocol):
    async def add(self, namespace: tuple[str, ...], value: str, metadata: dict[str, Any] | None = None) -> str:
        ...

    async def search(self, namespace: tuple[str, ...], query: str, limit: int = 5) -> list[dict[str, Any]]:
        ...

    async def list(self, namespace: tuple[str, ...]) -> list[dict[str, Any]]:
        ...

    async def clear(self, namespace: tuple[str, ...]) -> None:
        ...
```

第一版实现：

- `InMemoryStore`

只用 Python dict/list 即可。

例如：

```python
{
    ("user:123", "profile"): [...],
    ("user:123", "preferences"): [...],
    ("project:abc", "facts"): [...],
}
```

### C. Memory 管理器

新增：

- `memory/manager.py`

职责：

- 统一封装“什么时候查 memory、什么时候写 memory”
- 不让 `AgentRunner` 直接操作底层 store

建议接口：

```python
class MemoryManager:
    def __init__(self, store: MemoryStore):
        self.store = store

    async def build_context_block(self, query: str, namespaces: list[tuple[str, ...]], limit: int = 5) -> str:
        ...

    async def save_fact(self, namespace: tuple[str, ...], value: str, metadata: dict | None = None) -> str:
        ...
```

## 4.3 如何接入当前运行链路

接入点放在 `runtime`，不是 `agent`。

普通 `AgentRunner` 的一轮调用前：

1. 读取当前 session context
2. 读取相关 long-term memories
3. 把 memory block 注入到 system prompt 或额外 assistant/system message
4. 再交给模型

Plan 模式下：

- 每个 task 可以继承外层 session 的 memory namespace
- 但 task 自己的中间结果仍然通过 `Task.result` 传递，不要混入长期记忆

## 4.4 第一版不要做什么

- 不要先做向量数据库
- 不要先做自动 LLM 提炼记忆
- 不要先做“所有对话都自动写 memory”

更好的第一版是：

- 用户手动保存
- 或框架只提供一个简单策略：
  - `save_last_answer_as_memory()`
  - `save_fact_if_marked()`

---

## 5. Hook 怎么做

## 5.1 为什么 Hook 要先做

因为它是这几个能力的共同基础：

- 可观测
- memory 查询和写入
- 上下文压缩
- skill 激活

## 5.2 设计原则

Hook 不要设计成“神秘中间件”。

应该做成：

- 明确的事件对象
- 明确的触发点
- 明确的监听器接口

## 5.3 推荐实现

新增：

- `runtime/events.py`
- `runtime/hooks.py`

### A. Event 对象

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunEvent:
    type: str
    run_id: str
    session_id: str | None = None
    task_id: int | None = None
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
```

### B. HookHandler 接口

```python
from typing import Protocol


class HookHandler(Protocol):
    async def handle(self, event: RunEvent) -> None:
        ...
```

### C. HookManager

```python
class HookManager:
    def __init__(self, handlers: list[HookHandler] | None = None):
        self.handlers = handlers or []

    def register(self, handler: HookHandler) -> None:
        self.handlers.append(handler)

    async def emit(self, event: RunEvent) -> None:
        for handler in self.handlers:
            await handler.handle(event)
```

## 5.4 第一版事件列表

建议先控制在这些：

- `run.started`
- `run.completed`
- `run.failed`
- `llm.request.started`
- `llm.response.received`
- `tool.call.started`
- `tool.call.completed`
- `tool.call.failed`
- `planner.started`
- `planner.completed`
- `task.started`
- `task.completed`
- `task.failed`
- `memory.query.started`
- `memory.query.completed`
- `compression.started`
- `compression.completed`
- `skill.activated`

这已经足够支撑后面的所有能力。

---

## 6. 可观测怎么做

## 6.1 原则

可观测不要独立发明一套运行机制。

正确做法是：

- runtime 发事件
- observability 订阅事件

也就是说：

**可观测 = Hook 的消费者**

## 6.2 第一版不要上什么

- 不要一开始接 OpenTelemetry
- 不要一开始做 Web Dashboard
- 不要一开始做 metrics backend

## 6.3 第一版做什么

新增：

- `observability/console.py`
- `observability/jsonl.py`
- `observability/trace.py`

### A. ConsoleObserver

职责：

- 在终端里打印结构化运行信息

示例输出：

```text
[run.started] run_001
[planner.completed] tasks=3
[task.started] task=1
[tool.call.started] get_weather {"city":"北京"}
[tool.call.completed] get_weather {"city":"北京","temperature_c":25}
[task.completed] task=1
```

### B. JsonlObserver

职责：

- 把事件写入 `.jsonl` 文件
- 便于后续回放、测试、分析

### C. TraceCollector

职责：

- 在内存里把事件聚合成一个简单 trace

```python
class RunTrace:
    run_id: str
    events: list[RunEvent]
```

这个版本已经足够学习“可观测”。

## 6.4 为什么不先做 UI

因为当前项目的瓶颈不是“展示层不够炫”，而是：

- 运行事件还没有统一
- memory / skill / compression 还没接进去

等事件稳定后，再做 Web UI 才有意义。

---

## 7. 上下文压缩怎么做

## 7.1 核心思路

压缩不是 memory，压缩也不是 session。

压缩是：

**在模型调用前，对消息历史做统一加工。**

这个思路最接近 PydanticAI 的 `history_processors`。

## 7.2 推荐位置

新增：

- `memory/compression.py`

也可以叫：

- `memory/processors.py`

第一版建议叫 `compression.py`，更直观。

## 7.3 推荐接口

```python
from typing import Protocol

from wuwei.llm import Message


class ContextProcessor(Protocol):
    async def process(self, messages: list[Message]) -> list[Message]:
        ...
```

然后做一个链：

```python
class ContextProcessorChain:
    def __init__(self, processors: list[ContextProcessor]):
        self.processors = processors

    async def process(self, messages: list[Message]) -> list[Message]:
        current = messages
        for processor in self.processors:
            current = await processor.process(current)
        return current
```

## 7.4 第一版处理器

建议只做三个：

### A. KeepSystemProcessor

作用：

- 永远保留 system prompt

### B. KeepRecentTurnsProcessor

作用：

- 只保留最近 N 轮 user/assistant/tool 交互

注意：

- 工具调用和工具返回必须成对保留
- 不要只保留 tool_call，不保留 tool result

### C. SummaryProcessor

作用：

- 当历史过长时，把旧历史压缩成一段 summary message

第一版 summary 可以先不调用 LLM，先做简单摘要策略：

- 只提取最近若干轮的用户目标与助手结论
- 或者直接把旧记录变成一段“会话摘要”

如果后面再进化：

- 才考虑用一个 `summarizer_llm`

## 7.5 触发策略

第一版不要先做 token 精确统计。

可以先做：

- 按消息条数触发
- 按字符数粗略触发

例如：

- 超过 30 条消息就压缩
- 或总字符数超过 12000 就压缩

学习框架里这样完全够用。

---

## 8. Skill 怎么做

## 8.1 先明确 Skill 不是什么

在当前项目里，Skill 不应该一开始做成：

- 动态插件安装系统
- 代码热加载 marketplace
- 一整套复杂工作流 DSL

那样会把学习主线带偏。

## 8.2 最适合当前项目的 Skill 定义

Skill 应该是一个轻量能力包：

- 一组 instructions
- 一组默认工具白名单
- 一组 memory namespace
- 可选的输入输出提示
- 可选的激活条件

也就是：

```python
from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    description: str
    instructions: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    memory_namespaces: list[tuple[str, ...]] = field(default_factory=list)
    input_hint: str | None = None
    output_hint: str | None = None
```

## 8.3 Skill 的运行方式

新增：

- `skills/skill.py`
- `skills/registry.py`
- `skills/resolver.py`

### A. SkillRegistry

职责：

- 注册 skill
- 通过名称查找 skill

### B. SkillResolver

职责：

- 决定这次 run 应该激活哪些 skill

第一版支持两种方式：

1. 显式指定

```python
await agent.run("...", skills=["weather_analyst"])
```

2. 简单自动匹配

- 关键词命中
- 或调用方传入一个 resolver

## 8.4 Skill 如何注入

最简单的方式：

### 注入 instructions

把 skill instructions 拼接到 system prompt 后面：

```text
系统设定...

[Skill: weather_analyst]
你应该优先调用天气相关工具，并输出城市、天气、温度。
```

### 限制工具集

激活 skill 后，如果 skill 指定了 `allowed_tools`，那就只把这些工具传给模型。

### 查询 skill 对应的 memory namespace

例如：

- `weather_analyst` 读取 `("user:123", "travel_preferences")`

## 8.5 Plan 模式下的 Skill

第一版先不要让 planner 自动分配 skill。

更简单的做法：

- `PlanAgent.run(..., skills=[...])`
- 整个计划共用这些 skill

第二版再考虑：

- 按 task.description 选择 skill

例如：

- 查询资料 task -> `research_skill`
- 总结报告 task -> `writer_skill`

---

## 9. 推荐的实现顺序

不要五件事一起做。

最适合当前项目的顺序是：

### 第一步：Hook / Event

原因：

- 这是 memory、可观测、skill 的共同基础

交付物：

- `runtime/events.py`
- `runtime/hooks.py`
- 在 `AgentRunner` 和 `PlannerExecutorRunner` 埋点

### 第二步：上下文压缩

原因：

- 对单 agent 和 plan agent 都直接有用
- 改动边界清晰

交付物：

- `memory/compression.py`
- `ContextProcessorChain`
- `KeepRecentTurnsProcessor`

### 第三步：MemoryStore + MemoryManager

原因：

- 这时 runtime 已经有 hook 了，memory 查询/写入点更自然

交付物：

- `memory/store.py`
- `memory/manager.py`
- `InMemoryStore`

### 第四步：可观测

原因：

- 事件系统已经有了，直接做 observer 就行

交付物：

- `observability/console.py`
- `observability/jsonl.py`

### 第五步：Skill

原因：

- Skill 本质上是“基于 runtime/memory/tools 的组合层”
- 放最后做，理解会最顺

交付物：

- `skills/skill.py`
- `skills/registry.py`
- `skills/resolver.py`

---

## 10. 当前项目里不建议现在做的事

这些能力不是没价值，而是**现在做会把项目复杂度一下拉高**：

- 向量数据库接入
- 自动长期记忆提炼
- 复杂多 agent 通信总线
- Web 可观测大屏
- 插件市场
- 人机中断恢复
- 完整 checkpoint/replay 系统
- OpenTelemetry 全链路接入

对于学习型项目，最重要的是：

**每加一个模块，都能看清楚它插在运行链路的哪个位置。**

---

## 11. 最终推荐方案

一句话总结：

### 11.1 总体方向

借鉴：

- OpenAI Agents SDK 的“小原语 + sessions + tracing 思路”
- LangGraph 的“短期状态 / 长期记忆分离”
- PydanticAI 的“history processors”
- AutoGen / CrewAI 的“结构化事件与可观测”

但不要照搬它们的完整复杂度。

### 11.2 对 Wuwei 的最适合方案

最适合当前项目的是：

- 一个小核心
- 一个事件系统
- 一个轻量 memory 分层
- 一个上下文处理链
- 一个轻量 skill 层

你可以把它理解成：

```text
Agent / PlanAgent
    ↓
Runtime
    ↓
Hooks / Events
    ├─ Observability
    ├─ Memory
    ├─ Compression
    └─ Skills
```

这条路线足够清晰，也足够适合学习。

---

## 12. 参考资料

以下结论主要参考这些官方资料：

- OpenAI Agents SDK 总览：
  - https://openai.github.io/openai-agents-python/
- OpenAI Agents SDK Sessions：
  - https://openai.github.io/openai-agents-python/sessions/
- OpenAI Agents SDK Tracing：
  - https://openai.github.io/openai-agents-python/tracing/
- LangGraph Persistence / Memory Store：
  - https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Memory Overview：
  - https://docs.langchain.com/oss/python/langgraph/memory
- PydanticAI Message History / History Processors：
  - https://ai.pydantic.dev/message-history/
- AutoGen Memory Protocol：
  - https://microsoft.github.io/autogen/stable/reference/python/autogen_core.memory.html
- AutoGen Memory and RAG：
  - https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/memory.html
- AutoGen Tracing and Observability：
  - https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tracing.html
- CrewAI Event Listeners：
  - https://docs.crewai.com/concepts/event-listener
- CrewAI Memory：
  - https://docs.crewai.com/concepts/memory
- smolagents Guided Tour：
  - https://huggingface.co/docs/smolagents/guided_tour
- smolagents Memory：
  - https://huggingface.co/docs/smolagents/main/en/tutorials/memory

