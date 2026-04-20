# Wuwei 最小演进实现方案

这份文档只回答一个问题：

基于当前代码，不引入太复杂的设计，下面 4 件事应该怎么实现：

1. 上下文压缩链 + 长期记忆查询注入
2. Tool 运行时治理：timeout / retry / output limit
3. Planner 校验 + 并行执行 + 简单重规划
4. 结构化 trace/events + 基础回放

目标不是把 Wuwei 做成“大而全”框架，而是在不破坏现有边界的前提下，把最容易成为瓶颈的能力补齐。

## 当前框架现状

当前核心链路已经比较清楚：

- `wuwei.agent`
  - 对外门面：`Agent`、`PlanAgent`
- `wuwei.runtime`
  - 执行器：`AgentRunner`、`PlannerExecutorRunner`
  - 扩展点：`RuntimeHook`
- `wuwei.memory`
  - 目前只有 `Context`，负责保存消息历史
- `wuwei.tools`
  - `ToolRegistry`、`ToolExecutor`
- `wuwei.planning`
  - `Planner`、`Task`
- `wuwei.llm`
  - `LLMGateway`、adapter、统一消息类型

这说明内核已经有了，下一步不要再加新的“大总管对象”，而应该围绕现有边界补能力。

建议原则：

- 不新增复杂总线
- 不引入过多 manager/service/controller
- 优先复用 `RuntimeHook`
- 优先在现有模块里扩展，而不是新开一层抽象

---

## 1. 上下文压缩链 + 长期记忆查询注入

### 1.1 当前问题

当前 `AgentRunner` 在每一步都会把 `session.context` 的完整消息历史直接发给模型：

- `wuwei/runtime/agent_runner.py`
- `wuwei/memory/context.py`

这会导致两个问题：

1. 多轮会话越来越长，最终被上下文长度拖垮
2. 没有长期记忆注入点，用户跨轮、跨 session 的事实无法复用

### 1.2 最小实现原则

不要一开始做复杂 memory graph、向量数据库、自动摘要系统。

第一版只做两件事：

1. 在发给模型前，对消息做“轻量裁剪”
2. 在发给模型前，插入一条“相关记忆” system message

也就是说：

- `Context` 仍然只负责保存原始历史
- 压缩和记忆注入发生在 `before_llm`

### 1.3 最小实现方案

#### A. 新增一个轻量消息处理器

新增文件：

- `wuwei/memory/processors.py`

定义一个最小协议：

```python
class MessageProcessor(Protocol):
    def process(
        self,
        messages: list[Message],
        *,
        session: AgentSession,
        task: Task | None = None,
    ) -> list[Message]:
        ...
```

第一版只做 3 个内置 processor：

1. `KeepSystemMessageProcessor`
   - 永远保留最前面的 system message
2. `KeepRecentTurnsProcessor`
   - 只保留最近 N 轮 user/assistant/tool 相关消息
3. `KeepToolPairsProcessor`
   - 如果 assistant 有 tool_call，就尽量保留对应 tool message，避免上下文断裂

#### B. 用 Hook 接入，不改 `Context`

新增文件：

- `wuwei/runtime/context_hook.py`

思路：

- `ContextHook.before_llm(...)`
  - 接收当前完整 `messages`
  - 顺序执行 `processors`
  - 返回处理后的 `messages`

这样不需要改 `Context` 的职责。

#### C. 长期记忆先做协议 + 内存实现

新增文件：

- `wuwei/memory/store.py`

定义最小协议：

```python
class MemoryStore(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[str]:
        ...

    def add(self, text: str) -> None:
        ...
```

先给一个最简单的实现：

- `InMemoryMemoryStore`

第一版不做 embedding，不做向量检索，只做非常简单的文本包含匹配或关键词匹配。

#### D. 用 Hook 注入长期记忆

新增文件：

- `wuwei/runtime/memory_hook.py`

实现：

- `before_llm`
  - 找最后一条 user message
  - 用它去 `memory_store.search(...)`
  - 如果有结果，就在 system message 后面插入一条新的 system message，例如：

```text
[Relevant Memory]
1. 用户偏好中文输出
2. 该项目使用 Python 3.11
```

- `after_llm`
  - 第一版不做自动记忆抽取

### 1.4 第一版不做什么

- 不做自动摘要生成
- 不做向量数据库
- 不做跨用户 namespace
- 不做自动事实抽取

### 1.5 推荐落地顺序

1. `MessageProcessor` + `ContextHook`
2. `KeepRecentTurnsProcessor`
3. `MemoryStore` + `MemoryHook`
4. 后面再加 summary processor

---

## 2. Tool 运行时治理：timeout / retry / output limit

### 2.1 当前问题

当前 `ToolExecutor` 只有调用和错误包装：

- `wuwei/tools/executor.py`

缺少框架级治理能力：

- 超时
- 重试
- 输出截断
- 同步工具和异步工具的统一 timeout 处理

现在只有某些具体工具自己处理这些问题，例如 skill 脚本工具。

### 2.2 最小实现原则

不要先做取消令牌、进程池、复杂资源调度。

第一版只做：

1. 每个工具可配置 `timeout_s`
2. 每个工具可配置 `retries`
3. 每个工具可配置 `max_output_chars`

### 2.3 最小实现方案

#### A. 给 `Tool` 增加轻量配置字段

修改文件：

- `wuwei/tools/tool.py`

增加字段：

```python
timeout_s: float | None = None
retries: int = 0
max_output_chars: int | None = None
```

工具装饰器 `ToolRegistry.tool(...)` 也允许传这些参数。

#### B. 在 `ToolExecutor.execute_one()` 里统一治理

修改文件：

- `wuwei/tools/executor.py`

新增两个内部方法：

```python
async def _invoke_tool(self, tool: Tool, args: dict[str, Any]) -> Any:
    ...

def _truncate_output(self, content: str, limit: int | None) -> str:
    ...
```

处理策略：

- 如果 handler 是 async：
  - 用 `asyncio.wait_for(...)`
- 如果 handler 是 sync：
  - 用 `asyncio.to_thread(...)` 包一层，再 `wait_for(...)`

重试策略：

- 只在工具抛异常时重试
- 默认不重试
- 最大重试次数由 `tool.retries` 控制

输出限制：

- `serialize_output(...)` 之后统一截断
- 超过长度时追加：

```text
... [truncated N chars]
```

#### C. 保持 skill 工具只是“具体工具”

现在 `run_skill_python_script` 自己有 timeout 和 output limit。

后面可以保留它的目录校验逻辑，但把 timeout/output limit 迁移到框架层，让 skill 工具只关心：

- 是否先 `load_skill`
- 路径是否安全
- 如何调用脚本

### 2.4 第一版不做什么

- 不做用户中断取消
- 不做工具熔断
- 不做进程隔离
- 不做全局并发资源池

### 2.5 推荐落地顺序

1. `Tool.timeout_s`
2. `Tool.max_output_chars`
3. `Tool.retries`
4. 最后再把 skill 工具上的局部逻辑回收进框架

---

## 3. Planner 校验 + 并行执行 + 简单重规划

### 3.1 当前问题

当前 `Planner` 完全信任模型输出的 DAG：

- `wuwei/planning/planner.py`

当前 `PlannerExecutorRunner` 也只做了很有限的校验：

- 重复 id
- 下游节点是否存在

缺少：

- 单一入口校验
- DAG 无环校验
- 所有任务可达性校验
- ready tasks 真正并行执行
- 简单重规划

### 3.2 最小实现原则

不要一开始就做动态图 patch、复杂 checkpoint merge。

第一版只做：

1. 校验 planner 输出是否合法
2. ready tasks 支持并行执行
3. 执行停滞后允许一次简单重规划

### 3.3 最小实现方案

#### A. 新增一个纯函数 validator

新增文件：

- `wuwei/planning/validator.py`

提供一个函数：

```python
def validate_tasks(tasks: list[Task]) -> None:
    ...
```

第一版校验这些规则：

1. `id` 唯一
2. `next` 指向的节点必须存在
3. 必须且只能有一个起始任务
   - 起始任务的入度为 0
4. 图中不能有环
5. 所有节点都必须从起始任务可达

然后在：

- `wuwei/planning/planner.py`

里 `plan_task()` 结束后立刻调用 `validate_tasks(tasks)`。

#### B. 给 `PlannerExecutorRunner` 加一个并行开关

修改文件：

- `wuwei/runtime/planner_executor_runner.py`

增加初始化参数：

```python
parallel_tasks: bool = False
```

当同一轮出现多个 `ready_tasks` 时：

- `parallel_tasks=False`
  - 保持当前顺序执行
- `parallel_tasks=True`
  - 用 `asyncio.gather(...)` 并行执行本轮所有 ready tasks

这样不需要重新设计任务图模型。

#### C. 简单重规划不要“改旧图”，而是“补一轮新图”

这是最容易做错的地方。

最小方案：

- 增加参数：

```python
max_replans: int = 1
```

当执行结束后出现 `failed` 或 `blocked` 任务时：

1. 收集：
   - 原始 goal
   - 已完成任务结果
   - 失败任务错误
2. 发给 planner 一个新的 prompt：
   - “已完成这些任务”
   - “这些任务失败/阻塞”
   - “请只为剩余目标生成一份新的 DAG”
3. 用新任务列表替换当前未完成任务集合，再执行一轮

第一版不要求保留同一个 DAG 内部 id 体系，重新编号也可以。

### 3.4 第一版不做什么

- 不做 task 级 checkpoint 恢复
- 不做复杂 DAG merge
- 不做人工干预恢复
- 不做任务优先级调度器

### 3.5 推荐落地顺序

1. `validate_tasks()`
2. `parallel_tasks`
3. `max_replans=1`

---

## 4. 结构化 trace/events + 基础回放

### 4.1 当前问题

现在已经有 Hook：

- `wuwei/runtime/hooks.py`

这说明事件边界已经对了，但还没有结构化 tracing。

当前问题：

- 只有零散 print 能力
- 没有统一事件结构
- 没法回放一轮 agent 运行
- 不能按 step 看 LLM、tool、task 的时间线

### 4.2 最小实现原则

不要先做 OpenTelemetry、Web UI、复杂订阅总线。

第一版只做：

1. 定义一个统一事件结构
2. 做一个内存 recorder hook
3. 可选地写到 JSONL 文件

### 4.3 最小实现方案

#### A. 新增事件模型

新增文件：

- `wuwei/runtime/events.py`

定义：

```python
@dataclass
class RuntimeEvent:
    type: str
    session_id: str
    step: int | None = None
    task_id: int | None = None
    ts: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
```

第一版事件类型只要这些：

- `llm.before`
- `llm.after`
- `tool.before`
- `tool.after`
- `task.start`
- `task.end`

#### B. 做一个 RecordingHook

新增文件：

- `wuwei/runtime/recording_hook.py`

实现方式：

- 继承 `RuntimeHook`
- 在每个 hook 方法里 append 一个 `RuntimeEvent`
- 保存在 `self.events` 里

同时提供：

```python
def export_jsonl(self, path: str) -> None:
    ...
```

#### C. 事件里先放最关键字段

建议 payload：

- `llm.before`
  - `message_count`
  - `tool_count`
- `llm.after`
  - `finish_reason`
  - `latency_ms`
  - `usage`
- `tool.before`
  - `tool_name`
  - `arguments`
- `tool.after`
  - `tool_name`
  - `ok`
  - `error`
- `task.start/task.end`
  - `task_id`
  - `task_description`
  - `status`

#### D. 回放先做一个小工具函数

新增文件：

- `wuwei/runtime/replay.py`

提供一个简单函数：

```python
def print_replay(events: list[RuntimeEvent]) -> None:
    ...
```

只做文本时间线输出即可。

### 4.4 第一版不做什么

- 不做 Web UI
- 不做指标聚合系统
- 不做 OpenTelemetry
- 不做分布式 trace

### 4.5 推荐落地顺序

1. `RuntimeEvent`
2. `RecordingHook`
3. `export_jsonl`
4. `print_replay`

---

## 推荐开发顺序

如果按“投入最小、收益最大”的顺序来做，建议是：

1. Tool 运行时治理
   - 风险最低，收益立刻可见
2. Planner 校验
   - 能明显降低 plan-and-execute 不稳定性
3. 上下文压缩链
   - 会直接改善多轮会话可用性
4. 结构化 trace/events
   - 会显著提升调试效率
5. 长期记忆查询注入
   - 在压缩链稳定后再做更合适
6. 简单重规划
   - 放到 planner 校验之后

---

## 建议先做的最小版本

如果只做一个短周期版本，可以先落地下面这些：

### 第一阶段

- `Tool.timeout_s`
- `Tool.max_output_chars`
- `validate_tasks(tasks)`

### 第二阶段

- `KeepRecentTurnsProcessor`
- `RecordingHook`

### 第三阶段

- `InMemoryMemoryStore`
- `MemoryHook`
- `parallel_tasks=True`

这样做的好处是：

- 不会把架构搞复杂
- 每一步都能看到效果
- 所有改动都贴当前代码边界

---

## 一句话总结

Wuwei 现在最适合的路线不是“继续加功能模块”，而是把下面 4 条链补完整：

- 发给模型前，消息怎么裁剪
- 工具执行时，风险怎么收口
- planner 产出的 DAG，怎么校验和兜底
- agent 运行过程，怎么结构化记录

只要这 4 条链补上，当前框架就会从“最小可运行”进入“最小可持续扩展”阶段。
