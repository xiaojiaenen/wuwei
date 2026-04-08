# Wuwei ReAct Runtime 重构设计

状态: draft  
适用版本: `0.1.x -> 0.2.x`  
目标: 将当前 `Agent` 从 demo 级执行器，重构为正式的 ReAct runtime，并在此基础上支持 `strategy="react"` 和 `strategy="plan_execute"` 两种运行模式。

## 1. 设计目标

本次改造只解决四件事：

1. 把现有 `Agent` 正式定义为 ReAct runtime 的外部入口。
2. 把所有工具调用统一收口到 `ToolExecutor`。
3. 把 `planner.py` 从独立实验类改造成可插拔的 planning strategy。
4. 提供统一运行接口，支持 `strategy="react"` 和 `strategy="plan_execute"`。

本次改造不解决这些问题：

- 不做多智能体协作框架。
- 不做长期记忆系统。
- 不做分布式任务调度。
- 不做复杂 GUI 或可视化工作流编排器。

## 2. 当前实现的主要问题

当前仓库已经具备最小闭环，但模块边界还不够稳定。

### 2.1 `Agent` 同时承担了三种职责

当前 `wuwei/agent/agent.py` 同时承担：

- Agent 定义
- 单次运行的状态管理
- ReAct 执行循环

这会带来两个直接问题：

- `Agent` 不能被复用到不同运行策略中。
- 一旦要加入 `plan_execute`、workflow、checkpoint，就会继续把逻辑堆在同一个类里。

### 2.2 工具执行路径不统一

当前主链路在 `Agent` 内部直接执行 tool call，而没有统一走 `wuwei/tools/executor.py`。  
结果是：

- 工具错误格式不统一。
- 工具输出序列化逻辑重复。
- 后续如果要加入重试、超时、审计、middleware，会改很多地方。

### 2.3 `planner.py` 是孤立能力，不是 runtime 组件

当前 `wuwei/core/planner.py` 只是一个独立类，无法自然接入主运行时。  
它没有和以下对象建立稳定关系：

- session
- tool execution
- run result
- event / tracing
- strategy

### 2.4 没有显式的运行策略层

当前只有一种隐式策略，就是“模型推理，若返回 tool calls，则执行工具并继续循环”。  
这其实就是 ReAct，但代码里没有把它抽象成策略，因此：

- 不能显式切换运行模式
- 不能复用 ReAct 的执行能力作为 `plan_execute` 的子步骤执行器
- 后续很难扩展其它策略

### 2.5 运行状态与上下文对象过于粗糙

当前 `Context` 只是消息列表。  
一旦要支持计划、step 状态、usage、artifacts、运行指标，就需要引入更明确的 session 对象。

### 2.6 具体问题定位

下面不再只说抽象问题，而是直接对应到当前代码中的具体位置和后果。

#### 问题 1: `Agent` 既是定义对象，又是运行对象

位置：

- `wuwei/agent/agent.py:10`
- `wuwei/agent/agent.py:28`
- `wuwei/agent/agent.py:42`
- `wuwei/agent/agent.py:79`

现状：

- `__init__()` 里保存了 `llm / config / tools`
- 同时又创建了 `self.context`
- 同一个类里又实现了 `_run_non_stream()` 和 `_run_stream()`

具体后果：

- 一个 `Agent` 实例天然绑定一份 `context`
- 多次调用 `agent.run()` 时，历史消息会直接累积在同一个 `self.context` 中
- 这意味着 `Agent` 不是“静态定义”，而是带状态的运行体

为什么这是问题：

- 你无法自然地区分“这个 agent 是谁”和“这次运行发生了什么”
- 后面要支持 `session`、恢复执行、多策略时，这种耦合会很难拆

#### 问题 2: `Agent.run()` 多次调用会污染上下文

位置：

- `wuwei/agent/agent.py:28`
- `wuwei/agent/agent.py:30`
- `wuwei/agent/agent.py:44`
- `wuwei/agent/agent.py:81`

现状：

- `context` 在 `Agent.__init__()` 时创建一次
- system prompt 在初始化时就写进去
- 每次 `run()` 只是继续往这个 `context` 里 append 新消息

具体后果：

- 第一次调用和第二次调用不是相互隔离的
- 旧的 user/tool/assistant 消息会进入下一次运行
- 这会导致模型上下文污染

一个简单例子：

```python
agent = Agent(...)
await agent.run("帮我查北京天气")
await agent.run("2+2 等于几")
```

第二次运行时，模型看到的上下文不只是“2+2 等于几”，而是连第一次天气查询的历史也会一起看到。

#### 问题 3: `ToolExecutor` 已经存在，但主链路没有统一使用

位置：

- `wuwei/tools/executor.py:10`
- `wuwei/agent/agent.py:56`
- `wuwei/agent/agent.py:108`

现状：

- 你已经写了 `ToolExecutor`
- 但非流式和流式分支里都在 `Agent` 内部手动执行工具

具体后果：

- 工具执行逻辑重复两份
- 流式和非流式的工具错误输出格式不一致
- `ToolExecutor` 中的结构化错误逻辑没有真正成为主标准

为什么这是问题：

- 后续你想给工具执行加 timeout、retry、审计、tracing 时，要改两处甚至更多处
- 主链路和 executor 分裂后，代码会越来越难维护

#### 问题 4: 同步工具当前会直接报错

位置：

- `wuwei/tools/tool.py:34`
- `wuwei/tools/tool.py:37`

现状：

- `invoke()` 先判断是不是协程函数
- 如果不是，就执行 `result = await self.handler(**args)`

这在同步函数场景下是错误的，因为同步函数返回普通值，不能 `await`

我已经本地验证过，下面这种 tool 会直接失败：

```python
@registry.tool(description="sync tool")
def echo(text: str):
    return text
```

报错是：

`TypeError: 'str' object can't be awaited`

为什么这是问题：

- 你的 README 里虽然建议工具用 `async def`
- 但框架层不应该用“文档约束”掩盖运行时错误
- 一个工具系统如果连 sync handler 都不能安全处理，扩展性会很差

#### 问题 5: `ToolRegistry.get()` 的类型声明和实际行为不一致

位置：

- `wuwei/tools/registry.py:24`
- `wuwei/tools/registry.py:25`

现状：

- 签名声明返回 `Tool | None`
- 但实现是 `return self._tools[name]`

具体后果：

- 不存在的工具不是返回 `None`
- 而是直接抛 `KeyError`

这会影响两层：

1. `ToolExecutor.execute_one()` 里本来写了 `if tool is None` 分支，但由于 `get()` 先抛错，这个分支其实失效。
2. `Agent` 内部同样假设 `get()` 可以返回空值，但真实行为不是这样。

为什么这是问题：

- 这属于接口契约不一致
- 运行时行为和代码阅读预期完全不同

#### 问题 6: 流式和非流式返回类型不统一

位置：

- `wuwei/agent/agent.py:35`
- `wuwei/agent/agent.py:38`
- `wuwei/agent/agent.py:40`
- `wuwei/agent/agent.py:121`
- `wuwei/agent/agent.py:125`

现状：

- 非流式 `run()` 返回最终字符串
- 流式 `run()` 返回 async generator
- 但流式错误分支里还会 `yield f"...字符串..."`，而正常分支 `yield chunk`

具体后果：

- 流式消费者拿到的数据类型不稳定
- 有时是 `LLMResponseChunk`
- 有时是原始字符串

为什么这是问题：

- 上层调用方很难写稳定的消费逻辑
- 一旦你后续要做统一 Runner/Result，这种接口会非常难兼容

#### 问题 7: `Context` 只能表达消息，无法表达运行状态

位置：

- `wuwei/core/context.py:4`
- `wuwei/core/context.py:20`

现状：

- `Context` 里只有 `_messages`
- 没有 `session_id`
- 没有 `usage`
- 没有 `artifacts`
- 没有 `plan`

具体后果：

- 规划结果没地方挂
- 工具中间产物只能硬塞回 message
- 想做 resume/checkpoint 基本没有落点

为什么这是问题：

- message history 和 runtime state 不是一回事
- 当前结构只能做最简 demo，对策略层扩展支撑不够

#### 问题 8: `Planner` 是孤立实验类，没有进入主运行时

位置：

- `wuwei/core/planner.py:8`
- `wuwei/core/planner.py:38`
- `wuwei/core/planner.py:51`

现状：

- `Planner` 自己持有 `goal`
- 自己拼 prompt
- 自己单独跑 `llm.generate()`
- 文件尾部还有独立 `__main__` 示例

具体后果：

- planner 不能自然接入 `Agent`
- planner 不能共享 `context/session`
- planner 不能作为运行策略的一部分复用

为什么这是问题：

- 这说明 planner 目前是“实验脚本级能力”，不是“框架级能力”
- 你现在说想支持 `plan_execute`，但 planner 还没有被设计成可插拔 runtime 组件

#### 问题 9: `Planner` 文件里存在硬编码 API Key 示例

位置：

- `wuwei/core/planner.py:55`

现状：

- `__main__` 示例里写了明文 API key

为什么这是问题：

- 这是明显的安全问题
- 即使只是测试 key，也不该出现在仓库源码中
- 后续开源或共享代码时会留下风险

#### 问题 10: `Task`/`TaskList` 的语义位置不对

位置：

- `wuwei/core/task.py:6`
- `wuwei/core/task.py:12`

现状：

- `Task` 实际上是在服务 planner 输出
- 但它被放在 `core/`

为什么这是问题：

- `core` 看起来像全局基础层
- 但 `Task` 其实是 planning domain model
- 后面继续扩展时，类型边界会越来越乱

#### 问题 11: `LLMGateway` 的边界是对的，但运行态信息没有真正落地

位置：

- `wuwei/llm/gateway.py:45`
- `wuwei/llm/gateway.py:53`
- `wuwei/llm/gateway.py:114`

现状：

- gateway 已经能拿到 latency 和 usage
- 但这些数据没有被 session 持久化

具体后果：

- 运行结束后，框架层没有统一的 usage 汇总位置
- 无法很自然地做 tracing、成本统计和性能分析

为什么这是问题：

- 如果未来要支持生产级 agent runtime，usage 和 timing 不应该只是“临时从 response 里读一下”
- 它们应该进入 session/result

#### 问题 12: 当前代码没有测试基线

现状：

- `pyproject.toml` 里已经声明了 `pytest`
- 但仓库当前没有 `tests/` 目录

具体后果：

- 当前这些行为问题很难被稳定拦住
- 一旦开始重构 `Agent -> Runner -> Strategy`，很容易改出回归问题

为什么这是问题：

- 这不是“以后再补”的小事
- 你当前已经到了要做架构迁移的阶段，没有测试会显著拖慢重构速度

### 2.7 这些问题里，哪些是立刻要修的，哪些是架构级问题

立刻要修的实现问题：

- `Tool.invoke()` sync handler 报错
- `ToolRegistry.get()` 契约不一致
- 流式分支返回类型不一致
- `core/planner.py` 中硬编码 API key

应该作为架构改造主线推进的问题：

- `Agent` 同时承担定义和运行职责
- `Context` 不是 session
- `ToolExecutor` 没有成为统一入口
- `Planner` 没有接入主运行时
- 没有 strategy 层

### 2.8 用一句话总结当前问题

当前 Wuwei 的问题不是“没有 agent 能力”，而是：

`能力已经有了，但运行时抽象还没有成型，导致已有能力分散、重复、带状态耦合，并且存在若干会直接影响稳定性的实现错误。`

## 3. 推荐的总体方案

推荐把本次改造拆成三层：

1. `Agent` 负责“定义”。
2. `Runner` 负责“运行”。
3. `Strategy` 负责“如何运行”。

这三层的关系如下：

```text
User Input
  ->
Agent
  ->
Runner
  ->
ExecutionStrategy
  ->
LLMGateway / ToolExecutor / Planner
  ->
RunSession
```

核心原则只有一句话：

`Agent` 不再自己实现完整运行循环，真正的执行逻辑下沉到 `Runner + Strategy`。

## 4. 目标目录结构

推荐重构后的目录如下：

```text
wuwei/
  agent/
    __init__.py
    agent.py
    base.py
    runner.py
    session.py
    result.py
  strategies/
    __init__.py
    base.py
    react.py
    plan_execute.py
  planning/
    __init__.py
    base.py
    llm_planner.py
    types.py
  tools/
    __init__.py
    executor.py
    registry.py
    tool.py
  llm/
    ...
```

### 为什么建议新增 `strategies/` 和 `planning/`

原因很直接：

- `Strategy` 是运行期概念，不应该再塞进 `core/`
- `Planner` 是计划生成能力，不应该和消息上下文混在一起
- 后续如果要做 workflow node，可以在 `planning/` 或新增 `workflow/` 中继续扩展，不会污染 agent 主链路

## 5. 新的模块职责

## 5.1 `Agent`

文件建议: `wuwei/agent/agent.py`

`Agent` 改造后的职责：

- 保存 `llm`
- 保存 `tools`
- 保存 `config`
- 指定默认 `strategy`
- 对外暴露简化版 `run()` 入口

`Agent` 不再直接负责：

- 手动循环调用模型
- 手动执行工具
- 直接维护完整运行状态

推荐接口：

```python
class Agent(BaseAgent):
    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        config: AgentConfig | None = None,
        default_strategy: str = "react",
    ):
        ...

    async def run(
        self,
        user_input: str,
        strategy: str | None = None,
        stream: bool = False,
        session: RunSession | None = None,
    ):
        runner = Runner()
        return await runner.run(
            agent=self,
            user_input=user_input,
            strategy=strategy or self.default_strategy,
            stream=stream,
            session=session,
        )
```

说明：

- `Agent.run()` 作为兼容层保留，避免用户代码一次性全断。
- 真正的执行逻辑移交给 `Runner`。

## 5.2 `RunSession`

文件建议: `wuwei/agent/session.py`

新增 `RunSession`，用于承载一次运行的状态。

推荐字段：

```python
class RunSession(BaseModel):
    session_id: str
    messages: list[Message] = Field(default_factory=list)
    plan: Plan | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
```

`RunSession` 的职责：

- 保存消息历史
- 保存计划对象
- 保存阶段性产物
- 保存 usage / metrics
- 保存本次运行的元信息

它会替代当前 `Context` 承担大部分运行状态职责。

`Context` 有两种处理方式：

1. 最小改造方案：保留 `Context`，作为 `RunSession.messages` 的轻量包装器。
2. 推荐方案：逐步废弃 `Context`，让 `RunSession` 直接提供 `add_user_message()` 等接口。

推荐采用方案 2。

## 5.3 `Runner`

文件建议: `wuwei/agent/runner.py`

`Runner` 是本次重构的核心对象。

它负责：

- 初始化 session
- 选择 strategy
- 调用 LLM
- 调用 `ToolExecutor`
- 写回 session
- 统一产出 run result

推荐接口：

```python
class Runner:
    def __init__(
        self,
        tool_executor_cls: type[ToolExecutor] = ToolExecutor,
        strategy_registry: dict[str, type[ExecutionStrategy]] | None = None,
    ):
        ...

    async def run(
        self,
        agent: Agent,
        user_input: str,
        strategy: str = "react",
        stream: bool = False,
        session: RunSession | None = None,
    ) -> RunResult | AsyncIterator[LLMResponseChunk]:
        ...
```

`Runner` 内部应该提供这些私有方法：

- `_prepare_session(agent, session)`
- `_append_system_prompt_if_needed(agent, session)`
- `_call_llm(agent, session, stream=False, **kwargs)`
- `_execute_tool_calls(agent, tool_calls)`
- `_merge_usage(session, usage)`
- `_emit_event(...)`

注意：

- `Runner` 不应该知道某个策略的具体 prompt。
- `Runner` 应该只提供能力，不实现业务决策。

## 5.4 `ToolExecutor`

文件保留: `wuwei/tools/executor.py`

`ToolExecutor` 应成为唯一工具执行入口。

必须完成这几个改造：

1. 所有 tool call 都通过 `ToolExecutor.execute()` 或 `execute_one()` 执行。
2. 工具输出统一序列化。
3. 工具错误统一结构化。
4. 后续超时、重试、审计、中间件也都从这里扩展。

建议在 `Runner` 中这样使用：

```python
tool_messages = await tool_executor.execute(tool_calls)
for message in tool_messages:
    session.messages.append(message)
```

这样 `ReActStrategy` 和 `PlanAndExecuteStrategy` 都不再自己碰具体工具 handler。

## 5.5 `ExecutionStrategy`

文件建议: `wuwei/strategies/base.py`

新增运行策略抽象：

```python
class ExecutionStrategy(ABC):
    name: str

    @abstractmethod
    async def run(
        self,
        runner: Runner,
        agent: Agent,
        session: RunSession,
        user_input: str,
        stream: bool = False,
    ):
        ...
```

注意这个接口设计的重点：

- `runner` 作为能力提供者注入给 strategy
- `strategy` 不直接依赖底层实现细节
- 同一个 `runner` 可以驱动多个 `strategy`

## 5.6 `ReActStrategy`

文件建议: `wuwei/strategies/react.py`

`ReActStrategy` 用于承接你当前 `Agent` 的主循环。

核心流程：

1. 将用户输入写入 session
2. 调用模型
3. 若返回普通文本，则结束
4. 若返回 `tool_calls`，则统一走 `ToolExecutor`
5. 把工具结果写回 session
6. 继续下一轮，直到输出最终答案或达到 `max_steps`

推荐伪代码：

```python
class ReActStrategy(ExecutionStrategy):
    name = "react"

    async def run(self, runner, agent, session, user_input, stream=False):
        session.add_user_message(user_input)

        while session.step_count < agent.config.max_steps:
            response = await runner._call_llm(agent, session, stream=False)
            session.add_assistant_message(
                content=response.message.content,
                tool_calls=response.message.tool_calls,
            )

            if response.finish_reason == "tool_calls" and response.message.tool_calls:
                tool_messages = await runner._execute_tool_calls(
                    agent=agent,
                    tool_calls=response.message.tool_calls,
                )
                for message in tool_messages:
                    session.messages.append(message)
                session.step_count += 1
                continue

            return RunResult.final(response.message.content, session=session)

        return RunResult.limit_reached(session=session)
```

### 为什么要显式命名为 `ReActStrategy`

因为当前逻辑本质上已经是 ReAct，只是代码中没有把这个事实显式表达出来。  
一旦命名清楚：

- 外部用户知道这是什么运行模式
- `plan_execute` 可以把它作为子执行器复用
- 测试也可以按策略分组

## 5.7 `Planner` 抽象

文件建议: `wuwei/planning/base.py`

当前 `wuwei/core/planner.py` 的问题不是“能不能生成计划”，而是“不能插到 runtime 里”。  
所以要先抽象接口，再迁移实现。

建议定义：

```python
class Planner(ABC):
    @abstractmethod
    async def create_plan(
        self,
        goal: str,
        runner: Runner,
        agent: Agent,
        session: RunSession,
    ) -> Plan:
        ...
```

然后把当前 LLM 版 planner 迁移成：

文件建议: `wuwei/planning/llm_planner.py`

```python
class LLMPlanner(Planner):
    def __init__(self, llm: LLMGateway | None = None):
        self.llm = llm

    async def create_plan(self, goal, runner, agent, session) -> Plan:
        ...
```

这样做的好处：

- planner 可以复用 agent 默认模型
- planner 也可以单独指定模型
- planner 可以拿到 session，从而利用上下文和历史
- planner 未来可以替换成 rule-based planner

## 5.8 `Plan` 与 `PlanStep`

文件建议: `wuwei/planning/types.py`

当前 `Task` 更像 plan step。  
建议改名为：

- `Plan`
- `PlanStep`

推荐定义：

```python
class PlanStep(BaseModel):
    id: int
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    next_steps: list[int] = Field(default_factory=list)
    result: str | None = None


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
```

如果你暂时不想破坏兼容，也可以：

- 保留 `Task` / `TaskList`
- 在 `planning/types.py` 中先做别名过渡

但长期建议还是明确改名。

## 5.9 `PlanAndExecuteStrategy`

文件建议: `wuwei/strategies/plan_execute.py`

这是第二种运行模式。

其职责不是替代 ReAct，而是编排：

1. 先调用 planner 生成计划
2. 把计划存进 session
3. 逐步执行 plan step
4. 每个 step 的执行默认复用 `ReActStrategy` 的能力
5. 汇总所有 step 结果
6. 生成最终回答

最关键的设计点：

`PlanAndExecuteStrategy` 不应该自己重新实现一套工具循环，它应该复用 ReAct 的执行能力。

推荐伪代码：

```python
class PlanAndExecuteStrategy(ExecutionStrategy):
    name = "plan_execute"

    def __init__(self, planner: Planner | None = None):
        self.planner = planner or LLMPlanner()
        self.react = ReActStrategy()

    async def run(self, runner, agent, session, user_input, stream=False):
        session.add_user_message(user_input)

        plan = await self.planner.create_plan(
            goal=user_input,
            runner=runner,
            agent=agent,
            session=session,
        )
        session.plan = plan

        for step in plan.steps:
            if step.status == "completed":
                continue

            step.status = "in_progress"
            step_result = await self._execute_step(
                runner=runner,
                agent=agent,
                session=session,
                step=step,
            )
            step.result = step_result
            step.status = "completed"

        return await self._finalize(runner, agent, session, user_input)
```

### Step 执行建议

每个 step 的执行建议使用“受控 ReAct”：

- 输入不是整个用户目标，而是当前 step 描述
- prompt 中明确要求只完成当前 step
- step 结果写回 `session.plan.steps[i].result`

这样做可以避免 plan_execute 失控膨胀成第二套 runtime。

## 6. `planner.py` 应该改成 strategy 还是 workflow node

推荐答案：

先改成 planning strategy，再为 workflow node 留适配层。

理由：

- 你当前最急的是双运行模式，而不是完整 workflow engine
- strategy 接入成本低，能最快落地
- workflow node 可以在后续版本中把 planner 再包装一层，而不是反过来

推荐路线：

1. 先把 `core/planner.py` 迁到 `planning/llm_planner.py`
2. 再让 `PlanAndExecuteStrategy` 依赖它
3. 等后续需要 workflow 时，再加 `PlannerNode`

如果后续要做 workflow，推荐最小接口：

```python
class WorkflowNode(ABC):
    name: str

    @abstractmethod
    async def run(self, state: dict[str, Any], runner: Runner) -> dict[str, Any]:
        ...
```

其中 `PlannerNode` 只是：

- 读取 `state["goal"]`
- 生成 `state["plan"]`
- 返回更新后的 `state`

这样就不会和当前 strategy 设计冲突。

## 7. 详细迁移步骤

建议严格按下面顺序改，不要并行乱改。

### Step 1: 先修工具层基础问题

涉及文件：

- `wuwei/tools/tool.py`
- `wuwei/tools/registry.py`
- `wuwei/tools/executor.py`

必须先修的点：

1. `Tool.invoke()` 正确支持同步函数和异步函数
2. `ToolRegistry.get()` 返回 `None` 而不是抛 `KeyError`
3. `ToolExecutor` 统一处理 missing tool、exception、序列化

建议把 `Tool.invoke()` 改成：

```python
async def invoke(self, args: dict[str, Any] | None = None) -> Any:
    args = args or {}
    if inspect.iscoroutinefunction(self.handler):
        return await self.handler(**args)
    result = self.handler(**args)
    if inspect.isawaitable(result):
        return await result
    return result
```

### Step 2: 新增 `RunSession`

涉及文件：

- 新增 `wuwei/agent/session.py`
- 调整 `wuwei/core/context.py`

目标：

- 把“运行中的状态”从 `Agent` 身上拆下来
- 把 plan 和 artifacts 有地方放

这一步可以暂时保留 `Context`，但新逻辑应该围绕 `RunSession` 写。

### Step 3: 新增 `Runner`

涉及文件：

- 新增 `wuwei/agent/runner.py`
- 新增 `wuwei/agent/result.py`

目标：

- 把运行入口从 `Agent` 中抽走
- 给策略层提供统一能力集

`RunResult` 推荐最小字段：

```python
class RunResult(BaseModel):
    output_text: str | None = None
    finish_reason: str
    session: RunSession
```

### Step 4: 把现有 `Agent` 循环迁移到 `ReActStrategy`

涉及文件：

- 新增 `wuwei/strategies/base.py`
- 新增 `wuwei/strategies/react.py`
- 修改 `wuwei/agent/agent.py`

目标：

- 原有逻辑保留
- 只是位置从 `Agent` 移到 `ReActStrategy`

迁移完成后，`Agent.run()` 只做三件事：

1. 接收参数
2. 构造 `Runner`
3. 调用 `runner.run(..., strategy="react")`

### Step 5: 迁移 `Planner`

涉及文件：

- 新增 `wuwei/planning/base.py`
- 新增 `wuwei/planning/types.py`
- 新增 `wuwei/planning/llm_planner.py`
- 逐步废弃 `wuwei/core/planner.py`

目标：

- 让 planner 成为可插拔组件
- 不再把 planner 写死成独立实验类

此时 `core/planner.py` 可以先保留一个兼容 wrapper，内部直接转调 `LLMPlanner`，并标记 deprecated。

### Step 6: 实现 `PlanAndExecuteStrategy`

涉及文件：

- 新增 `wuwei/strategies/plan_execute.py`
- 更新 `wuwei/agent/runner.py`

目标：

- 让用户可以显式指定 `strategy="plan_execute"`
- 让 planner 真正接入主运行时

### Step 7: 加策略注册表和默认策略

涉及文件：

- `wuwei/agent/runner.py`
- `wuwei/agent/agent.py`
- `wuwei/strategies/__init__.py`

推荐实现：

```python
DEFAULT_STRATEGIES = {
    "react": ReActStrategy,
    "plan_execute": PlanAndExecuteStrategy,
}
```

这样运行时只需要：

```python
await runner.run(agent=agent, user_input="...", strategy="react")
await runner.run(agent=agent, user_input="...", strategy="plan_execute")
```

## 8. 对外 API 建议

推荐保留两种调用方式。

### 8.1 简单调用

```python
agent = Agent(llm=llm, tools=registry, config=AgentConfig(...))

result = await agent.run("帮我总结这份文档", strategy="react")
```

### 8.2 显式运行时调用

```python
runner = Runner()
session = RunSession.new()

result = await runner.run(
    agent=agent,
    user_input="帮我调研这个问题",
    strategy="plan_execute",
    session=session,
)
```

建议保留 `Agent.run()`，因为：

- 新用户用起来简单
- 老代码兼容成本低
- 底层仍然可以升级

## 9. 向后兼容策略

为了避免一次性破坏现有用户代码，建议这样处理：

1. `Agent` 类名不改
2. `AgentConfig` 暂时保留
3. `Agent.run(user_input, stream=False)` 保持可用
4. 只是在内部改为委托给 `Runner`
5. `core/planner.py` 保留兼容入口，但标记 deprecated

兼容期结束后再逐步清理：

- 旧 `Context`
- 旧 `Planner`
- `Agent` 内部直接循环逻辑

## 10. 测试清单

这次重构必须配套测试，否则后面策略层会很快失控。

建议新增：

- `tests/tools/test_tool_invoke.py`
- `tests/tools/test_executor.py`
- `tests/agent/test_runner_react.py`
- `tests/strategies/test_plan_execute.py`
- `tests/planning/test_llm_planner.py`

至少覆盖这些场景：

1. sync tool 可以正常执行
2. async tool 可以正常执行
3. missing tool 返回结构化错误
4. tool exception 返回结构化错误
5. ReAct 在 `tool_calls -> tool result -> final answer` 链路下正常完成
6. ReAct 超过 `max_steps` 后正确退出
7. `plan_execute` 能生成 plan 并逐步执行
8. `plan_execute` 的 step result 会写回 `session.plan`

## 11. 推荐的最小落地顺序

如果你只想先把主线打通，建议按下面顺序开发：

1. 修 `Tool.invoke()` 和 `ToolRegistry.get()`
2. 引入 `RunSession`
3. 引入 `Runner`
4. 新增 `ReActStrategy`
5. 让 `Agent.run()` 改为委托 `Runner`
6. 迁移 `Planner` 到 `planning/`
7. 新增 `PlanAndExecuteStrategy`
8. 补测试

只要做到第 5 步，你的 ReAct runtime 就基本成型了。  
做到第 7 步，你的双策略框架就成型了。

## 12. 推荐的第一版提交拆分

建议拆成以下几个 PR 或 commit：

1. `fix(tools): normalize tool invoke and registry lookup`
2. `feat(agent): add RunSession and RunResult`
3. `feat(agent): introduce Runner`
4. `refactor(agent): move react loop into ReActStrategy`
5. `refactor(planning): move planner into planning module`
6. `feat(strategy): add plan_execute strategy`
7. `test: add runtime and strategy coverage`

这样每一步都可验证，也方便回滚。

## 13. 最终目标形态

重构完成后，Wuwei 的主链路应该变成：

```python
agent = Agent(
    llm=llm,
    tools=registry,
    config=AgentConfig(
        name="research-agent",
        system_prompt="你是一个会调用工具的研究助手",
        max_steps=8,
    ),
)

result = await agent.run(
    "请分析这个需求并给出方案",
    strategy="react",
)

result = await agent.run(
    "请先拆解任务，再逐步执行",
    strategy="plan_execute",
)
```

从框架设计角度看，最终稳定下来的关系应该是：

- `Agent` 是定义
- `Runner` 是运行时
- `ReActStrategy` 是默认执行模式
- `PlanAndExecuteStrategy` 是高阶执行模式
- `ToolExecutor` 是唯一工具执行入口
- `Planner` 是可插拔能力，不再是孤立脚本

这套结构能够支撑后续继续扩展：

- workflow node
- middleware
- tracing
- human approval
- checkpoint / resume

但不会把这些复杂度提前压到当前版本里。

## 14. `Runner` 是什么，为什么要单独设计

一句话定义：

`Runner` 是运行时协调器，负责把一次请求真正跑起来。  
它不是 Agent，也不是 Strategy，而是把 Agent、Strategy、LLM、ToolExecutor、Session 串起来的执行控制器。

如果把整个框架分层来看：

- `Agent` 是静态定义
- `RunSession` 是动态状态
- `ExecutionStrategy` 是执行算法
- `Runner` 是调度者

### 14.1 为什么不能继续把逻辑放在 `Agent`

当前 `Agent` 里已经同时承担：

- 保存配置
- 保存上下文
- 选择工具
- 驱动模型
- 执行工具
- 控制循环退出

这种写法在 demo 阶段足够，但进入框架阶段会有三个问题：

1. 同一个 Agent 无法复用不同运行策略。
2. 一旦要引入 planner、checkpoint、resume、middleware，`Agent` 会继续膨胀。
3. 运行状态和定义耦合后，很难实现多次独立运行。

`Runner` 的价值就是把“运行”这件事从“定义”里拆出来。

### 14.2 `Runner` 应该负责什么

`Runner` 应负责以下事情：

- 创建或接管 `RunSession`
- 保证 system prompt 在 session 中正确初始化
- 根据 `strategy` 选择具体执行策略
- 调用 LLM
- 调用 `ToolExecutor`
- 把模型输出、工具输出和 usage 写回 session
- 生成统一的 `RunResult`
- 未来提供统一 event hooks / tracing 接口

### 14.3 `Runner` 不应该负责什么

这些事情不应该放到 `Runner`：

- 不负责定义 agent 的身份和默认配置
- 不负责编写 planner prompt
- 不负责具体的 ReAct 推理策略
- 不负责业务领域逻辑
- 不负责决定每个 plan step 的业务语义

也就是说，`Runner` 只负责“执行控制”，不负责“业务决策”。

### 14.4 `Runner` 的最小职责模型

可以把 `Runner` 理解成一个最小 runtime kernel：

```text
input
  ->
prepare session
  ->
resolve strategy
  ->
strategy.run(...)
  ->
LLM / ToolExecutor / Planner
  ->
write session
  ->
result
```

它对 strategy 提供能力，而不是替 strategy 做决定。

### 14.5 `Runner` 的建议接口

推荐接口：

```python
class Runner:
    def __init__(
        self,
        tool_executor_cls: type[ToolExecutor] = ToolExecutor,
        strategy_registry: dict[str, type[ExecutionStrategy]] | None = None,
    ):
        self.tool_executor_cls = tool_executor_cls
        self.strategy_registry = strategy_registry or DEFAULT_STRATEGIES

    async def run(
        self,
        agent: Agent,
        user_input: str,
        strategy: str = "react",
        stream: bool = False,
        session: RunSession | None = None,
    ) -> RunResult | AsyncIterator[LLMResponseChunk]:
        ...
```

### 14.6 `Runner` 内部建议拆成哪些方法

建议把 `Runner` 内部能力拆成这些小方法：

```python
class Runner:
    def _prepare_session(self, agent: Agent, session: RunSession | None) -> RunSession:
        ...

    def _append_system_prompt_if_needed(self, agent: Agent, session: RunSession) -> None:
        ...

    def _resolve_strategy(self, strategy: str | ExecutionStrategy) -> ExecutionStrategy:
        ...

    async def _call_llm(
        self,
        agent: Agent,
        session: RunSession,
        *,
        stream: bool = False,
        **kwargs,
    ):
        ...

    async def _execute_tool_calls(
        self,
        agent: Agent,
        tool_calls: list[ToolCall],
    ) -> list[Message]:
        ...

    def _merge_usage(self, session: RunSession, usage: dict[str, int] | None) -> None:
        ...

    def _emit_event(self, event_name: str, **payload) -> None:
        ...
```

这种拆法的好处是：

- `ReActStrategy` 和 `PlanAndExecuteStrategy` 只依赖统一能力
- 后续加 tracing，不需要侵入所有 strategy
- 测试粒度更小

### 14.7 `Runner` 的生命周期

推荐的运行生命周期如下：

1. 接收 `agent / user_input / strategy / session`
2. 如果未传 session，则创建新 session
3. 初始化 system prompt
4. 解析 strategy
5. 发出 `run_started` 事件
6. 调用 strategy 执行
7. 发出 `run_completed` 或 `run_failed` 事件
8. 返回 `RunResult`

未来如果加 `resume`，实际上也是再次交给 `Runner.run(...)`，只是传入已有 session。

### 14.8 `Runner` 和 streaming 的关系

建议 `Runner` 本身不直接编码业务级流式逻辑，只做流式能力透传。

也就是说：

- `ReActStrategy` 可以实现流式回答
- `Runner` 只负责让 strategy 能拿到 `stream=True`
- `Runner` 不应该把所有流式差异逻辑都写死在自己内部

否则 `Runner` 会变成另一个巨型类。

### 14.9 `Runner` 和 `ToolExecutor` 的关系

`Runner` 不自己执行工具，它只持有工具执行能力。

推荐关系：

```python
tool_executor = self.tool_executor_cls(agent.tool_registry)
tool_messages = await tool_executor.execute(tool_calls)
```

这样未来要替换为：

- 带 timeout 的 executor
- 带 tracing 的 executor
- 带权限控制的 executor

都不需要改 strategy。

### 14.10 `Runner` 和 `Planner` 的关系

`Runner` 不直接规划任务，但会把 `runner / session / agent` 注入给 planner。

原因是 planner 可能需要：

- 使用同一个模型网关
- 记录 usage
- 把 plan 写入 session
- 发出 tracing 事件

所以 planner 是 strategy 的依赖，runner 是 planner 的运行能力来源。

## 15. `RunSession` 是什么，为什么不能只用 `Context`

一句话定义：

`RunSession` 是一次运行或一个会话的状态容器。  
它不只是消息列表，而是运行过程中的完整状态快照。

当前 `Context` 只解决了“保存 message history”，但在框架层这远远不够。

### 15.1 `RunSession` 和 `Context` 的本质区别

当前 `Context` 的定位是：

- 一个 message list wrapper

`RunSession` 的定位应该是：

- 当前消息历史
- 当前计划
- 当前执行进度
- 当前 artifacts
- 当前 usage / metrics
- 当前运行元数据

也就是说：

- `Context` 更像消息容器
- `RunSession` 更像运行状态对象

### 15.2 为什么 session 很重要

没有 session，你很难做好这些事情：

- 暂停后恢复执行
- 计划生成后逐步更新 step 状态
- 记录工具中间产物
- 记录 token/latency 指标
- 让多个 strategy 共享同一份运行状态

所以 session 不是“可选抽象”，而是 runtime 的基础对象。

### 15.3 `RunSession` 的生命周期应该怎么理解

建议先明确两个概念：

1. `RunSession` 不是 Agent 本身
2. `RunSession` 也不等于长期 Memory

比较准确的理解是：

- `Agent` 可以被多次运行
- 每次运行都可以产生一个新的 `RunSession`
- 同一个 `RunSession` 也可以被续跑、恢复或追加消息

也就是说，session 是“运行态”，不是“定义态”。

### 15.4 `RunSession` 最少应该包含哪些字段

建议第一版就至少有这些字段：

```python
class RunSession(BaseModel):
    session_id: str
    messages: list[Message] = Field(default_factory=list)
    plan: Plan | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    status: Literal["idle", "running", "completed", "failed"] = "idle"
```

字段解释：

- `session_id`: 运行实例标识，用于恢复、追踪、日志
- `messages`: 完整消息历史
- `plan`: 当前运行生成的计划，给 `plan_execute` 使用
- `artifacts`: 中间产物，比如检索结果、代码片段、文件路径
- `usage`: token、调用次数、延迟聚合
- `metadata`: 其它运行元数据，比如用户 ID、标签、策略名
- `step_count`: 当前执行轮数
- `status`: 当前 session 的整体状态

### 15.5 为什么要有 `artifacts`

很多人一开始只存 messages，但后面会发现工具执行结果不适合全塞回消息。

例如这些更适合放 artifacts：

- 已下载文件路径
- 检索命中文档列表
- 中间结构化 JSON
- 已解析网页正文
- plan step 的原始执行结果

推荐原则：

- 给模型看的内容放 `messages`
- 给 runtime 和后续步骤消费的中间数据放 `artifacts`

### 15.6 为什么要有 `metadata`

`metadata` 不是临时凑数，它很重要。

第一版你可能会放这些：

- `strategy`
- `agent_name`
- `created_at`
- `user_id`
- `tags`

后续这对 tracing、debug、监控都很有价值。

### 15.7 `RunSession` 应该提供哪些方法

建议不要只把 session 做成纯数据包，最好同时提供少量行为方法：

```python
class RunSession(BaseModel):
    ...

    @classmethod
    def new(cls, **metadata) -> "RunSession":
        ...

    def add_system_message(self, content: str) -> None:
        ...

    def add_user_message(self, content: str) -> None:
        ...

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        ...

    def add_tool_message(self, content: str, tool_call_id: str | None) -> None:
        ...

    def merge_usage(self, usage: dict[str, int] | None) -> None:
        ...

    def fork(self) -> "RunSession":
        ...
```

这样 `Context` 里的常用能力就能平滑迁移到 `RunSession`。

### 15.8 `RunSession` 和 Memory 的边界

建议你现在就把 session 和 memory 分清楚：

- `RunSession`: 这次运行的上下文和状态
- `Memory`: 跨运行保存的长期知识

例如：

- 一个用户今天问了 3 次问题，可能会有 3 个 session
- 但它们都可以共享同一个 memory store

如果这两个边界一开始不分清，后面很容易把框架搞乱。

### 15.9 `RunSession` 和 checkpoint / resume 的关系

未来如果要支持恢复执行，session 就是最自然的持久化对象。

推荐方向：

- 把 `RunSession` 设计成可序列化
- 能够保存到文件、数据库或缓存
- 恢复时重新交给 `Runner.run(...)`

也就是说，checkpoint 的最小落点就是 session snapshot。

## 16. 现有架构的模块职责

下面是你当前代码结构中，各模块更准确的职责描述。

### 16.1 `wuwei/agent/base.py`

当前职责：

- 定义 `AgentConfig`
- 定义 `BaseAgent`

问题：

- `AgentConfig` 只覆盖 name、system prompt、max steps，还是偏 demo 级
- `BaseAgent` 抽象过薄，无法表达 strategy 或 session 概念

### 16.2 `wuwei/agent/agent.py`

当前职责：

- 初始化 llm、tools、context
- 执行非流式主循环
- 执行流式主循环
- 在内部直接执行工具调用

问题：

- 把定义和运行耦合在一起
- 没复用 `ToolExecutor`
- 没有显式 strategy 层
- 运行状态只能绑定在 agent 实例内部

### 16.3 `wuwei/core/context.py`

当前职责：

- 保存消息列表
- 添加 system/user/assistant/tool 消息

问题：

- 只能表达消息，不表达运行状态
- 无法承载 plan、artifacts、usage、step progress

### 16.4 `wuwei/core/task.py`

当前职责：

- 定义 `Task` 和 `TaskList`
- 表达 planner 输出的任务结构

问题：

- 语义上更像 planning types，而不是 core 通用概念
- `next` / `status` 只是静态字段，还没有和 runtime 接起来

### 16.5 `wuwei/core/planner.py`

当前职责：

- 用 LLM 生成 DAG 风格计划
- 解析成 `TaskList`

问题：

- 是实验能力，不是 runtime 组件
- 和 Agent 主链路解耦
- 没有 session 接入
- 文件内还有示例运行代码，不适合作为正式框架模块

### 16.6 `wuwei/llm/gateway.py`

当前职责：

- 选择 adapter
- 发起模型请求
- 支持 retry、timeout、stream
- 组装流式 tool call

评价：

- 这是当前架构里最接近正式 runtime 组件的一层
- 它已经有比较清晰的边界

后续建议：

- 保持“模型访问层”定位，不去吸收 agent 运行逻辑

### 16.7 `wuwei/llm/types.py`

当前职责：

- 定义消息、tool call、LLM response 结构

评价：

- 这层方向是对的
- 后续可以继续成为 runtime 的基础类型层

### 16.8 `wuwei/tools/tool.py`

当前职责：

- 定义 `Tool` 和 `ToolParameters`
- 负责把 tool 转成模型 schema
- 负责调用 tool handler

问题：

- `invoke()` 对同步函数处理错误
- 参数校验还不够强
- 默认可变值写法不理想

### 16.9 `wuwei/tools/registry.py`

当前职责：

- 注册工具
- 从函数签名推断工具 schema
- 提供装饰器风格注册方式

问题：

- `get()` 的行为和类型声明不一致
- schema 推断能力还比较弱

### 16.10 `wuwei/tools/executor.py`

当前职责：

- 执行 tool calls
- 返回标准化 `Message(role="tool")`

评价：

- 方向是对的
- 但现在没有成为主链路唯一入口

### 16.11 当前架构一句话总结

当前 Wuwei 更像：

“一个已经有 LLM gateway 和 tool system 的单 Agent ReAct demo runtime”

它不是一套坏架构，但还没有把“定义、运行、状态、策略”拆开。

## 17. 未来架构的模块职责

下面是建议重构后的职责划分。

### 17.1 `wuwei/agent/base.py`

未来职责：

- 定义 `AgentConfig`
- 定义 `BaseAgent`
- 只表达 agent 的静态配置接口

建议扩展：

- `default_strategy`
- `max_steps`
- `planner_model` 或其它可选运行参数

### 17.2 `wuwei/agent/agent.py`

未来职责：

- 表达 Agent 定义
- 保存 llm、tools、config
- 提供面向用户的快捷 `run()`

不再负责：

- 手动驱动工具循环
- 直接维护内部 session

### 17.3 `wuwei/agent/runner.py`

未来职责：

- 单次运行的总控制器
- 统一 strategy 调度入口
- 持有 LLM 调用能力和工具执行能力
- 统一结果产出与事件发射

这是未来 runtime 的核心。

### 17.4 `wuwei/agent/session.py`

未来职责：

- 承载单次运行状态
- 成为 checkpoint / resume 的基础对象
- 提供消息和运行元信息的统一存储

### 17.5 `wuwei/agent/result.py`

未来职责：

- 统一 run 输出格式
- 把最终文本、finish reason、session、错误信息放在一个对象里

这样外部调用者就不必自己猜运行结果语义。

### 17.6 `wuwei/strategies/base.py`

未来职责：

- 定义所有运行策略的统一接口

这层的意义是把“如何运行”从“运行什么”中分离出来。

### 17.7 `wuwei/strategies/react.py`

未来职责：

- 实现默认 ReAct 执行模式

这是你的基础 runtime 策略。

### 17.8 `wuwei/strategies/plan_execute.py`

未来职责：

- 先规划再执行
- 把 planner 接入主运行时
- 复用 ReAct 作为 step executor

这是你的高阶策略，而不是另一套平行 runtime。

### 17.9 `wuwei/planning/base.py`

未来职责：

- 定义 planner 抽象接口

这样 planner 可以有多种实现：

- LLMPlanner
- RuleBasedPlanner
- WorkflowPlanner

### 17.10 `wuwei/planning/llm_planner.py`

未来职责：

- 基于模型生成 plan
- 解析并返回统一 `Plan`

### 17.11 `wuwei/planning/types.py`

未来职责：

- 定义 `Plan`、`PlanStep`
- 成为 planning 层的标准类型

### 17.12 `wuwei/tools/tool.py`

未来职责：

- tool 定义
- schema 导出
- handler 调用入口

### 17.13 `wuwei/tools/registry.py`

未来职责：

- tool 注册中心
- schema 推断
- tool 检索

### 17.14 `wuwei/tools/executor.py`

未来职责：

- 成为唯一工具执行入口
- 统一序列化、错误格式、审计、超时、重试扩展点

### 17.15 `wuwei/llm/gateway.py`

未来职责：

- 保持模型访问层定位
- 向 runtime 提供统一的请求和流式响应能力

### 17.16 未来架构一句话总结

未来 Wuwei 应该是：

“一个以 Runner 为运行核心、以 Session 为状态核心、以 ReAct 为默认策略、以 plan_execute 为高阶策略的轻量 Agent runtime”

## 18. 从“现有架构”到“未来架构”的本质变化

本次重构最本质的变化不是“多了几个文件”，而是抽象边界变化了。

### 18.1 当前边界

```text
Agent
  = 定义 + 状态 + 执行
```

### 18.2 未来边界

```text
Agent
  = 定义

RunSession
  = 状态

Runner
  = 调度

ExecutionStrategy
  = 执行算法
```

这才是从 demo 走向 runtime 的关键一步。

## 19. 设计上的几个关键判断

最后明确几个容易混淆的判断。

### 19.1 `Runner` 不是多余封装

它不是为了“看起来更面向对象”，而是为了真正把：

- 状态
- 策略
- 执行能力

拆开。

### 19.2 `RunSession` 不是换个名字的 `Context`

`Context` 只是消息容器。  
`RunSession` 是运行状态容器。

### 19.3 `plan_execute` 不应该复制一套 ReAct

正确做法是：

- `react` 负责基础工具推理循环
- `plan_execute` 负责先规划、再逐步调用 `react`

### 19.4 你的 runtime 核心不是 planner，而是 runner + session

planner 是重要能力，但不是框架的根。  
真正的根是：

- `Runner`: 驱动运行
- `RunSession`: 承载状态

先把这两个抽象做稳，后面扩展才不会散。
