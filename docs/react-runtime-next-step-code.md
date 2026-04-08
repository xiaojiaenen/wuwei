# Wuwei 下一步重构代码稿

目标：给出一版可以直接照着改的代码草稿。  
原则：先把 `react runtime` 跑稳，再接 `plan_execute`。  
说明：这份文档是“代码稿”，不是最终完整实现；你可以按这里的文件逐步落库。

## 1. 建议的改造顺序

不要一次性全改，按下面顺序落：

1. 修工具层
2. 加 `RunSession`
3. 加 `RunResult`
4. 加 `Runner`
5. 把当前循环迁到 `ReActStrategy`
6. 让 `Agent.run()` 委托给 `Runner`
7. 再上 `planning/` 和 `plan_execute`

---

## 2. 第一阶段代码：先把 ReAct runtime 跑稳

这部分是建议你先真正落库的代码。

### 2.1 `wuwei/tools/tool.py`

先修同步工具报错问题。

相对当前版本，这里改了什么：

- 把 `properties` 和 `required` 从可变默认值改成 `Field(default_factory=...)`
- 修复 `invoke()` 对同步函数错误使用 `await` 的问题
- 增加 `args = args or {}`，避免 `None` 展开报错

```python
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class ToolParameters(BaseModel):
    type: str = "object"
    # 修复：避免多个 ToolParameters 实例共享同一个默认 dict/list
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
        }


class Tool(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    handler: Callable[..., Any] | Callable[..., Awaitable[Any]]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_schema(),
            },
        }

    async def invoke(self, args: dict[str, Any] | None = None) -> Any:
        # 修复：调用方可能传 None，这里统一归一化
        args = args or {}

        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**args)

        # 修复：同步函数不能直接 await，应该先正常调用
        result = self.handler(**args)
        # 兼容：有些同步函数可能返回 awaitable，这里再补一层处理
        if inspect.isawaitable(result):
            return await result
        return result
```

### 2.2 `wuwei/tools/registry.py`

关键点是 `get()` 必须真的返回 `None`。

相对当前版本，这里改了什么：

- `get()` 从 `self._tools[name]` 改成 `self._tools.get(name)`
- `to_schema()` 的返回类型写准确
- 参数推断里补了 `bool -> boolean`
- 代码结构保持和现有版本接近，避免一次性重写太多逻辑

```python
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, get_type_hints

from wuwei.tools.tool import Tool, ToolParameters


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已经注册了")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, tool: Tool):
        if tool.name not in self._tools:
            raise ValueError(f"工具 {tool.name} 没有注册")
        del self._tools[tool.name]

    def get(self, name: str) -> Tool | None:
        # 修复：当前实现会抛 KeyError，这里改成真正返回 None
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_schema(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        parameters: ToolParameters | None = None,
    ):
        def decorator(func: Callable[..., Any] | Callable[..., Awaitable[Any]]):
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or f"执行 {func.__name__} 操作"

            if parameters is not None:
                tool_parameters = parameters
            else:
                sig = inspect.signature(func)
                properties: dict[str, Any] = {}
                required: list[str] = []
                type_hints = get_type_hints(func)

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue

                    hinted = type_hints.get(param_name, str)
                    hinted_name = getattr(hinted, "__name__", str(hinted))
                    if hinted_name in {"int", "float", "complex"}:
                        json_type = "number"
                    # 补充：bool 在工具 schema 里应该映射到 boolean
                    elif hinted_name in {"bool"}:
                        json_type = "boolean"
                    elif hinted_name in {"list", "tuple"}:
                        json_type = "array"
                    elif hinted_name in {"dict"}:
                        json_type = "object"
                    else:
                        json_type = "string"

                    properties[param_name] = {
                        "type": json_type,
                        "description": f"参数 {param_name}",
                    }

                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

                tool_parameters = ToolParameters(
                    properties=properties,
                    required=required,
                )

            self.register(
                Tool(
                    name=tool_name,
                    description=tool_description,
                    parameters=tool_parameters,
                    handler=func,
                )
            )
            return func

        return decorator
```

### 2.3 `wuwei/tools/executor.py`

让它成为唯一工具执行入口。

相对当前版本，这里改了什么：

- 保留现有 `ToolExecutor` 的总体职责
- 把 `ToolNotFound` 和异常处理保留为统一结构化输出
- 未来 `Agent/ReActStrategy` 不再自己执行工具，而是统一调这个类
- 新增 `_error()`，避免拼接错误 JSON 的逻辑散在多个分支

```python
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from wuwei.llm import Message, ToolCall
from wuwei.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_calls: list[ToolCall]) -> list[Message]:
        results: list[Message] = []
        for tool_call in tool_calls:
            results.append(await self.execute_one(tool_call))
        return results

    async def execute_one(self, tool_call: ToolCall) -> Message:
        tool = self.registry.get(tool_call.function.name)
        if tool is None:
            return Message(
                role="tool",
                tool_call_id=tool_call.id,
                # 统一：所有工具找不到的情况都走同一套错误结构
                content=self._error(
                    error_type="ToolNotFound",
                    message=f"Tool '{tool_call.function.name}' not found",
                ),
            )

        try:
            output = await tool.invoke(tool_call.function.arguments)
            # 统一：所有工具输出都在 executor 内部序列化
            content = self.serialize_output(output)
        except Exception as exc:
            # 统一：所有工具异常都转成标准化错误消息
            content = self._error(
                error_type=type(exc).__name__,
                message=str(exc),
            )

        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=content,
        )

    def serialize_output(self, output: Any) -> str:
        if isinstance(output, str):
            return output

        if isinstance(output, BaseModel):
            return output.model_dump_json(exclude_none=True)

        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except TypeError:
            return json.dumps({"value": str(output)}, ensure_ascii=False)

    def _error(self, error_type: str, message: str) -> str:
        return json.dumps(
            {
                "ok": False,
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            ensure_ascii=False,
        )
```

### 2.4 `wuwei/agent/session.py`

这是第一版 `RunSession`。

相对当前仓库，这个文件是新增的。  
它替代了“把运行状态直接挂在 `Agent.context` 上”的方式。

这里引入它，是为了解决这些问题：

- 多次 `agent.run()` 默认污染同一个上下文
- planner 和 plan step 没地方挂
- usage、artifacts、metadata 没地方放
- 未来 resume/checkpoint 没有状态对象可持久化

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from wuwei.llm import Message, ToolCall


class RunSession(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    # 这次运行的消息历史
    messages: list[Message] = Field(default_factory=list)
    # 预留给 plan_execute 的计划对象
    plan: Any | None = None
    # 中间产物，不一定都适合塞回 message
    artifacts: dict[str, Any] = Field(default_factory=dict)
    # token / latency 等累计指标
    usage: dict[str, int] = Field(default_factory=dict)
    # 用户 ID、策略名等运行元数据
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 当前 ReAct 已经走了多少步
    step_count: int = 0
    status: Literal["idle", "running", "completed", "failed"] = "idle"

    @classmethod
    def new(cls, **metadata: Any) -> "RunSession":
        return cls(metadata=metadata)

    def add_system_message(self, content: str) -> None:
        self.messages.append(Message(role="system", content=content))

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_tool_message(self, content: str, tool_call_id: str | None) -> None:
        self.messages.append(
            Message(role="tool", content=content, tool_call_id=tool_call_id)
        )

    def merge_usage(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        for key, value in usage.items():
            # 聚合：把每次 llm/tool 返回的 usage 累加进 session
            self.usage[key] = self.usage.get(key, 0) + value

    def has_system_message(self) -> bool:
        return any(message.role == "system" for message in self.messages)

    def fork(self) -> "RunSession":
        # 预留：后面 plan_execute 执行单独 step 时可以复制一份 session
        return RunSession(
            messages=deepcopy(self.messages),
            plan=deepcopy(self.plan),
            artifacts=deepcopy(self.artifacts),
            usage=deepcopy(self.usage),
            metadata=deepcopy(self.metadata),
            step_count=self.step_count,
            status=self.status,
        )
```

### 2.5 `wuwei/agent/result.py`

统一返回对象。

相对当前仓库，这个文件是新增的。  
当前非流式 `Agent.run()` 直接返回字符串，后续很难携带 session、finish reason、error。

这个类的作用是统一：

- 最终文本
- 退出原因
- 运行后的 session
- 错误信息

```python
from __future__ import annotations

from pydantic import BaseModel

from wuwei.agent.session import RunSession


class RunResult(BaseModel):
    output_text: str | None = None
    finish_reason: str
    # 把运行后的完整状态一并返回出去
    session: RunSession
    error: str | None = None

    @classmethod
    def final(cls, output_text: str | None, session: RunSession) -> "RunResult":
        session.status = "completed"
        return cls(
            output_text=output_text,
            finish_reason="stop",
            session=session,
        )

    @classmethod
    def limit_reached(cls, session: RunSession) -> "RunResult":
        # 当前实现里 max_steps 也算运行失败的一种
        session.status = "failed"
        return cls(
            output_text="任务未完成，已达到最大步骤限制",
            finish_reason="max_steps",
            session=session,
        )

    @classmethod
    def failed(cls, session: RunSession, error: str) -> "RunResult":
        session.status = "failed"
        return cls(
            output_text=None,
            finish_reason="error",
            session=session,
            error=error,
        )
```

### 2.6 `wuwei/strategies/base.py`

相对当前仓库，这个文件是新增的。  
它解决的是：当前 ReAct 逻辑虽然存在，但没有被显式抽象成策略。

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wuwei.agent.agent import Agent
    from wuwei.agent.runner import Runner
    from wuwei.agent.session import RunSession


class ExecutionStrategy(ABC):
    name: str

    @abstractmethod
    async def run(
        self,
        runner: "Runner",
        agent: "Agent",
        session: "RunSession",
        user_input: str,
        stream: bool = False,
    ) -> Any:
        raise NotImplementedError
```

### 2.7 `wuwei/strategies/react.py`

先把你当前主循环迁到这里。

相对当前版本，这里改了什么：

- 把 `Agent` 里 `_run_non_stream()` 和 `_run_stream()` 的主循环搬到 `ReActStrategy`
- LLM 调用统一通过 `runner._call_llm(...)`
- tool call 统一通过 `runner._execute_tool_calls(...)`
- usage 聚合写入 `RunSession`
- 非流式返回 `RunResult`，而不是裸字符串

```python
from __future__ import annotations

from typing import Any, AsyncIterator

from wuwei.agent.result import RunResult
from wuwei.llm import LLMResponse, LLMResponseChunk
from wuwei.strategies.base import ExecutionStrategy


class ReActStrategy(ExecutionStrategy):
    name = "react"

    async def run(
        self,
        runner,
        agent,
        session,
        user_input: str,
        stream: bool = False,
    ) -> Any:
        if stream:
            return self._run_stream(runner, agent, session, user_input)
        return await self._run_non_stream(runner, agent, session, user_input)

    async def _run_non_stream(self, runner, agent, session, user_input: str) -> RunResult:
        session.add_user_message(user_input)
        session.status = "running"

        while session.step_count < agent.config.max_steps:
            # 统一：所有策略都通过 runner 调模型
            response: LLMResponse = await runner._call_llm(agent, session, stream=False)
            # 统一：usage 不再丢掉，累计进 session
            session.merge_usage(response.usage)
            session.add_assistant_message(
                content=response.message.content,
                tool_calls=response.message.tool_calls,
            )

            if response.finish_reason == "tool_calls" and response.message.tool_calls:
                # 统一：工具执行不再直接在 strategy 里写 handler 调用逻辑
                tool_messages = await runner._execute_tool_calls(agent, response.message.tool_calls)
                for message in tool_messages:
                    session.messages.append(message)
                session.step_count += 1
                continue

            return RunResult.final(response.message.content, session=session)

        return RunResult.limit_reached(session=session)

    async def _run_stream(
        self,
        runner,
        agent,
        session,
        user_input: str,
    ) -> AsyncIterator[LLMResponseChunk]:
        session.add_user_message(user_input)
        session.status = "running"

        while session.step_count < agent.config.max_steps:
            full_content = ""
            full_tool_calls = None

            stream = await runner._call_llm(agent, session, stream=True)
            async for chunk in stream:
                full_content += chunk.content
                if chunk.tool_calls_complete:
                    full_tool_calls = chunk.tool_calls_complete
                if chunk.usage:
                    # 流式下也把 usage 统一汇总到 session
                    session.merge_usage(chunk.usage)
                if chunk.content:
                    yield chunk

            session.add_assistant_message(
                content=full_content,
                tool_calls=full_tool_calls,
            )

            if full_tool_calls:
                tool_messages = await runner._execute_tool_calls(agent, full_tool_calls)
                for message in tool_messages:
                    session.messages.append(message)
                session.step_count += 1
                continue

            session.status = "completed"
            break
```

### 2.8 `wuwei/strategies/__init__.py`

相对当前仓库，这个文件是新增的。  
它的作用是集中维护策略注册表，避免把 `"react"` / `"plan_execute"` 这些字符串散落在多个文件。

```python
from wuwei.strategies.react import ReActStrategy

DEFAULT_STRATEGIES = {
    "react": ReActStrategy,
}

__all__ = ["ReActStrategy", "DEFAULT_STRATEGIES"]
```

### 2.9 `wuwei/agent/runner.py`

这是第一版运行时控制器。

相对当前仓库，这个文件是新增的。  
它是整个重构里最关键的部分。

这里具体替代了原来 `Agent` 里哪些职责：

- 原来 `Agent` 里初始化上下文的逻辑，迁到 `_prepare_session()`
- 原来 `Agent` 里直接调 `llm.generate()`，迁到 `_call_llm()`
- 原来 `Agent` 里直接执行工具，迁到 `_execute_tool_calls()`
- 原来 `Agent` 自己决定执行哪套主循环，现在改成 `_resolve_strategy()`

```python
from __future__ import annotations

from typing import AsyncIterator

from wuwei.agent.result import RunResult
from wuwei.agent.session import RunSession
from wuwei.llm import LLMResponse, LLMResponseChunk, Message, ToolCall
from wuwei.tools import ToolExecutor


class Runner:
    def __init__(
        self,
        tool_executor_cls: type[ToolExecutor] = ToolExecutor,
        strategy_registry: dict[str, type] | None = None,
    ):
        if strategy_registry is None:
            from wuwei.strategies import DEFAULT_STRATEGIES

            strategy_registry = DEFAULT_STRATEGIES

        self.tool_executor_cls = tool_executor_cls
        self.strategy_registry = strategy_registry

    async def run(
        self,
        agent,
        user_input: str,
        strategy: str = "react",
        stream: bool = False,
        session: RunSession | None = None,
    ) -> RunResult | AsyncIterator[LLMResponseChunk]:
        # 新增：每次运行先准备 session，而不是直接复用 agent 内部 context
        session = self._prepare_session(agent, session)
        # 新增：运行模式从字符串解析成策略对象
        strategy_obj = self._resolve_strategy(strategy)

        return await strategy_obj.run(
            runner=self,
            agent=agent,
            session=session,
            user_input=user_input,
            stream=stream,
        )

    def _prepare_session(self, agent, session: RunSession | None) -> RunSession:
        session = session or RunSession.new(
            agent_name=agent.config.name,
            strategy=agent.default_strategy,
        )
        # 新增：system prompt 的注入从 Agent.__init__ 挪到运行时处理
        self._append_system_prompt_if_needed(agent, session)
        return session

    def _append_system_prompt_if_needed(self, agent, session: RunSession) -> None:
        if not session.has_system_message():
            session.add_system_message(agent.config.system_prompt)

    def _resolve_strategy(self, strategy: str):
        strategy_cls = self.strategy_registry.get(strategy)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {strategy}")
        return strategy_cls()

    async def _call_llm(
        self,
        agent,
        session: RunSession,
        *,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | AsyncIterator[LLMResponseChunk]:
        # 统一：所有策略对模型的访问入口都走这里
        return await agent.llm.generate(
            session.messages,
            tools=agent.tools,
            stream=stream,
            **kwargs,
        )

    async def _execute_tool_calls(
        self,
        agent,
        tool_calls: list[ToolCall],
    ) -> list[Message]:
        if not hasattr(agent, "tool_registry"):
            return []

        # 统一：所有工具执行都走 ToolExecutor
        executor = self.tool_executor_cls(agent.tool_registry)
        return await executor.execute(tool_calls)
```

### 2.10 `wuwei/agent/base.py`

这里只需要补默认 strategy。

相对当前版本，这里改了什么：

- `AgentConfig` 新增 `default_strategy`
- `BaseAgent.run()` 改成接受 `**kwargs`，为 `strategy/session/stream` 这些扩展参数留口子

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str = "WUWEI AGENT"
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 10
    # 新增：允许 AgentConfig 声明默认运行策略
    default_strategy: str = "react"


class BaseAgent(ABC):
    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> Any:
        raise NotImplementedError
```

### 2.11 `wuwei/agent/agent.py`

现在它只做定义和委托。

相对当前版本，这里改了什么：

- 删除 `self.context = Context()`
- 删除 `_run_non_stream()` 和 `_run_stream()`
- `Agent` 不再自己执行工具和控制循环
- `run()` 只负责把请求委托给 `Runner`

这一步就是把：

- `Agent = 定义 + 状态 + 执行`

改成：

- `Agent = 定义`

```python
from __future__ import annotations

from typing import Any

from wuwei.agent.base import AgentConfig, BaseAgent
from wuwei.agent.runner import Runner
from wuwei.llm import LLMGateway
from wuwei.tools import ToolRegistry


class Agent(BaseAgent):
    def __init__(
        self,
        llm: LLMGateway,
        tools: list | ToolRegistry | None = None,
        config: AgentConfig | None = None,
        default_strategy: str | None = None,
    ):
        self.llm = llm
        self.config = config or AgentConfig()
        self.default_strategy = default_strategy or self.config.default_strategy

        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
            self.tools = tools.list_tools()
        elif tools is None:
            # 修正：即使没有传 tools，也统一初始化一个 registry
            self.tool_registry = ToolRegistry()
            self.tools = []
        else:
            self.tool_registry = ToolRegistry()
            for tool in tools:
                self.tool_registry.register(tool)
            self.tools = self.tool_registry.list_tools()

    async def run(
        self,
        user_input: str,
        strategy: str | None = None,
        stream: bool = False,
        session=None,
    ) -> Any:
        # 新设计：Agent 只作为用户侧便捷入口，真正执行交给 Runner
        runner = Runner()
        return await runner.run(
            agent=self,
            user_input=user_input,
            strategy=strategy or self.default_strategy,
            stream=stream,
            session=session,
        )
```

### 2.12 `wuwei/agent/__init__.py`

相对当前版本，这里改了什么：

- 导出 `Runner`
- 导出 `RunSession`
- 导出 `RunResult`

```python
from wuwei.agent.agent import Agent
from wuwei.agent.base import AgentConfig, BaseAgent
from wuwei.agent.result import RunResult
from wuwei.agent.runner import Runner
from wuwei.agent.session import RunSession

__all__ = [
    "Agent",
    "AgentConfig",
    "BaseAgent",
    "RunResult",
    "RunSession",
    "Runner",
]
```

### 2.13 第一阶段落地后，外部调用方式

```python
agent = Agent(
    llm=llm,
    tools=registry,
    config=AgentConfig(
        name="demo-agent",
        system_prompt="你是一个会调用工具的中文助手",
        max_steps=5,
    ),
)

result = await agent.run("帮我查一下上海天气", strategy="react")
print(result.output_text)
```

如果你想复用同一个 session：

```python
session = RunSession.new(user_id="u_001")

result1 = await agent.run("帮我查一下上海天气", session=session)
result2 = await agent.run("再比较一下北京天气", session=session)
```

这里 session 复用就是显式行为，不再是当前 `Agent` 内部偷偷复用 `context`。

---

## 3. 第二阶段代码：接入 planning 和 `plan_execute`

这部分建议在第一阶段稳定后再落。

### 3.1 `wuwei/planning/types.py`

相对当前版本，这部分对应原来的 `wuwei/core/task.py`，只是把语义改明确：

- `Task` 改成 `PlanStep`
- `TaskList` 改成 `Plan`
- 把 planning 类型放回 `planning/`，不再放在 `core/`

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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

### 3.2 `wuwei/planning/base.py`

相对当前仓库，这个文件是新增的。  
作用是把 planner 先抽象成接口，而不是直接把某个 LLM 实现写死。

```python
from __future__ import annotations

from abc import ABC, abstractmethod

from wuwei.planning.types import Plan


class Planner(ABC):
    @abstractmethod
    async def create_plan(self, goal: str, runner, agent, session) -> Plan:
        raise NotImplementedError
```

### 3.3 `wuwei/planning/llm_planner.py`

相对当前 `wuwei/core/planner.py`，这里改了什么：

- `goal` 不再在构造函数里写死，而是在 `create_plan(goal=...)` 时传入
- planner 不再作为独立脚本运行
- planner 现在是可插拔组件，可以被 strategy 调用
- 删除文件尾部 `__main__` 风格的运行逻辑

```python
from __future__ import annotations

from wuwei.llm import Message
from wuwei.planning.base import Planner
from wuwei.planning.types import Plan


class LLMPlanner(Planner):
    def __init__(self, llm=None):
        self.llm = llm

    def build_plan_prompt(self, goal: str) -> str:
        return f"""
# Role
你是一个高级工作流规划引擎。

# Task
请为以下目标生成计划：{goal}

# Rules
1. 任务必须原子化。
2. 最多 5 个步骤。
3. 返回 JSON。

# Output Schema
{{
  "goal": "string",
  "steps": [
    {{
      "id": 1,
      "description": "string",
      "status": "pending",
      "next_steps": []
    }}
  ]
}}
""".strip()

    async def create_plan(self, goal: str, runner, agent, session) -> Plan:
        # 优先复用 agent.llm，避免 planner 自己再维护一套模型配置
        llm = self.llm or agent.llm
        response = await llm.generate(
            messages=[Message(role="user", content=self.build_plan_prompt(goal))],
            stream=False,
            response_format={"type": "json_object"},
        )
        return Plan.model_validate_json(response.message.content)
```

### 3.4 `wuwei/planning/__init__.py`

```python
from wuwei.planning.base import Planner
from wuwei.planning.llm_planner import LLMPlanner
from wuwei.planning.types import Plan, PlanStep

__all__ = ["Planner", "LLMPlanner", "Plan", "PlanStep"]
```

### 3.5 `wuwei/strategies/plan_execute.py`

这里有一个关键设计：`plan_execute` 不重新发明一套工具循环，而是复用 `react`。

相对当前仓库，这个文件是新增的。  
它解决的是：你已经有 planner 雏形，但还没有一个“先规划再执行”的正式运行模式。

这里最关键的改动不是 planner 本身，而是：

- planner 生成 plan
- 每个 step 的执行仍然走 ReAct
- 这样不会复制第二套工具执行循环

```python
from __future__ import annotations

from wuwei.agent.result import RunResult
from wuwei.planning import LLMPlanner, Planner
from wuwei.strategies.base import ExecutionStrategy
from wuwei.strategies.react import ReActStrategy


class PlanAndExecuteStrategy(ExecutionStrategy):
    name = "plan_execute"

    def __init__(self, planner: Planner | None = None):
        self.planner = planner or LLMPlanner()
        self.react = ReActStrategy()

    async def run(self, runner, agent, session, user_input: str, stream: bool = False):
        if stream:
            raise ValueError("plan_execute 暂不支持 stream=True")

        session.add_user_message(user_input)
        session.status = "running"

        # 第一步：先生成计划
        plan = await self.planner.create_plan(
            goal=user_input,
            runner=runner,
            agent=agent,
            session=session,
        )
        session.plan = plan

        for step in session.plan.steps:
            if step.status == "completed":
                continue

            step.status = "in_progress"
            # 关键：每个 step 不是手写一套执行器，而是复用 react
            step_prompt = (
                "你正在执行一个计划中的单独步骤。\n"
                "只完成当前步骤，不要尝试完成整个大任务。\n\n"
                f"当前步骤: {step.description}"
            )

            # fork 一份 session，避免 step 执行过程把总 session 搅乱
            step_session = session.fork()
            step_result = await self.react._run_non_stream(
                runner=runner,
                agent=agent,
                session=step_session,
                user_input=step_prompt,
            )

            step.result = step_result.output_text
            step.status = "completed"

        final_summary = self._summarize_plan(session)
        return RunResult.final(final_summary, session=session)

    def _summarize_plan(self, session) -> str:
        lines = ["计划执行完成："]
        for step in session.plan.steps:
            lines.append(f"{step.id}. {step.description}")
            lines.append(f"结果: {step.result or ''}")
        return "\n".join(lines)
```

### 3.6 更新 `wuwei/strategies/__init__.py`

```python
from wuwei.strategies.plan_execute import PlanAndExecuteStrategy
from wuwei.strategies.react import ReActStrategy

DEFAULT_STRATEGIES = {
    "react": ReActStrategy,
    "plan_execute": PlanAndExecuteStrategy,
}

__all__ = [
    "ReActStrategy",
    "PlanAndExecuteStrategy",
    "DEFAULT_STRATEGIES",
]
```

---

## 4. 你现在真正应该先改哪几处

如果你问“下一步不是最终形态，而是马上应该写什么代码”，答案是：

### P0：立刻改

- `wuwei/tools/tool.py`
- `wuwei/tools/registry.py`
- `wuwei/tools/executor.py`

原因：

- 这是当前已经有 bug 的地方
- 不修这里，后面 `Runner` 接起来也不稳

### P1：接着加

- `wuwei/agent/session.py`
- `wuwei/agent/result.py`
- `wuwei/agent/runner.py`
- `wuwei/strategies/base.py`
- `wuwei/strategies/react.py`

原因：

- 这是把当前架构从 demo 变成 runtime 的关键

### P2：最后再接

- `wuwei/planning/`
- `wuwei/strategies/plan_execute.py`

原因：

- `plan_execute` 依赖前面的 runtime 抽象
- 先上 planner，反而会把结构继续搞乱

---

## 5. 一个最小 smoke test

第一阶段改完后，建议先用这个脚本验证：

```python
import asyncio

from wuwei.agent import Agent, AgentConfig, RunSession
from wuwei.llm import LLMGateway
from wuwei.tools import ToolRegistry


registry = ToolRegistry()


@registry.tool(description="回显输入")
def echo(text: str) -> str:
    return f"echo: {text}"


async def main():
    llm = LLMGateway(
        {
            "provider": "openai",
            "api_key": "YOUR_API_KEY",
            "model": "gpt-5.4",
            "temperature": 0.2,
        }
    )

    agent = Agent(
        llm=llm,
        tools=registry,
        config=AgentConfig(
            name="demo-agent",
            system_prompt="你是一个会调用工具的助手",
            max_steps=5,
        ),
    )

    session = RunSession.new(user_id="demo")
    result = await agent.run("请调用 echo 工具输出 hello", session=session)
    print(result.output_text)
    print(result.session.usage)


asyncio.run(main())
```

---

## 6. 这份代码稿的使用方式

建议你这样用这份文档：

1. 先按 `2.x` 部分改源码
2. 跑通最小 smoke test
3. 再按 `3.x` 部分接 `plan_execute`

不要反过来。

如果你愿意，下一步我可以不只给文档稿，而是直接把第一阶段代码真正改进仓库里。  
那样我会先落：

- `tool.py`
- `registry.py`
- `executor.py`
- `session.py`
- `result.py`
- `runner.py`
- `strategies/react.py`
- `agent.py`
