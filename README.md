# Wuwei

Wuwei 是一个轻量级、可扩展的 Agent 框架，目标是把「模型调用」「上下文管理」「工具注册与执行」拆成清晰的模块，让你可以用很少的代码搭出带工具能力的智能体。

当前版本已经具备这些核心能力：

- 通过 `Agent` 驱动多轮推理与工具调用
- 通过 `Context` 管理 system / user / assistant / tool 消息
- 通过 `LLMGateway` 统一封装模型请求、重试和流式输出
- 通过 `ToolRegistry` / `ToolExecutor` 注册并执行工具
- 基于 OpenAI SDK，支持标准 OpenAI 接口，也支持兼容 OpenAI 协议的 `base_url`

## 当前目录结构

```text
wuwei/
├─ wuwei/
│  ├─ core/
│  │  ├─ agent.py
│  │  ├─ base.py
│  │  └─ context.py
│  ├─ llm/
│  │  ├─ adapters/
│  │  │  ├─ base.py
│  │  │  └─ openai.py
│  │  ├─ gateway.py
│  │  └─ types.py
│  └─ tools/
│     ├─ executor.py
│     ├─ registry.py
│     └─ tool.py
└─ pyproject.toml
```

## 安装

要求：

- Python `>=3.10`

使用 `pip`：

```bash
pip install -e .
```

使用 `uv`：

```bash
uv sync
```

开发依赖：

```bash
pip install -e ".[dev]"
```

## 快速开始

下面是一个最小可运行的 Agent 示例，展示了模型、上下文和工具如何串起来。

```python
import asyncio

from wuwei.agent.agent import Agent
from wuwei.agent.base import AgentConfig
from wuwei.llm import LLMGateway
from wuwei.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool(description="获取指定城市的天气")
async def get_weather(city: str) -> str:
  weather_data = {
    "北京": "晴，25C",
    "上海": "多云，23C",
    "广州": "阵雨，28C",
  }
  return weather_data.get(city, f"暂未找到 {city} 的天气信息")


async def main():
  llm = LLMGateway(
    {
      "provider": "openai",
      "api_key": "YOUR_API_KEY",
      "model": "gpt-5.4",
      # 如果你使用兼容 OpenAI 协议的平台，可以加上 base_url
      # "base_url": "https://your-provider.example/v1",
      "temperature": 0.2,
    }
  )

  agent = Agent(
    llm=llm,
    tools=registry,
    config=AgentConfig(
      name="demo-agent",
      system_prompt="你是一个会调用工具的中文助手。",
      max_steps=5,
    ),
  )

  result = await agent.run("帮我查一下上海天气")
  print(result)


asyncio.run(main())
```

如果你想用流式输出：

```python
async for chunk in await agent.run("帮我查一下广州天气", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

说明：

- 当前版本建议工具处理函数统一使用 `async def`
- 如果你接入兼容 OpenAI 协议的平台，通常只需要调整 `base_url` 和 `model`

## 模块说明

### `wuwei.core`

Agent 运行时主干，负责把模型、消息历史和工具执行串成一条完整链路。

#### `base.py`

- `AgentConfig`
  - Agent 基础配置
  - 当前包含 `name`、`system_prompt`、`max_steps`
- `BaseAgent`
  - Agent 抽象基类
  - 约定所有实现都提供异步 `run()`

#### `context.py`

- `Context`
  - 维护消息列表
  - 提供 `add_system_message()`、`add_user_message()`、`add_ai_message()`、`add_tool_message()`
  - Agent 每一轮都会从这里读取完整历史发给模型

#### `agent.py`

- `Agent`
  - 框架最核心的执行器
  - 支持非流式 `run(..., stream=False)` 和流式 `run(..., stream=True)`
  - 内部流程是：
    1. 写入用户消息
    2. 调用 `LLMGateway.generate()`
    3. 如果模型返回 `tool_calls`，就执行工具并把结果追加到上下文
    4. 继续下一轮推理，直到模型直接返回最终答案或达到 `max_steps`

### `wuwei.llm`

模型访问层，负责把框架内部消息格式转换成具体厂商 SDK 请求。

#### `types.py`

- `Message`
  - 统一的消息结构，支持 `system`、`user`、`assistant`、`tool`
- `FunctionCall` / `ToolCall`
  - 工具调用描述
- `LLMResponse`
  - 非流式响应对象
- `LLMResponseChunk`
  - 流式增量响应对象

#### `gateway.py`

- `LLMGateway`
  - 模型网关
  - 根据配置选择 adapter
  - 统一处理超时、重试、单次请求和流式请求
  - 当前默认配置项包括：
    - `provider`
    - `api_key`
    - `model`
    - `base_url`
    - `temperature`
    - `max_tokens`
    - `retry`
    - `timeout`

#### `adapters/base.py`

- `BaseAdapter`
  - 模型适配器抽象接口
  - 定义 `build_request()`、`call()`、`parse_response()`、`parse_stream_chunk()`

#### `adapters/openai.py`

- `OpenAIAdapter`
  - 当前唯一内置适配器
  - 使用 `openai.AsyncOpenAI`
  - 既可以直连 OpenAI，也可以通过 `base_url` 访问兼容 OpenAI API 的服务

### `wuwei.tools`

工具系统，负责描述工具 schema、注册工具、执行工具以及标准化返回结果。

#### `tool.py`

- `ToolParameters`
  - 定义工具参数 schema
- `Tool`
  - 定义工具元信息：`name`、`description`、`parameters`、`handler`
  - `to_schema()` 可直接转换为模型函数调用 schema
  - `invoke()` 负责真正调用工具函数

#### `registry.py`

- `ToolRegistry`
  - 工具注册中心
  - 支持 `register()`、`unregister()`、`get()`、`list_tools()`
  - 提供 `@registry.tool()` 装饰器，可从函数签名自动推断参数 schema

#### `executor.py`

- `ToolExecutor`
  - 独立的工具执行器
  - 接收 `ToolCall`，执行对应工具，并返回标准 `Message(role="tool")`
  - 如果工具不存在或执行报错，会返回结构化错误信息

## 框架执行流程

```text
User Input
   ↓
Context.add_user_message()
   ↓
Agent 调用 LLMGateway.generate()
   ↓
LLM 返回普通文本 或 tool_calls
   ↓
如果是 tool_calls:
  ToolRegistry / ToolExecutor 执行工具
   ↓
  Context.add_tool_message()
   ↓
  再次请求模型
   ↓
最终答案
```

## Example 1：带工具的 Agent

这个例子适合快速理解框架主链路。

```python
import asyncio

from wuwei.agent.agent import Agent
from wuwei.agent.base import AgentConfig
from wuwei.llm import LLMGateway
from wuwei.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool(description="执行简单数学运算")
async def calculate(a: float, b: float, operation: str) -> str:
  if operation == "add":
    return str(a + b)
  if operation == "subtract":
    return str(a - b)
  if operation == "multiply":
    return str(a * b)
  if operation == "divide":
    if b == 0:
      return "除数不能为 0"
    return str(a / b)
  return f"不支持的 operation: {operation}"


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
      system_prompt="你可以在需要时调用工具完成计算。",
      max_steps=5,
    ),
  )

  result = await agent.run("请帮我计算 20 除以 4")
  print(result)


asyncio.run(main())
```

## Example 2：离线执行 ToolExecutor

如果你想单独测试工具层，而不连接模型，可以直接构造 `ToolCall` 交给 `ToolExecutor`。

```python
import asyncio

from wuwei.llm import FunctionCall, ToolCall
from wuwei.tools import ToolExecutor, ToolRegistry


registry = ToolRegistry()


@registry.tool(description="回显输入")
async def echo(text: str) -> str:
    return f"echo: {text}"


async def main():
    executor = ToolExecutor(registry)

    tool_call = ToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(
            name="echo",
            arguments={"text": "hello"},
        ),
    )

    result = await executor.execute_one(tool_call)
    print(result.content)


asyncio.run(main())
```

## 适合扩展的方向

从现在的代码结构来看，后续最容易扩展的是这几部分：

- 在 `wuwei.llm.adapters` 下增加更多模型提供方适配器
- 在 `wuwei.core.agent` 中补充更完整的错误处理、日志和回调机制
- 为 `tools` 增加更强的参数校验、同步函数支持和工具结果格式约束
- 增加独立的 `examples/` 和正式的 `pytest` 单元测试

## 当前版本说明

当前仓库更接近一个可运行的 Alpha 版本骨架，优点是结构清晰、模块边界明确，适合继续往上加：

- 自定义 Agent 行为
- 多工具协同
- 多模型适配
- 更完整的可观测性和测试体系

如果你想继续扩展，建议优先从 `Agent -> LLMGateway -> ToolRegistry/ToolExecutor` 这条主链路入手。
