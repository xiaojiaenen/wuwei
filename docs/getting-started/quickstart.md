# 快速上手

本指南将带你从零开始运行第一个 Wuwei Agent。

## 1. 配置 LLM

Wuwei 通过环境变量配置 LLM。创建一个 `.env` 文件：

```bash
# .env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

!!! tip "支持 OpenAI 兼容协议"
    Wuwei 使用 OpenAI 兼容协议，支持任何兼容的 LLM 服务：
    
    - OpenAI (GPT-4o, GPT-4, etc.)
    - Azure OpenAI
    - 本地模型 (Ollama, vLLM, etc.)
    - 第三方服务 (DeepSeek, Moonshot, etc.)
    
    只需修改 `OPENAI_BASE_URL` 即可切换服务。

## 2. 最小示例

```python title="hello.py"
import asyncio
from wuwei import Agent

async def main():
    # 从环境变量创建 Agent
    agent = Agent.from_env(
        builtin_tools=["time"],
        system_prompt="你是一个有用的助手",
    )
    
    # 运行一次对话
    result = await agent.run("现在几点？")
    print(result.content)
    # => 现在是 2026 年 4 月 29 日 22:30:00

asyncio.run(main())
```

运行：

```bash
python hello.py
```

## 3. 添加自定义工具

```python title="custom_tool.py"
import asyncio
from wuwei import Agent, Tool, ToolParameters

# 方式一：使用装饰器注册
async def main():
    agent = Agent.from_env(system_prompt="你是一个数学助手")
    
    @agent.tool_registry.tool(
        name="add",
        description="两数相加",
    )
    def add(a: float, b: float) -> float:
        return a + b
    
    @agent.tool_registry.tool(
        name="multiply", 
        description="两数相乘",
    )
    def multiply(a: float, b: float) -> float:
        return a * b
    
    result = await agent.run("帮我算一下 (3 + 5) × 7 等于多少")
    print(result.content)

asyncio.run(main())
```

## 4. 流式输出

```python title="streaming.py"
import asyncio
from wuwei import Agent

async def main():
    agent = Agent.from_env(builtin_tools=["time"])
    
    # 方式一：流式 chunk
    async for chunk in await agent.run("写一首关于AI的诗", stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    
    # 方式二：结构化事件流
    async for event in agent.stream_events("现在几点？"):
        if event.type == "text_delta":
            print(event.data["content"], end="", flush=True)
        elif event.type == "tool_start":
            print(f"
[调用工具] {event.data['tool_name']}")
        elif event.type == "done":
            print(f"
[完成] 耗时 {event.data['latency_ms']}ms")

asyncio.run(main())
```

## 5. 多轮会话

```python title="session.py"
import asyncio
from wuwei import Agent

async def main():
    agent = Agent.from_env(builtin_tools=["time", "calc"])
    
    # 创建会话
    session = agent.create_session(
        session_id="demo",
        system_prompt="你是一个友好的助手",
    )
    
    # 第一轮
    result = await agent.run("你好，我叫小明", session=session)
    print(result.content)
    
    # 第二轮 —— Agent 能记住上下文
    result = await agent.run("我叫什么名字？", session=session)
    print(result.content)  # => 你叫小明

asyncio.run(main())
```

## 下一步

- [配置](configuration.md) — 深入了解配置选项
- [Agent](../core/agent.md) — 了解 Agent 的完整用法
- [工具系统](../tools/overview.md) — 学习如何创建自定义工具
- [Hook 系统](../core/hooks.md) — 扩展 Agent 行为