# 最小示例

最简单的 Wuwei Agent 只需几行代码。

## 基础用法

```python
import asyncio
from wuwei import Agent

async def main():
    # 从环境变量创建 Agent（需要 OPENAI_API_KEY）
    agent = Agent.from_env()

    # 运行一次
    result = await agent.run("你好，介绍一下你自己")
    print(result.content)

asyncio.run(main())
```

## 带内置工具

```python
import asyncio
from wuwei import Agent

async def main():
    agent = Agent.from_env(
        builtin_tools=["time", "calc"],
        system_prompt="你是一个数学助手，可以获取当前时间和计算表达式。",
    )

    result = await agent.run("现在几点了？帮我算一下 sqrt(144) + 3.14")
    print(result.content)
    print(f"Token 用量: {result.usage}")

asyncio.run(main())
```

## 带自定义工具

```python
import asyncio
from wuwei import Agent

def search_db(query: str, limit: int = 10) -> dict:
    """搜索数据库。

    :param query: 搜索关键词
    :param limit: 返回结果数量
    """
    # 模拟数据库查询
    return {"results": [{"id": 1, "title": f"结果: {query}"}], "total": 1}

async def main():
    agent = Agent.from_env(
        tools=[search_db],  # 自动从签名生成 Schema
    )

    result = await agent.run("帮我搜索关于机器学习的文章")
    print(result.content)

asyncio.run(main())
```

## 流式输出

```python
import asyncio
from wuwei import Agent

async def main():
    agent = Agent.from_env(builtin_tools=["time"])

    async for event in agent.stream_events("你好"):
        if event.type == "text_delta":
            print(event.data["content"], end="", flush=True)
        elif event.type == "done":
            print(f"\n\n[完成] 用时 {event.data['latency_ms']}ms")

asyncio.run(main())
```

## 运行

```bash
pip install wuwei-agent
export OPENAI_API_KEY="sk-..."
python minimal_example.py
```

> :bulb: 也支持在项目根目录创建 `.env` 文件，框架会自动加载。
