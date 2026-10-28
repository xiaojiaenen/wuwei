"""全流程示例：网关 + 自定义工具 + 中间件 + 流式事件 + 会话统计

演示 v3.1 的核心能力：
- LLMGateway（openai / responses / anthropic 三个 provider，openai 兼容任意端点）
- 单一执行循环：run() 与 stream_events() 行为一致
- 工具结果回写会话、usage 逐步累计
- run() 默认复用会话 → 多轮对话

    python examples/full_flow.py
"""

from __future__ import annotations

import asyncio

from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.middleware import LoggingMiddleware, MiddlewareStack
from wuwei.tools import Tool, ToolParameters, ToolRegistry


def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询城市天气（演示用假数据）。

    :param city: 城市名
    :param unit: 温度单位 celsius / fahrenheit
    """
    return {"city": city, "temp": 22, "unit": unit, "condition": "晴"}


async def main() -> None:
    llm = LLMGateway.from_env()  # 读 OPENAI_* / ANTHROPIC_* 环境变量或 .env

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_weather",
            description="查询指定城市的当前天气",
            parameters=ToolParameters(),
            handler=get_weather,
            display_name="天气查询",
        )
    )

    middleware = MiddlewareStack()
    middleware.add(LoggingMiddleware())

    agent = Agent(
        llm=llm,
        tools=registry,
        load_builtins="minimal",
        default_system_prompt="你是一个乐于助人的助手，回答简洁。",
        default_max_steps=6,
        middleware=middleware,
    )

    # ── 1. 非流式 + 工具调用 ─────────────────────────────────────
    result = await agent.run("北京今天天气怎么样？")
    print("回答:", result.content)
    print("用量:", result.usage, "LLM 调用次数:", result.llm_calls)

    # ── 2. 流式事件（多轮：默认复用上一次会话） ───────────────────
    print("\n--- 流式多轮 ---")
    async for event in agent.stream_events("那上海呢？和北京比哪个更热？"):
        if event.type == "text_delta":
            print(event.data["content"], end="", flush=True)
        elif event.type == "tool_start":
            print(f"\n[调用工具 {event.data['tool_name']} {event.data['args']}]", flush=True)
        elif event.type == "tool_end":
            print(f"\n[工具完成]", flush=True)
    print()

    print("\n会话累计用量:", agent._default_session.last_usage)


if __name__ == "__main__":
    asyncio.run(main())
