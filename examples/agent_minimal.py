import asyncio

from wuwei.agent import Agent
from wuwei.runtime import ConsoleHook

async def get_weather(city: str) -> dict:
    weather_data = {
        "北京": {"city": "北京", "condition": "sunny", "temperature_c": 25},
        "上海": {"city": "上海", "condition": "cloudy", "temperature_c": 23},
        "广州": {"city": "广州", "condition": "clear", "temperature_c": 28},
    }
    return weather_data.get(
        city,
        {
            "city": city,
            "condition": "unknown",
            "temperature_c": None,
            "error": f"weather for {city} was not found",
        },
    )


async def main() -> None:
    agent = Agent.from_env(
        env_prefix="WUWEI",
        builtin_tools=["file", "time"],
        tools=[get_weather],
        system_prompt="你是一个会优先调用工具获取天气信息的助手。",
        max_steps=5,
        hooks=[ConsoleHook()],
    )

    session = agent.create_session(session_id="agent-minimal")
    question = "请告诉我当前时间，再告诉我北京天气怎么样。"

    print("question:", question)
    print("answer:")
    async for chunk in await agent.run(question, session=session, stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
