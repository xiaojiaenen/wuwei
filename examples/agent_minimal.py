import asyncio

from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.tools import ToolRegistry


registry = ToolRegistry()


@registry.tool(description="查询一个城市的天气。")
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


def build_llm() -> LLMGateway:
    return LLMGateway.from_env(
        provider="openai",
        api_key_env="WUWEI_API_KEY",
        base_url_env="WUWEI_BASE_URL",
        model_env="WUWEI_MODEL",
        default_model="deepseek-chat",
        default_base_url="https://api.deepseek.com",
        temperature=0.2,
    )


async def main() -> None:
    agent = Agent(
        llm=build_llm(),
        tools=registry,
        default_system_prompt="你是一个会优先调用工具获取天气信息的助手。",
        default_max_steps=5,
    )

    session = agent.create_session(session_id="agent-minimal")
    question = "请先调用工具，再告诉我北京天气怎么样。"

    print("question:", question)
    print("answer:")
    async for chunk in await agent.run(question, session=session, stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
