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
        "深圳": {"city": "深圳", "condition": "rain", "temperature_c": 26},
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
        env_prefix="WUWEI",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature=0.2,
    )


async def main() -> None:
    agent = Agent(
        llm=build_llm(),
        tools=registry,
        default_system_prompt="你是一个会复用对话历史的中文助手。",
        default_max_steps=5,
    )

    session = agent.create_session(session_id="session-demo")

    first_question = "帮我查询一下上海的天气。"
    second_question = "我刚才问的是哪个城市？顺便再复述一下天气。"

    first_answer = await agent.run(first_question, session=session, stream=False)
    second_answer = await agent.run(second_question, session=session, stream=False)

    print("question 1:", first_question)
    print("answer 1:", first_answer.content)
    print("usage 1:", first_answer.usage)
    print()
    print("question 2:", second_question)
    print("answer 2:", second_answer.content)
    print("usage 2:", second_answer.usage)


if __name__ == "__main__":
    asyncio.run(main())
