import asyncio
import json

from wuwei.agent import PlanAgent
from wuwei.llm import LLMGateway
from wuwei.llm.types import ToolCall
from wuwei.tools import ToolExecutor, ToolRegistry


GOAL = """
必须使用现有两个工具完成这个目标，不要跳过工具、不要心算：
1. 查询北京、上海、广州、深圳四个城市的天气和气温。
2. 计算北京、上海、广州三城平均气温。
3. 计算四个城市里最高气温和最低气温的差值。
4. 根据天气、气温和计算结果，给出一个适合白天户外活动的城市排序与简短建议。
""".strip()


registry = ToolRegistry()


@registry.tool(description="查询一个城市的天气。")
async def get_weather(city: str) -> dict:
    weather_data = {
        "北京": {"city": "北京", "condition": "sunny", "temperature_c": 25},
        "上海": {"city": "上海", "condition": "cloudy", "temperature_c": 23},
        "广州": {"city": "广州", "condition": "clear", "temperature_c": 28},
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


@registry.tool(description="执行一个基础数学运算。")
async def calculate(a: float, b: float, operation: str) -> dict:
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"ok": False, "error": "division by zero"}
        result = a / b
    else:
        return {"ok": False, "error": f"unsupported operation: {operation}"}

    return {
        "ok": True,
        "a": a,
        "b": b,
        "operation": operation,
        "result": result,
    }


class LoggingToolExecutor(ToolExecutor):
    async def execute_one(self, tool_call: ToolCall):
        args_text = json.dumps(tool_call.function.arguments, ensure_ascii=False)
        print(f"\n[tool request] {tool_call.function.name} {args_text}")
        message = await super().execute_one(tool_call)
        error_message = self.extract_error_message(message.content)
        if error_message:
            print(f"[tool error] {error_message}")
        else:
            print(f"[tool result] {message.content}")
        return message


def build_llm() -> LLMGateway:
    return LLMGateway.from_env(
        env_prefix="WUWEI",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature=0.2,
    )


def print_plan(tasks) -> None:
    print("\n=== planned tasks ===")
    for task in sorted(tasks, key=lambda item: item.id):
        print(f"[Task {task.id}] {task.description}")
        print(f"status: {task.status}")
        print(f"next: {task.next}")
        print()


def print_task_results(tasks) -> None:
    print("\n=== task results ===")
    for task in sorted(tasks, key=lambda item: item.id):
        print(f"[Task {task.id}] {task.description}")
        print(f"status: {task.status}")
        print(f"next: {task.next}")
        if task.result:
            print(f"result: {task.result}")
        if task.error:
            print(f"error: {task.error}")
        print()


def print_detected_tool_calls(tool_calls: list[ToolCall]) -> None:
    print("\n[assistant decided to call tools]")
    for tool_call in tool_calls:
        args_text = json.dumps(tool_call.function.arguments, ensure_ascii=False)
        print(f"- {tool_call.function.name} {args_text}")


async def main() -> None:
    agent = PlanAgent(
        llm=build_llm(),
        tools=registry,
        default_system_prompt=(
            "你是一个会先规划、再执行任务的助手。"
            "需要工具时必须调用工具，不要猜测结果。"
        ),
        default_max_steps=6,
        default_parallel_tool_calls=True,
    )
    agent.tool_executor = LoggingToolExecutor(agent.tool_registry)

    tasks = await agent.plan(GOAL)

    print("=== fixed goal ===")
    print(GOAL)
    print_plan(tasks)
    print("\n=== streaming output ===")

    async for chunk in await agent.execute(GOAL, tasks, stream=True):
        if chunk.tool_calls_complete:
            print_detected_tool_calls(chunk.tool_calls_complete)
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()
    print_task_results(tasks)


if __name__ == "__main__":
    asyncio.run(main())
