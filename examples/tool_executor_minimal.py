import asyncio

from wuwei.llm import FunctionCall, ToolCall
from wuwei.tools import ToolExecutor, ToolRegistry


registry = ToolRegistry()


@registry.tool(description="计算一个整数的平方。")
async def square(value: int) -> dict:
    return {"value": value, "square": value * value}


async def main() -> None:
    executor = ToolExecutor(registry)
    tool_call = ToolCall(
        id="call_square_1",
        type="function",
        function=FunctionCall(
            name="square",
            arguments={"value": 7},
        ),
    )

    result = await executor.execute_one(tool_call)
    print("tool name: square")
    print("tool result:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
