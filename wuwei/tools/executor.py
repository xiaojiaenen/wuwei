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
                content=json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "type": "ToolNotFound",
                            "message": f"Tool '{tool_call.function.name}' not found",
                        },
                    },
                    ensure_ascii=False,
                ),
            )

        try:
            output = await tool.invoke(tool_call.function.arguments)
            content = self.serialize_output(output)
        except Exception as e:
            content = json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                },
                ensure_ascii=False,
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