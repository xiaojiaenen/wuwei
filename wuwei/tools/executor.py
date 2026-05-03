import asyncio
import json
from typing import Any

from pydantic import BaseModel

from wuwei.llm.types import Message, ToolCall
from wuwei.tools.registry import ToolRegistry
from wuwei.tools.tool import Tool


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(
        self,
        tool_calls: list[ToolCall],
        concurrent: bool = False,
    ) -> list[Message]:
        if concurrent and len(tool_calls) > 1:
            return await asyncio.gather(*(self.execute_one(tool_call) for tool_call in tool_calls))

        results: list[Message] = []
        for tool_call in tool_calls:
            results.append(await self.execute_one(tool_call))
        return results

    async def execute_one(self, tool_call: ToolCall) -> Message:
        tool = self.registry.get(tool_call.function.name)
        if tool is None:
            return self.build_error_message(
                tool_call,
                error_type="ToolNotFound",
                message=f"Tool '{tool_call.function.name}' not found",
                metadata={"attempts": 0},
            )

        max_attempts = tool.execution.retry_policy.normalized_max_attempts()
        backoff_seconds = max(0.0, tool.execution.retry_policy.backoff_seconds)
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                output = await self._invoke_tool(tool, tool_call.function.arguments)
                content = self.serialize_output(output)
                return Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=content,
                )
            except Exception as exc:
                last_exception = exc
                if attempt >= max_attempts:
                    break
                if backoff_seconds:
                    await asyncio.sleep(backoff_seconds)

        assert last_exception is not None
        error_type = type(last_exception).__name__
        error_message = str(last_exception)
        if isinstance(last_exception, TimeoutError):
            error_type = "ToolTimeout"
            error_message = (
                f"Tool '{tool.name}' timed out after {tool.execution.timeout_seconds} seconds"
            )
        return self.build_error_message(
            tool_call,
            error_type=error_type,
            message=error_message,
            metadata={
                "attempts": max_attempts,
                "retryable": max_attempts > 1,
            },
        )

    async def _invoke_tool(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        timeout_seconds = tool.execution.timeout_seconds
        if timeout_seconds is None:
            return await tool.invoke(arguments)
        return await asyncio.wait_for(tool.invoke(arguments), timeout=timeout_seconds)

    def build_error_message(
        self,
        tool_call: ToolCall,
        *,
        error_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        if metadata:
            payload.update(metadata)
        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=json.dumps(payload, ensure_ascii=False),
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

    @staticmethod
    def extract_error_message(content: str | None) -> str | None:
        if not content:
            return None

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict) or payload.get("ok") is not False:
            return None

        error = payload.get("error")
        if not isinstance(error, dict):
            return None

        message = error.get("message")
        return message if isinstance(message, str) else None
