import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing import get_type_hints


class ToolParameters(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list = Field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ToolRetryPolicy(BaseModel):
    max_attempts: int = 1
    backoff_seconds: float = 0.0

    def normalized_max_attempts(self) -> int:
        return max(1, int(self.max_attempts))


class ToolExecutionPolicy(BaseModel):
    capability: str | None = None
    timeout_seconds: float | None = None
    side_effect: bool = False
    requires_approval: bool = False
    retry_policy: ToolRetryPolicy = Field(default_factory=ToolRetryPolicy)


class Tool(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    handler: Callable[..., Any] | Callable[..., Awaitable[Any]]
    execution: ToolExecutionPolicy = Field(default_factory=ToolExecutionPolicy)
    display_name: str | None = None
    is_concurrency_safe: bool = True
    """是否可以在同一批次中与其他工具并发执行。

    设为 False 的工具会串行执行。
    读写文件的工具如果路径可能重叠，应设为 False。
    """

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_schema(),
            },
        }

    async def invoke(self, args: dict[str, Any] | None = None) -> Any:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        # Validate the public JSON contract before any handler side effects.
        from jsonschema import validate, ValidationError
        try:
            validate(args, self.parameters.to_schema())
        except ValidationError as exc:
            raise ValueError(f"Invalid tool arguments: {exc.message}") from exc
        inspect.signature(self.handler).bind(**args)
        hints = get_type_hints(self.handler)
        for key, value in args.items():
            if key in hints:
                TypeAdapter(hints[key]).validate_python(value, strict=True)
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**args)
        result = await asyncio.to_thread(self.handler, **args)
        if inspect.isawaitable(result):
            return await result
        return result
