import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolParameters(BaseModel):
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list = Field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
        }


class ToolRetryPolicy(BaseModel):
    max_attempts: int = 1
    backoff_seconds: float = 0.0

    def normalized_max_attempts(self) -> int:
        return max(1, int(self.max_attempts))


class ToolExecutionPolicy(BaseModel):
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
        args = args or {}
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**args)
        result = self.handler(**args)
        if inspect.isawaitable(result):
            return await result
        return result
