import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter
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
    idempotent: bool = False
    """重复执行是否安全。

    工具超时后无法确认它是否已经生效：声明 side_effect=True 且未声明幂等的工具
    会被标为 unknown，要求人工对账后才能继续；声明幂等则标为失败并允许重试。
    """
    retry_policy: ToolRetryPolicy = Field(default_factory=ToolRetryPolicy)


class Tool(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    handler: Callable[..., Any] | Callable[..., Awaitable[Any]]
    execution: ToolExecutionPolicy = Field(default_factory=ToolExecutionPolicy)
    display_name: str | None = None
    is_concurrency_safe: bool | None = None
    """是否可以在同一批次中与其他工具并发执行。

    None 表示按 side_effect 推导：无副作用工具并发，有副作用工具串行。
    读写文件的工具如果路径可能重叠，应显式设为 False。
    """

    # 校验缓存：编译一次 jsonschema validator、反射一次 handler，之后每次
    # invoke 直接复用（此前每次调用重建 schema dict / get_type_hints /
    # signature / TypeAdapter，共四份重复工作）。
    _validator: Any = PrivateAttr(default=None)
    _signature: Any = PrivateAttr(default=None)
    _injected: frozenset = PrivateAttr(default=None)
    _adapters: dict = PrivateAttr(default=None)

    def _validation_cache(self) -> dict:
        if self._validator is None:
            from typing import get_origin

            from wuwei.runtime.contracts import RunContext

            schema = self.parameters.to_schema()
            self._validator = validator_for(schema)(schema)
            self._signature = inspect.signature(self.handler)
            hints = get_type_hints(self.handler)
            self._injected = frozenset(
                key
                for key, hint in hints.items()
                if hint is RunContext or get_origin(hint) is RunContext
            )
            self._adapters = {
                key: TypeAdapter(hints[key])
                for key in hints
                if key not in self._injected
            }
        return {
            "validator": self._validator,
            "signature": self._signature,
            "injected": self._injected,
            "adapters": self._adapters,
        }

    @property
    def concurrency_safe(self) -> bool:
        if self.is_concurrency_safe is not None:
            return self.is_concurrency_safe
        return not self.execution.side_effect

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
        cache = self._validation_cache()
        # Validate the public JSON contract before any handler side effects.
        try:
            cache["validator"].validate(args)
        except ValidationError as exc:
            raise ValueError(f"Invalid tool arguments: {exc.message}") from exc
        from wuwei.runtime.context import current_context

        injected = cache["injected"]
        if injected.intersection(args):
            raise ValueError("Injected context cannot be supplied by model arguments")
        args = dict(args)
        for key in injected:
            active = current_context.get()
            context = active.config.get("run_context") if active else None
            if context is None:
                raise ValueError("Tool requires an active RunContext")
            args[key] = context
        cache["signature"].bind(**args)
        adapters = cache["adapters"]
        for key, value in args.items():
            adapter = adapters.get(key)
            if adapter is not None:
                adapter.validate_python(value, strict=True)
        try:
            if inspect.iscoroutinefunction(self.handler):
                return await self.handler(**args)
            result = await asyncio.to_thread(self.handler, **args)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:
            exc._wuwei_tool_executed = True
            raise
