import inspect
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints

from wuwei.tools.tool import Tool, ToolExecutionPolicy, ToolParameters, ToolRetryPolicy


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已经注册了")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, tool: Tool):
        if tool.name not in self._tools:
            raise ValueError(f"工具 {tool.name} 没有注册")
        del self._tools[tool.name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_schema(self) -> list[Tool]:
        return [tool.to_schema() for tool in self._tools.values()]

    def register_callable(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: ToolParameters | None = None,
        execution: ToolExecutionPolicy | None = None,
        timeout_seconds: float | None = None,
        side_effect: bool = False,
        requires_approval: bool = False,
        retry_policy: ToolRetryPolicy | None = None,
        display_name: str | None = None,
    ) -> Tool:
        tool_name = name or func.__name__
        self.tool(
            name=tool_name,
            description=description,
            parameters=parameters,
            execution=execution,
            timeout_seconds=timeout_seconds,
            side_effect=side_effect,
            requires_approval=requires_approval,
            retry_policy=retry_policy,
            display_name=display_name,
        )(func)

        tool = self.get(tool_name)
        if tool is None:
            raise ValueError(f"工具 {tool_name} 注册失败")
        return tool

    def tool(
        self,
        name: str = None,
        description: str = None,
        parameters: ToolParameters = None,
        execution: ToolExecutionPolicy | None = None,
        timeout_seconds: float | None = None,
        side_effect: bool = False,
        requires_approval: bool = False,
        retry_policy: ToolRetryPolicy | None = None,
        display_name: str | None = None,
    ):
        def decorator(func: Callable[..., Any] | Callable[..., Awaitable[Any]]):
            tool_name = name or func.__name__

            # 自动生成工具描述
            tool_description = description or func.__doc__ or f"执行{func.__name__}操作"
            if parameters:
                tool_parameters = parameters
            else:
                from pydantic import create_model
                sig = inspect.signature(func)
                hints = get_type_hints(func)
                fields = {}
                for key, param in sig.parameters.items():
                    if key == "self" or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                        continue
                    fields[key] = (hints.get(key, Any), ... if param.default is inspect.Parameter.empty else param.default)
                schema = create_model(f"{tool_name}Arguments", **fields).model_json_schema()
                schema.pop("title", None)
                tool_parameters = ToolParameters.model_validate(schema)
            self.register(
                Tool(
                    name=tool_name,
                    description=tool_description,
                    parameters=tool_parameters,
                    handler=func,
                    execution=execution
                    or ToolExecutionPolicy(
                        timeout_seconds=timeout_seconds,
                        side_effect=side_effect,
                        requires_approval=requires_approval,
                        retry_policy=retry_policy or ToolRetryPolicy(),
                    ),
                    display_name=display_name,
                )
            )
            return func

        return decorator
