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
                sig = inspect.signature(func)
                properties = {}
                required = []

                # 获取类型注解
                type_hints = get_type_hints(func)

                for param_name, param in sig.parameters.items():
                    # 跳过self参数
                    if param_name == "self":
                        continue

                    # 推断参数类型
                    raw_type = type_hints.get(param_name, str)
                    # 处理 Union 类型（如 str | None），取第一个非 NoneType 的类型
                    if hasattr(raw_type, "__args__"):
                        # typing.Union 或 types.UnionType (Python 3.10+ X | Y)
                        args = [a for a in raw_type.__args__ if a is not type(None)]
                        raw_type = args[0] if args else str
                    param_type = getattr(raw_type, "__name__", "string")
                    if param_type in ["int", "float", "complex"]:
                        param_type = "number"
                    elif param_type in {"bool"}:
                        param_type = "boolean"
                    elif param_type in ["list", "tuple"]:
                        param_type = "array"
                    elif param_type in ["dict"]:
                        param_type = "object"
                    else:
                        param_type = "string"

                    # 获取参数描述（从docstring中提取）
                    param_description = f"参数 {param_name}"
                    if func.__doc__:
                        # 简单的docstring解析，提取参数描述
                        for line in func.__doc__.split("\n"):
                            line = line.strip()
                            if line.startswith(f":param {param_name}:"):
                                param_description = line.split(":", 2)[2].strip()
                                break

                    properties[param_name] = {"type": param_type, "description": param_description}

                    # 非可选参数
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

                tool_parameters = ToolParameters(properties=properties, required=required)
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
