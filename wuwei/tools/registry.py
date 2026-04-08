from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from wuwei.tools.tool import Tool, ToolParameters


class ToolRegistry:
    def __init__(self):
        self._tools:dict[str,Tool]={}

    def register(self,tool:Tool)->Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已经注册了")
        self._tools[tool.name]=tool
        return tool

    def unregister(self,tool:Tool):
        if tool.name not in self._tools:
            raise ValueError(f"工具 {tool.name} 没有注册")
        del self._tools[tool.name]

    def get(self,name:str)->Tool|None:
        return self._tools[name]

    def list(self)->list[Tool]:
        return list(self._tools.values())

    def to_schema(self)->list[dict[str,Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def tool(
            self,
            name:str,
            description:str,
            parameters:ToolParameters,
            required:list[str],
             ):
        def decorator(func:Callable[..., Any] | Callable[..., Awaitable[Any]]):
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    required=required,
                    handler=func
                )
            )
            return func
        return decorator
