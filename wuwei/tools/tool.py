import inspect
from typing import Callable, Any, Awaitable

from pydantic import BaseModel, Field


class ToolParameters(BaseModel):
    type:str="object"
    properties:dict[str, Any]=Field(default_factory=dict)
    required:list=Field(default_factory=list)

    def to_schema(self)->dict[str, Any]:
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
        }

class Tool(BaseModel):
    name:str
    description:str
    parameters:ToolParameters
    handler:Callable[..., Any] | Callable[..., Awaitable[Any]]

    def to_schema(self)->dict[str, Any]:
        return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_schema(),
            }
        }

    async def invoke(self,args:dict[str, Any]|None=None)->Any:
        args=args or {}
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**args)
        result=self.handler(**args)
        if inspect.isawaitable(result):
            return await result
        return result