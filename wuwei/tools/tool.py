import inspect
from typing import Callable, Any, Awaitable

from pydantic import BaseModel

class ToolParameters(BaseModel):
    type:str="object"
    properties:dict[str, Any]={}
    required:list=[]

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
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**args)
        result=await self.handler(**args)
        if inspect.isawaitable(result):
            return await result
        return result