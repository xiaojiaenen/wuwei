from typing import Literal, Any

from pydantic import BaseModel

class FunctionCall(BaseModel):
    name:str
    arguments:dict[str, Any]

class ToolCall(BaseModel):
    id:str
    type:Literal["function"]
    function:FunctionCall

class Message(BaseModel):
    role:Literal["system", "user", "assistant", "tool"]
    content:str|None=None
    tool_calls:list[ToolCall]|None=None
    tool_call_id:str|None=None
    name:str|None=None

class LLMResponse(BaseModel):
    message: Message
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]
    usage: dict[str, int]
    model: str
    latency_ms: int

class LLMResponseChunk(BaseModel):
    content: str
    tool_calls_delta: dict[str, Any]|None = None   # 流式工具调用的增量
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]|None = None
    usage: dict[str, int]|None = None