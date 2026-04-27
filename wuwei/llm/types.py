from typing import Any, Literal

from pydantic import BaseModel, Field

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
    reasoning_content:str|None=None
    tool_calls:list[ToolCall]|None=None
    tool_call_id:str|None=None

class LLMResponse(BaseModel):
    message: Message
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]
    usage: dict[str, int]
    model: str
    latency_ms: int

class LLMResponseChunk(BaseModel):
    content: str
    reasoning_content: str | None = None
    tool_calls_delta: list[dict[str, Any]]|str = None  # 每个元素的格式：{"index": int, "id": str, "name": str, "arguments": str}
    tool_calls_complete: list[ToolCall]|None = None
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]|None = None
    usage: dict[str, int]|None = None


class AgentEvent(BaseModel):
    type: Literal["text_delta", "tool_start", "tool_end", "done", "error"]
    session_id: str
    step: int
    data: dict[str, Any] = Field(default_factory=dict)


class AgentRunResult(BaseModel):
    content: str
    usage: dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    latency_ms: int = 0
    llm_calls: int = 0
