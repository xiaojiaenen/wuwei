from typing import Any, Literal

from pydantic import BaseModel, Field

# 从 core.message 导入统一的消息类型
from wuwei.core.message import BaseMessage as Message, FunctionCall, ToolCall


class LLMResponse(BaseModel):
    message: Message
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]
    usage: dict[str, int]
    model: str
    latency_ms: int


class LLMResponseChunk(BaseModel):
    content: str
    reasoning_content: str | None = None
    tool_calls_delta: list[dict[str, Any]] | str = (
        None  # 每个元素的格式：{"index": int, "id": str, "name": str, "arguments": str}
    )
    tool_calls_complete: list[ToolCall] | None = None
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"] | None = None
    usage: dict[str, int] | None = None


AgentEventType = Literal[
    "run_start",
    "run_end",
    "llm_start",
    "llm_end",
    "text_delta",
    "reasoning_delta",
    "tool_start",
    "tool_end",
    "tool_error",
    "task_start",
    "task_end",
    "approval_required",
    "done",
    "error",
]


class AgentEvent(BaseModel):
    type: AgentEventType
    session_id: str
    step: int
    run_id: str | None = None
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
