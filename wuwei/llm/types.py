from typing import Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

# 从 core.message 导入统一的消息类型（再导出，供外部统一从 wuwei.llm 导入）
from wuwei.core.message import BaseMessage as Message
from wuwei.core.message import ContentPart, FunctionCall, ToolCall

__all__ = [
    "Message",
    "FunctionCall",
    "ToolCall",
    "LLMResponse",
    "LLMResponseChunk",
    "AgentEvent",
    "AgentEventType",
    "AgentRunResult",
    "normalize_finish_reason",
]


from wuwei.core.types import LLMResponse, normalize_finish_reason


class ToolCallDelta(TypedDict, total=False):
    index: int
    id: str
    name: str
    arguments: str


class AdapterEvent(TypedDict, total=False):
    """Provider-normalized event. Usage fields are cumulative, never increments."""

    content: str
    reasoning_content: str
    tool_calls_delta: list[ToolCallDelta] | None
    finish_reason: str | None
    raw_finish_reason: str | None
    provider_metadata: dict[str, Any]
    usage: dict[str, int]


class LLMResponseChunk(BaseModel):
    content_parts: list[ContentPart] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str
    reasoning_content: str | None = None
    tool_calls_delta: list[ToolCallDelta] | None = (
        None  # 每个元素的格式：{"index": int, "id": str, "name": str, "arguments": str}
    )
    tool_calls_complete: list[ToolCall] | None = None
    finish_reason: str | None = None
    raw_finish_reason: str | None = None
    usage_available: bool = False
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
    content_parts: list[ContentPart] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage_available: bool = True
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
    status: Literal["completed", "interrupted", "error", "incomplete", "cancelled", "unknown"] = (
        "completed"
    )
    interrupt: dict[str, Any] | None = None
    termination_reason: str | None = None
    output: Any = None
