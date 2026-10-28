"""基础类型定义"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .message import BaseMessage


class Usage(BaseModel):
    """Token 使用量"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def normalize_finish_reason(reason: str | None) -> str:
    return {
        "eos": "stop",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "completed": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "max_output_tokens": "length",
        "aborted": "cancelled",
        "abort": "cancelled",
        "failed": "error",
        "incomplete": "incomplete",
    }.get(
        reason,
        reason
        if reason in {"stop", "tool_calls", "length", "content_filter", "cancelled", "error"}
        else "unknown",
    )


class LLMResponse(BaseModel):
    """Canonical response; legacy content/tool_calls constructors remain supported."""

    message: BaseMessage = Field(default_factory=lambda: BaseMessage(role="assistant"))
    finish_reason: str = "unknown"
    raw_finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    usage_available: bool = False
    model: str = ""
    latency_ms: int = 0
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def adapt_legacy(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        fields = {
            k: value.pop(k) for k in ("content", "tool_calls", "reasoning_content") if k in value
        }
        if fields:
            if "message" in value:
                raise ValueError("Use message or legacy content fields, not both")
            value["message"] = BaseMessage(role="assistant", **fields)
        if isinstance(value.get("usage"), BaseModel):
            value["usage"] = value["usage"].model_dump()
        if value.get("usage") is None:
            value["usage"] = {}
        if value.get("model") is None:
            value["model"] = ""
        if "usage_available" not in value:
            value["usage_available"] = bool(value.get("usage"))
        raw = value.get("raw_finish_reason", value.get("finish_reason"))
        value["raw_finish_reason"] = raw
        value["finish_reason"] = normalize_finish_reason(value.get("finish_reason"))
        return value

    @property
    def content(self):
        return self.message.content

    @property
    def tool_calls(self):
        return self.message.tool_calls

    @property
    def reasoning_content(self):
        return self.message.reasoning_content

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)


class ToolResult(BaseModel):
    """工具执行结果"""

    content: str
    status: Literal["success", "error"] = "success"
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.content
