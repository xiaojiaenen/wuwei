"""基础类型定义"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
from datetime import datetime


class Usage(BaseModel):
    """Token 使用量"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM 响应"""
    content: str = ""
    tool_calls: list = Field(default_factory=list)
    reasoning_content: Optional[str] = None
    usage: Optional[Usage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """是否有工具调用"""
        return len(self.tool_calls) > 0


class ToolResult(BaseModel):
    """工具执行结果"""
    content: str
    status: Literal["success", "error"] = "success"
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.content
