"""消息类型 - 统一的消息体系

借鉴 LangChain 的 BaseMessage，支持 OpenAI/Anthropic 格式互转。
"""

import json
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """函数调用"""

    name: str
    arguments: dict = Field(default_factory=dict)

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        return {
            "name": self.name,
            "arguments": json.dumps(self.arguments),
        }

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        return {
            "name": self.name,
            "input": self.arguments,
        }


class ToolCall(BaseModel):
    """工具调用"""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    type: Literal["function"] = "function"
    function: FunctionCall

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_openai(),
        }

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.function.name,
            "input": self.function.arguments,
        }


class BaseMessage(BaseModel):
    """消息基类"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)  # 所有消息都有 tool_calls
    tool_call_id: str | None = None
    name: str | None = None  # 工具消息的工具名
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.role == "tool" and self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        return {"role": self.role, "content": self.content}


class SystemMessage(BaseMessage):
    """系统消息"""

    role: Literal["system"] = "system"

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class HumanMessage(BaseMessage):
    """用户消息"""

    role: Literal["user"] = "user"

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class AIMessage(BaseMessage):
    """AI 消息"""

    role: Literal["assistant"] = "assistant"
    reasoning_content: str | None = None
    usage: dict | None = None

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        result = super().to_openai()
        if self.tool_calls:
            result["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        content = []
        if self.content:
            content.append({"type": "text", "text": self.content})
        for tc in self.tool_calls:
            content.append(tc.to_anthropic())
        return {"role": "assistant", "content": content}


class ToolMessage(BaseMessage):
    """工具消息"""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    status: Literal["success", "error"] = "success"

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.content,
        }
