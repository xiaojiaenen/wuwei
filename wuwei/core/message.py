"""消息类型 - 统一的消息体系

借鉴 LangChain 的 BaseMessage，支持 OpenAI/Anthropic 格式互转。
"""

import json
import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from wuwei.artifacts import ArtifactRef


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    url: str | None = None
    artifact: ArtifactRef | None = None
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def source_is_unambiguous(self):
        if sum(value is not None for value in (self.url, self.artifact)) != 1:
            raise ValueError("Image requires exactly one URL or artifact")
        if self.artifact and not self.artifact.media_type.startswith("image/"):
            raise ValueError("Image artifact must have an image media type")
        return self


class FilePart(BaseModel):
    type: Literal["file"] = "file"
    file_id: str | None = None
    url: str | None = None
    artifact: ArtifactRef | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def source_is_unambiguous(self):
        if sum(value is not None for value in (self.file_id, self.url, self.artifact)) != 1:
            raise ValueError("File requires exactly one file ID, URL or artifact")
        return self


class CitationPart(BaseModel):
    type: Literal["citation"] = "citation"
    url: str | None = None
    title: str | None = None
    text: str | None = None
    provider_data: dict = Field(default_factory=dict)


ContentPart = Annotated[TextPart | ImagePart | FilePart | CitationPart, Field(discriminator="type")]


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
    content_parts: list[ContentPart] = Field(default_factory=list)
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)  # 所有消息都有 tool_calls
    tool_call_id: str | None = None
    name: str | None = None  # 工具消息的工具名
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

    def provider_content(self, provider: str):
        if not self.content_parts:
            return self.content
        blocks = (
            [{"type": "text" if provider != "responses" else "input_text", "text": self.content}]
            if self.content and not any(isinstance(p, TextPart) for p in self.content_parts)
            else []
        )
        for part in self.content_parts:
            if isinstance(part, CitationPart):
                continue  # Output annotations are retained, not sent as unsupported input blocks.
            if isinstance(part, TextPart):
                blocks.append(
                    {"type": "input_text" if provider == "responses" else "text", "text": part.text}
                )
            elif isinstance(part, ImagePart):
                if not part.url or part.artifact:
                    raise ValueError(
                        "Resolve artifact image to an authorized provider URL before sending"
                    )
                if provider == "anthropic":
                    source = {"type": "url", "url": part.url}
                    if part.url.startswith("data:"):
                        header, data = part.url.split(",", 1)
                        source = {
                            "type": "base64",
                            "media_type": header[5:].split(";")[0],
                            "data": data,
                        }
                    blocks.append({"type": "image", "source": source})
                elif provider == "responses":
                    blocks.append(
                        {"type": "input_image", "image_url": part.url, "detail": part.detail}
                    )
                else:
                    blocks.append(
                        {"type": "image_url", "image_url": {"url": part.url, "detail": part.detail}}
                    )
            elif isinstance(part, FilePart):
                if part.artifact:
                    raise ValueError(
                        "Resolve artifact file to an authorized provider reference before sending"
                    )
                if provider == "responses" and (part.file_id or part.url):
                    blocks.append(
                        {
                            "type": "input_file",
                            **(
                                {"file_id": part.file_id}
                                if part.file_id
                                else (
                                    {"file_data": part.url, "filename": part.filename or "artifact"}
                                    if part.url.startswith("data:")
                                    else {"file_url": part.url}
                                )
                            ),
                        }
                    )
                elif provider == "openai" and (
                    part.file_id or (part.url and part.url.startswith("data:"))
                ):
                    blocks.append(
                        {
                            "type": "file",
                            "file": {"file_id": part.file_id}
                            if part.file_id
                            else {"file_data": part.url, "filename": part.filename or "artifact"},
                        }
                    )
                elif provider == "anthropic" and part.url:
                    source = {"type": "url", "url": part.url}
                    if part.url.startswith("data:"):
                        header, data = part.url.split(",", 1)
                        source = {
                            "type": "base64",
                            "media_type": header[5:].split(";")[0],
                            "data": data,
                        }
                    blocks.append({"type": "document", "source": source})
                else:
                    raise ValueError("Unsupported provider file reference")
        return blocks

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        result = {"role": self.role, "content": self.provider_content("openai")}
        if self.tool_calls:
            result["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.role == "tool" and self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        return {"role": self.role, "content": self.provider_content("anthropic")}


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
        value = self.provider_content("anthropic")
        if isinstance(value, list):
            content.extend(value)
        elif value:
            content.append({"type": "text", "text": value})
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
            "content": self.provider_content("openai"),
            "tool_call_id": self.tool_call_id,
        }

    def to_anthropic(self) -> dict:
        """转换为 Anthropic 格式"""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.provider_content("anthropic"),
        }
