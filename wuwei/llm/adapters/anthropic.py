"""Anthropic 适配器 - 支持 Claude 系列模型"""

import json
from typing import Any

from .base import BaseAdapter
from ..types import Message, LLMResponse, LLMResponseChunk


class AnthropicAdapter(BaseAdapter):
    """Anthropic 适配器"""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "使用 Anthropic 适配器需要安装 anthropic 包：\n"
                "pip install wuwei[anthropic]"
            )

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build_request(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        stream: bool | None = False,
        **kwargs,
    ) -> dict:
        """构建 Anthropic API 请求"""
        # 分离系统消息
        system_msg = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        request = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": anthropic_messages,
        }

        if system_msg:
            request["system"] = system_msg

        if tools:
            # 转换工具格式
            request["tools"] = self._convert_tools(tools)

        return request

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """转换工具格式为 Anthropic 格式"""
        converted = []
        for tool in tools:
            if "function" in tool:
                converted.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {}),
                })
            else:
                converted.append(tool)
        return converted

    async def call(self, request: dict) -> Any:
        """调用 Anthropic API"""
        return await self.client.messages.create(**request)

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """解析 Anthropic 响应"""
        content = ""
        tool_calls = []

        for block in raw_response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        usage = None
        if hasattr(raw_response, "usage"):
            usage = {
                "prompt_tokens": raw_response.usage.input_tokens,
                "completion_tokens": raw_response.usage.output_tokens,
                "total_tokens": (
                    raw_response.usage.input_tokens
                    + raw_response.usage.output_tokens
                ),
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=raw_response.model,
            finish_reason=raw_response.stop_reason,
            usage=usage,
        )

    def parse_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """解析流式响应块"""
        if chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "text"):
                return {"type": "text", "content": chunk.delta.text}
        elif chunk.type == "content_block_start":
            if hasattr(chunk.content_block, "type"):
                if chunk.content_block.type == "tool_use":
                    return {
                        "type": "tool_start",
                        "id": chunk.content_block.id,
                        "name": chunk.content_block.name,
                    }
        return None
