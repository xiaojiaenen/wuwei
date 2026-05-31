"""Anthropic 适配器 - 支持 Claude 系列模型"""

import json
from typing import Any

from .base import BaseAdapter
from ..types import Message, LLMResponse, LLMResponseChunk, ToolCall, FunctionCall
from ...tools import Tool


class AnthropicAdapter(BaseAdapter):
    """Anthropic 适配器 — 支持 Claude 系列模型 (Claude 3/4)"""

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
                "pip install anthropic"
            )

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build_request(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool | None = False,
        **kwargs,
    ) -> dict:
        """构建 Anthropic API 请求"""
        system_msg = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                # 添加历史 tool_use blocks
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                        })
                # tool_result blocks
                if msg.role == "tool" and msg.tool_call_id:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }],
                    })
                    continue

                anthropic_messages.append({
                    "role": msg.role,
                    "content": content if content else msg.content or "",
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
            request["tools"] = self._convert_tools(tools)

        return request

    def _convert_tools(self, tools: list) -> list[dict]:
        """转换工具格式为 Anthropic 格式

        支持 Tool 对象和 dict schema 两种输入。
        """
        converted = []
        for tool in tools:
            if hasattr(tool, 'to_schema'):
                # Tool 对象
                schema = tool.to_schema()
                func = schema.get("function", schema)
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}, "required": []}),
                })
            elif isinstance(tool, dict) and "function" in tool:
                # OpenAI 格式 dict
                converted.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}, "required": []}),
                })
            elif isinstance(tool, dict) and "name" in tool:
                # 已经是 Anthropic 格式
                converted.append(tool)
        return converted

    async def call(self, request: dict) -> Any:
        """调用 Anthropic API"""
        return await self.client.messages.create(**request)

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """解析 Anthropic 响应 — 正确返回 LLMResponse(message=Message(...))"""
        content_parts = []
        tool_calls = []

        for block in raw_response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                try:
                    args = json.loads(block.input) if isinstance(block.input, str) else block.input
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(name=block.name, arguments=args),
                    )
                )

        usage = {}
        if hasattr(raw_response, "usage"):
            usage = {
                "prompt_tokens": raw_response.usage.input_tokens,
                "completion_tokens": raw_response.usage.output_tokens,
                "total_tokens": raw_response.usage.input_tokens + raw_response.usage.output_tokens,
            }

        internal_msg = Message(
            role="assistant",
            content="".join(content_parts) or "",
            tool_calls=tool_calls,
        )

        return LLMResponse(
            message=internal_msg,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
            model=raw_response.model,
            latency_ms=0,
        )

    def parse_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """解析流式响应块 — 返回 gateway 兼容格式

        返回格式：{"content": str, "tool_calls_delta": [...], "finish_reason": str|None}
        """
        result = {"content": "", "tool_calls_delta": None, "finish_reason": None}

        if chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "text"):
                result["content"] = chunk.delta.text
            elif hasattr(chunk.delta, "partial_json"):
                result["content"] = chunk.delta.partial_json

        elif chunk.type == "content_block_start":
            if hasattr(chunk.content_block, "type") and chunk.content_block.type == "tool_use":
                result["tool_calls_delta"] = [{
                    "index": getattr(chunk, "index", 0),
                    "id": chunk.content_block.id,
                    "name": chunk.content_block.name,
                    "arguments": "",
                }]

        elif chunk.type == "message_delta":
            if hasattr(chunk.delta, "stop_reason") and chunk.delta.stop_reason == "tool_use":
                result["finish_reason"] = "tool_calls"
            elif hasattr(chunk.delta, "stop_reason") and chunk.delta.stop_reason == "end_turn":
                result["finish_reason"] = "stop"
            # usage info
            if hasattr(chunk, "usage"):
                result["usage"] = {
                    "prompt_tokens": chunk.usage.input_tokens,
                    "completion_tokens": chunk.usage.output_tokens,
                    "total_tokens": chunk.usage.input_tokens + chunk.usage.output_tokens,
                }

        return result if (result["content"] or result["tool_calls_delta"] or result["finish_reason"]) else None
