"""Ollama 适配器 - 支持本地模型

Ollama 使用 OpenAI 兼容格式。
"""

from typing import Any

from .base import BaseAdapter
from ..types import Message, LLMResponse, LLMResponseChunk


class OllamaAdapter(BaseAdapter):
    """Ollama 适配器"""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "使用 Ollama 适配器需要安装 openai 包：\n"
                "pip install openai"
            )

        self.client = AsyncOpenAI(
            api_key="ollama",  # Ollama 不需要真实的 API key
            base_url=f"{base_url}/v1",
        )
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
        """构建 Ollama API 请求（OpenAI 兼容格式）"""
        request = {
            "model": self.model,
            "messages": [msg.to_openai() for msg in messages],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if tools:
            request["tools"] = tools

        return request

    async def call(self, request: dict) -> Any:
        """调用 Ollama API"""
        return await self.client.chat.completions.create(**request)

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """解析 Ollama 响应（OpenAI 兼容格式）"""
        choice = raw_response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        usage = None
        if raw_response.usage:
            usage = {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            model=raw_response.model or self.model,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    def parse_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """解析流式响应块"""
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                return {"type": "text", "content": delta.content}
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function:
                        return {
                            "type": "tool_call",
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
        return None
