import json
from typing import Any

from openai import AsyncOpenAI

from .base import BaseAdapter
from ..types import LLMResponse, Message, ToolCall, FunctionCall
from ...tools import Tool


class OpenAIAdapter(BaseAdapter):
    def __init__(self,api_key:str,model:str|None="gpt-5.4",base_url:str|None="https://api.openai.com/v1",**kwargs):
        self.client=AsyncOpenAI(api_key=api_key,base_url=base_url)
        self.model = model
        self.default_params = kwargs
    def build_request(self, messages: list[Message], tools: list[Tool] | None = None, stream: bool | None = False,
                      **kwargs) -> dict[str, Any]:
        openai_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"]=[
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": json.dumps(tc.function.arguments)
                        }
                    }
                    for tc in msg.tool_calls
                ]

            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            openai_messages.append(m)

        request = {
            "model": self.model,
            "messages": openai_messages,
            "stream": stream,
            **self.default_params,
            **kwargs,
        }
        if tools:
            request["tools"] = [tool.to_schema() for tool in tools]

        return request

    async def call(self, request: dict[str, Any]) -> Any:
        return await self.client.chat.completions.create(**request)

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """解析非流式响应"""
        choice = raw_response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                )
                for tc in message.tool_calls
            ]

        internal_msg = Message(
            role="assistant",
            content=message.content,
            tool_calls=tool_calls,
        )

        return LLMResponse(
            message=internal_msg,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens,
            },
            model=raw_response.model,
            latency_ms=0,  # 由网关填充
        )

    def parse_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        if not chunk.choices:
            return None
        delta = chunk.choices[0].delta
        result = {
            "content": delta.content or "",
            "finish_reason": chunk.choices[0].finish_reason,
        }
        if delta.tool_calls:
            result["tool_calls_delta"] = []
            for tc in delta.tool_calls:
                item = {"index": tc.index}
                if tc.id:
                    item["id"] = tc.id
                if tc.function.name:
                    item["name"] = tc.function.name
                if tc.function.arguments:
                    item["arguments"] = tc.function.arguments
                result["tool_calls_delta"].append(item)
        return result