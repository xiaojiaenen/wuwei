import asyncio
import json
import time
from typing import Any, AsyncIterator, Union

from .adapters import OpenAIAdapter
from .adapters.base import BaseAdapter
from .types import Message, LLMResponse, LLMResponseChunk, FunctionCall, ToolCall
from ..tools import Tool


class LLMGateway:
    def __init__(self,config:dict[str,Any]):
        self.adapter:BaseAdapter
        provider = config.get("provider", "openai")
        self.adapter: BaseAdapter
        if provider == "openai":
            adapter_kwargs = {
                "api_key": config["api_key"],
                "model": config.get("model", "gpt-5.4"),
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 4096),
            }
            if config.get("base_url"):
                adapter_kwargs["base_url"] = config["base_url"]

            self.adapter = OpenAIAdapter(**adapter_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.retry_policy = config.get("retry", {"max_attempts": 3})
        self.timeout = config.get("timeout", 60)

    async def generate(self,
                       messages: list[Message],
                       tools: list[Tool]|None=None,
                       stream:bool=False,
                       **kwargs
                       )->Union[LLMResponse, AsyncIterator[LLMResponseChunk]]:
        if stream:
            return self._generate_stream(messages=messages,tools=tools,**kwargs)
        else:
            return await self._generate_single(messages=messages,tools=tools,**kwargs)

    async def _generate_single(self,messages:list[Message],tools:list[Tool],**kwargs)->LLMResponse:
        request=self.adapter.build_request(messages=messages,tools=tools,stream=False,**kwargs)
        start=time.monotonic()
        last_exception = None
        for attempt in range(self.retry_policy["max_attempts"]):
            try:
                raw=await asyncio.wait_for(self.adapter.call(request), timeout=self.timeout)
                response=self.adapter.parse_response(raw)
                response.latency_ms = int((time.monotonic() - start) * 1000)
                return response
            except Exception as e:
                last_exception=e
                if attempt == self.retry_policy["max_attempts"] - 1:
                    raise
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        raise last_exception

    async def _generate_stream(self,messages:list[Message],tools:list[Tool],**kwargs)->AsyncIterator[LLMResponseChunk]:
        request = self.adapter.build_request(messages, tools, stream=True, **kwargs)
        stream = await self.adapter.call(request)

        # 按 index 累积工具调用数据
        pending: dict[int, dict[str, Any]] = {}  # index -> {"id": str, "name": str, "arguments": str}

        async for chunk in stream:
            data = self.adapter.parse_stream_chunk(chunk)
            if not data:
                continue

            content = data.get("content", "")
            finish_reason = data.get("finish_reason")
            tool_calls_delta = data.get("tool_calls_delta")

            if tool_calls_delta:
                for delta_item in tool_calls_delta:
                    idx = delta_item["index"]
                    if idx not in pending:
                        pending[idx] = {"id": "", "name": "", "arguments": ""}
                    if "id" in delta_item:
                        pending[idx]["id"] = delta_item["id"]
                    if "name" in delta_item:
                        pending[idx]["name"] = delta_item["name"]
                    if "arguments" in delta_item:
                        pending[idx]["arguments"] += delta_item["arguments"]
            # print(f"pending: {pending}")
            # 构建输出 chunk
            out_chunk = LLMResponseChunk(content=content)

            if finish_reason == "tool_calls":
                # 组装完整的 tool_calls
                complete = []
                for idx, data in pending.items():
                    if not data["id"] or not data["name"]:
                        continue  # 不完整，理论上不会发生
                    try:
                        args = json.loads(data["arguments"]) if data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    complete.append(ToolCall(
                        id=data["id"],
                        type="function",
                        function=FunctionCall(name=data["name"], arguments=args)
                    ))
                out_chunk.tool_calls_complete = complete
                out_chunk.finish_reason = finish_reason
            elif finish_reason == "stop":
                out_chunk.finish_reason = finish_reason

            # 可选：在最后一个 chunk 中附上 usage
            if hasattr(chunk, "usage") and chunk.usage:
                out_chunk.usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            # print(f"out_chunk: {out_chunk}")
            yield out_chunk