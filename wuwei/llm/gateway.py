import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Union

from .adapters import OpenAIAdapter
from .adapters.base import BaseAdapter
from .types import FunctionCall, LLMResponse, LLMResponseChunk, Message, ToolCall
from ..tools import Tool


class LLMGateway:
    def __init__(self, config: dict[str, Any]):
        """根据显式配置初始化模型网关。"""
        self.adapter: BaseAdapter
        provider = config.get("provider", "openai")

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

    @classmethod
    def from_env(
        cls,
        provider: str = "openai",
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        model_env: str = "OPENAI_MODEL",
        default_base_url: str | None = None,
        default_model: str | None = None,
        **config: Any,
    ) -> "LLMGateway":
        """
        从环境变量构造 LLMGateway。

        设计原则：
        - 只帮用户读取最常见的 api_key / base_url / model
        - 其他参数例如 temperature / timeout 仍然走显式传参
        - 用户可以自定义环境变量名，避免框架写死业务约定
        """
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")

        env_config: dict[str, Any] = {
            "provider": provider,
            "api_key": api_key,
        }

        base_url = os.getenv(base_url_env) or default_base_url
        if base_url:
            env_config["base_url"] = base_url

        model = os.getenv(model_env) or default_model
        if model:
            env_config["model"] = model

        env_config.update(config)
        return cls(env_config)

    async def generate(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[LLMResponse, AsyncIterator[LLMResponseChunk]]:
        """统一处理单次和流式生成请求。"""
        if stream:
            return self._generate_stream(messages=messages, tools=tools, **kwargs)
        return await self._generate_single(messages=messages, tools=tools, **kwargs)

    async def _generate_single(self, messages: list[Message], tools: list[Tool] | None, **kwargs) -> LLMResponse:
        """发送一次非流式请求。"""
        request = self.adapter.build_request(messages=messages, tools=tools, stream=False, **kwargs)
        start = time.monotonic()
        last_exception = None

        for attempt in range(self.retry_policy["max_attempts"]):
            try:
                raw = await asyncio.wait_for(self.adapter.call(request), timeout=self.timeout)
                response = self.adapter.parse_response(raw)
                response.latency_ms = int((time.monotonic() - start) * 1000)
                return response
            except Exception as exc:
                last_exception = exc
                if attempt == self.retry_policy["max_attempts"] - 1:
                    raise
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

        raise last_exception

    async def _generate_stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        """发送一次流式请求，并把 tool call 增量拼成完整结构。"""
        request = self.adapter.build_request(messages, tools, stream=True, **kwargs)
        stream = await self.adapter.call(request)

        # 按 index 累积工具调用增量。
        pending: dict[int, dict[str, Any]] = {}

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

            out_chunk = LLMResponseChunk(content=content)

            if finish_reason == "tool_calls":
                complete = []
                for idx, item in pending.items():
                    if not item["id"] or not item["name"]:
                        continue
                    try:
                        args = json.loads(item["arguments"]) if item["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    complete.append(
                        ToolCall(
                            id=item["id"],
                            type="function",
                            function=FunctionCall(name=item["name"], arguments=args),
                        )
                    )
                out_chunk.tool_calls_complete = complete
                out_chunk.finish_reason = finish_reason
            elif finish_reason == "stop":
                out_chunk.finish_reason = finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                out_chunk.usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            yield out_chunk
