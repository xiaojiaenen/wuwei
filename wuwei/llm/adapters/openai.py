"""OpenAI Chat Completions 适配器

同时兼容所有 OpenAI 协议端点（DeepSeek / vLLM / Ollama / 各类聚合网关），
只需在构造时传入对应的 base_url。
"""

import json
from typing import Any

from openai import AsyncOpenAI

from ...tools import Tool
from ..types import (
    AdapterEvent,
    FunctionCall,
    LLMResponse,
    Message,
    ToolCall,
    normalize_finish_reason,
)
from .base import BaseAdapter, make_client


class OpenAIAdapter(BaseAdapter):
    """OpenAI Chat Completions 协议适配器（含兼容端点）。"""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        client=None,
        client_factory=None,
        owns_client=None,
        **kwargs,
    ):
        self.client, self.owns_client = make_client(
            AsyncOpenAI, api_key, base_url, kwargs, client, client_factory, owns_client
        )
        self.model = model
        self.supports_reasoning_content = kwargs.pop("supports_reasoning_content", False)
        token_parameter = kwargs.pop("output_token_parameter", "max_tokens")
        if token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Unsupported Chat Completions output token parameter")
        kwargs.update(temperature=temperature)
        if "max_completion_tokens" not in kwargs:
            kwargs[token_parameter] = max_tokens
        self.default_params: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}

    # ── 请求构建 ─────────────────────────────────────────────────────

    def build_request(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool | None = False,
        **kwargs,
    ) -> dict[str, Any]:
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            m = msg.to_openai()
            if self.supports_reasoning_content and msg.reasoning_content:
                m["reasoning_content"] = msg.reasoning_content
            else:
                m.pop("reasoning_content", None)
            openai_messages.append(m)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": bool(stream),
            **self.default_params,
            **kwargs,
        }
        if stream and "stream_options" not in request:
            request["stream_options"] = {"include_usage": True}
        if tools:
            request["tools"] = [
                tool.to_schema() if hasattr(tool, "to_schema") else tool for tool in tools
            ]
        return request

    async def call(self, request: dict[str, Any]) -> Any:
        return await self.client.chat.completions.create(**request)

    # ── 非流式解析 ───────────────────────────────────────────────────

    def parse_response(self, raw_response: Any) -> LLMResponse:
        choice = raw_response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls and normalize_finish_reason(choice.finish_reason) in (
            "stop",
            "tool_calls",
        ):
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError("Malformed tool arguments") from exc
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(name=tc.function.name, arguments=arguments),
                    )
                )

        if normalize_finish_reason(choice.finish_reason) == "tool_calls" and not tool_calls:
            raise ValueError("Tool-call termination without a tool call")

        internal_msg = Message(
            role="assistant",
            content=message.content or "",
            reasoning_content=self._get_extra_field(message, "reasoning_content"),
            tool_calls=tool_calls,
        )

        return LLMResponse(
            message=internal_msg,
            finish_reason=choice.finish_reason,
            usage=self._parse_usage(getattr(raw_response, "usage", None)),
            model=raw_response.model,
            usage_available=getattr(raw_response, "usage", None) is not None,
            latency_ms=0,  # 由网关填充
        )

    # ── 流式解析 ─────────────────────────────────────────────────────

    def parse_stream_chunk(self, chunk: Any) -> AdapterEvent | None:
        if not chunk.choices:
            # stream_options.include_usage 的最后一个 usage-only chunk
            usage = self._parse_usage(getattr(chunk, "usage", None))
            return {"usage": usage} if getattr(chunk, "usage", None) is not None else None

        delta = chunk.choices[0].delta
        result: dict[str, Any] = {
            "content": delta.content or "",
            "reasoning_content": self._get_extra_field(delta, "reasoning_content") or "",
            "finish_reason": chunk.choices[0].finish_reason,
        }
        if delta.tool_calls:
            result["tool_calls_delta"] = []
            for tc in delta.tool_calls:
                item: dict[str, Any] = {"index": tc.index}
                if tc.id:
                    item["id"] = tc.id
                if tc.function and tc.function.name:
                    item["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    item["arguments"] = tc.function.arguments
                result["tool_calls_delta"].append(item)

        usage = self._parse_usage(getattr(chunk, "usage", None))
        if getattr(chunk, "usage", None) is not None:
            result["usage"] = usage
        return result

    # ── 公共工具 ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_usage(usage: Any) -> dict[str, int]:
        """统一解析 usage，避免不同端点返回 None 或缺字段时崩溃。"""
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", None)
        total = total if total is not None else prompt + completion
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    @staticmethod
    def _get_extra_field(obj: Any, field_name: str) -> Any:
        """从 SDK 模型对象读取扩展字段（如 reasoning_content）。"""
        value = getattr(obj, field_name, None)
        if value is not None:
            return value
        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(field_name)
        return None
