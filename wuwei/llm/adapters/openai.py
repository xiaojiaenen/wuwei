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
from .base import BaseAdapter, make_client, output_parts, provider_dump
from .usage import parse_openai_usage


def _coalesce_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把开头的多条 system 合并成一条，保证 system 唯一且在首位。

    Chat Completions 协议要求 system 出现在最前面；严格实现的端点（DeepSeek、
    vLLM、多数内网网关）还要求它唯一，遇到相邻两条直接 400。框架里 RAG、
    记忆、结构化输出等中间件都会向消息视图注入 system，合并放在适配器边界，
    任何注入源都不必各自保证顺序。

    Anthropic 与 Responses 适配器一直是这样处理的，这里补齐 OpenAI 一侧。
    只合并开头连续的一段：出现在对话中段的 system 属于调用方有意为之，
    合并会改变语义，原样透传让端点自己报错更有助于定位。
    """
    head: list[dict[str, Any]] = []
    rest_start = 0
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            rest_start = index
            break
        head.append(message)
        rest_start = index + 1
    if len(head) <= 1:
        return messages

    texts: list[str] = []
    for message in head:
        content = message.get("content")
        if isinstance(content, list):
            # 多模态 system（文本块 + 图片块）：只合并文本，其余块按序保留。
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    if block.get("text"):
                        texts.append(block["text"])
                elif isinstance(block, str) and block:
                    texts.append(block)
        elif content:
            texts.append(str(content))

    merged: dict[str, Any] = {"role": "system", "content": "\n\n".join(texts)}
    name = next((m.get("name") for m in head if m.get("name")), None)
    if name:
        merged["name"] = name
    return [merged, *messages[rest_start:]]


class OpenAIAdapter(BaseAdapter):
    """OpenAI Chat Completions 协议适配器（含兼容端点）。"""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float | None = 0.2,
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
        openai_messages = _coalesce_system_messages(openai_messages)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": bool(stream),
            **self.default_params,
            **kwargs,
        }
        if stream and "stream_options" not in request:
            request["stream_options"] = {"include_usage": True}
        if stream:
            # Some compatible gateways choose the response format from Accept and
            # answer a streaming request with concatenated JSON, which the SSE
            # decoder silently drops. The header is request-scoped so non-stream
            # calls keep the SDK default.
            request["extra_headers"] = {
                **(request.get("extra_headers") or {}),
                "Accept": "text/event-stream",
            }
        if tools:
            request["tools"] = [
                tool.to_schema() if hasattr(tool, "to_schema") else tool for tool in tools
            ]
        return {k: v for k, v in request.items() if v is not None}

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
            content_parts=output_parts(
                [
                    {
                        "type": "text",
                        "text": message.content or "",
                        "annotations": getattr(message, "annotations", []) or [],
                    }
                ]
            ),
            metadata={
                "provider_output": provider_dump(message),
                "refusal": getattr(message, "refusal", None),
            },
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
            "provider_metadata": {
                "provider_delta": provider_dump(delta),
                "refusal_delta": getattr(delta, "refusal", None),
            },
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
        return parse_openai_usage(usage)

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
