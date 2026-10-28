"""Anthropic 适配器 - 支持 Claude 系列模型

修复历史问题：
- 旧实现 build_request 从不写 stream 字段，流式请求返回 Message 而非 Stream，
  网关 async for 直接 TypeError；
- 旧实现把 tool 参数的 input_json_delta.partial_json 当作文本 content 返回，
  工具参数碎片污染文本流、arguments 永远拼不出来。
现在 tool 参数增量走 tool_calls_delta 通道，由网关按 index 聚合。
"""

import inspect
import json
from typing import Any

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


class AnthropicAdapter(BaseAdapter):
    """Anthropic 适配器 — 支持 Claude 3/4 系列"""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        temperature: float | None = None,
        enable_prompt_caching: bool = True,
        base_url: str | None = None,
        client=None,
        client_factory=None,
        owns_client=None,
        **kwargs,
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "使用 Anthropic 适配器需要安装 anthropic 包：\npip install anthropic"
            ) from None

        self.client, self.owns_client = make_client(
            AsyncAnthropic, api_key, base_url, kwargs, client, client_factory, owns_client
        )
        self.default_params = kwargs
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_prompt_caching = enable_prompt_caching

    def build_request(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool | None = False,
        **kwargs,
    ) -> dict:
        """构建 Anthropic API 请求"""
        system_msg: Any = ""
        anthropic_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_msg = "\n\n".join(filter(None, [system_msg, msg.content]))
                continue

            if msg.role == "tool" and msg.tool_call_id:
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.provider_content("anthropic"),
                    "is_error": getattr(msg, "status", "success") == "error",
                }
                if (
                    anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and isinstance(anthropic_messages[-1]["content"], list)
                ):
                    anthropic_messages[-1]["content"].append(block)
                else:
                    anthropic_messages.append({"role": "user", "content": [block]})
                continue

            content: Any = list(msg.metadata.get("anthropic_reasoning", []))
            if msg.content_parts:
                content.extend(msg.provider_content("anthropic"))
            elif msg.content:
                content.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                        }
                    )
            anthropic_messages.append(
                {
                    "role": msg.role,
                    "content": content if content else (msg.content or ""),
                }
            )

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
            "messages": anthropic_messages,
            "stream": bool(stream),
            **self.default_params,
            **kwargs,
        }

        if system_msg:
            if self.enable_prompt_caching and isinstance(system_msg, str):
                # 对 system prompt 启用 prompt caching（长对话成本显著下降）
                request["system"] = [
                    {
                        "type": "text",
                        "text": system_msg,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                request["system"] = system_msg

        if tools:
            request["tools"] = self._convert_tools(tools)

        return {k: v for k, v in request.items() if v is not None}

    def _convert_tools(self, tools: list) -> list[dict]:
        """转换工具格式为 Anthropic 格式（支持 Tool 对象与 dict schema）"""
        converted = []
        for tool in tools:
            if hasattr(tool, "to_schema"):
                schema = tool.to_schema()
                func = schema.get("function", schema)
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}, "required": []}
                        ),
                    }
                )
            elif isinstance(tool, dict) and "function" in tool:
                converted.append(
                    {
                        "name": tool["function"]["name"],
                        **(
                            {"strict": tool["function"]["strict"]}
                            if "strict" in tool["function"]
                            else {}
                        ),
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"].get(
                            "parameters", {"type": "object", "properties": {}, "required": []}
                        ),
                    }
                )
            elif isinstance(tool, dict) and "name" in tool:
                converted.append(tool)
        return converted

    async def call(self, request: dict) -> Any:
        """调用 Anthropic API（stream=True 时返回 AsyncStream）"""
        # SDK versions differ: newer Messages APIs removed sampling controls.
        parameters = inspect.signature(self.client.messages.create).parameters
        if "temperature" not in parameters and not any(
            p.kind == p.VAR_KEYWORD for p in parameters.values()
        ):
            if request.get("temperature") is not None:
                raise ValueError("Installed Anthropic SDK does not support temperature")
            request = {k: v for k, v in request.items() if k != "temperature"}
        return await self.client.messages.create(**request)

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """解析非流式响应"""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw_response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "thinking":
                reasoning_parts.append(block.thinking)
            elif block.type == "tool_use" and normalize_finish_reason(
                getattr(raw_response, "stop_reason", None)
            ) in ("stop", "tool_calls"):
                try:
                    args = json.loads(block.input) if isinstance(block.input, str) else block.input
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError("Malformed tool arguments") from exc
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(name=block.name, arguments=args),
                    )
                )

        usage = self._parse_usage(raw_response)

        internal_msg = Message(
            role="assistant",
            content="".join(content_parts),
            content_parts=output_parts(raw_response.content),
            tool_calls=tool_calls,
            reasoning_content="".join(reasoning_parts) or None,
            metadata={
                "refusal": "provider_refusal"
                if getattr(raw_response, "stop_reason", None) == "refusal"
                else None,
                "provider_output": [provider_dump(b) for b in raw_response.content],
                "anthropic_reasoning": [
                    b.model_dump() if hasattr(b, "model_dump") else vars(b)
                    for b in raw_response.content
                    if b.type in ("thinking", "redacted_thinking")
                ],
            },
        )

        return LLMResponse(
            message=internal_msg,
            finish_reason=getattr(raw_response, "stop_reason", None),
            usage_available=getattr(raw_response, "usage", None) is not None,
            usage=usage,
            model=raw_response.model,
            latency_ms=0,  # 由网关填充
        )

    def parse_stream_chunk(self, chunk: Any) -> AdapterEvent | None:
        """解析流式事件 → 网关协议格式。

        - text_delta         → {"content": ...}
        - input_json_delta   → {"tool_calls_delta": [{"index", "arguments"}]}
        - content_block_start(tool_use) → {"tool_calls_delta": [{"index", "id", "name"}]}
        - message_start      → {"usage": {...input_tokens}}
        - message_delta      → {"finish_reason", "usage": {...output_tokens}}
        """
        ctype = getattr(chunk, "type", None)

        if ctype == "content_block_start":
            block = getattr(chunk, "content_block", None)
            if block is not None and getattr(block, "type", None) == "tool_use":
                return {
                    "content": "",
                    "tool_calls_delta": [
                        {
                            "index": getattr(chunk, "index", 0),
                            "id": block.id,
                            "name": block.name,
                        }
                    ],
                }
            if block is not None and getattr(block, "type", None) in (
                "thinking",
                "redacted_thinking",
            ):
                return {
                    "provider_metadata": {
                        "anthropic_reasoning_start": {
                            "index": getattr(chunk, "index", 0),
                            "block": block.model_dump()
                            if hasattr(block, "model_dump")
                            else vars(block),
                        }
                    }
                }
            return {"provider_metadata": {"provider_event": provider_dump(chunk)}}

        if ctype == "content_block_delta":
            delta = chunk.delta
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta" or (
                hasattr(delta, "text") and not hasattr(delta, "partial_json")
            ):
                return {"content": delta.text or "", "tool_calls_delta": None}
            if delta_type == "input_json_delta" or hasattr(delta, "partial_json"):
                # 工具参数增量：走 tool_calls_delta 通道，绝不混入 content
                return {
                    "content": "",
                    "tool_calls_delta": [
                        {
                            "index": getattr(chunk, "index", 0),
                            "arguments": delta.partial_json or "",
                        }
                    ],
                }
            if getattr(delta, "type", None) == "signature_delta":
                return {
                    "provider_metadata": {
                        "anthropic_signature_delta": {
                            "index": getattr(chunk, "index", 0),
                            "signature": delta.signature,
                        }
                    }
                }
            if hasattr(delta, "thinking"):
                return {"reasoning_content": delta.thinking or "", "content": ""}
            return {"provider_metadata": {"provider_event": provider_dump(chunk)}}

        if ctype == "message_start":
            usage = self._parse_usage(getattr(chunk, "message", chunk))
            return {"content": "", "usage": usage}

        if ctype == "message_delta":
            result: dict[str, Any] = {"content": ""}
            stop_reason = getattr(getattr(chunk, "delta", None), "stop_reason", None)
            if stop_reason is not None:
                result["finish_reason"] = stop_reason
                if stop_reason == "refusal":
                    result["provider_metadata"] = {"refusal": "provider_refusal"}
            if hasattr(chunk, "usage") and chunk.usage is not None:
                result["usage"] = self._parse_usage(chunk)
            return result if (result.get("finish_reason") or result.get("usage")) else None

        return None

    @staticmethod
    def _parse_usage(response_or_event: Any) -> dict[str, int]:
        """从响应或流事件中解析 usage。

        非流式：usage.input_tokens / usage.output_tokens
        流式 message_start：usage.input_tokens（output 为 0）
        流式 message_delta：usage.output_tokens（input 为 0，由网关取 max 合并）
        """
        usage = getattr(response_or_event, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = getattr(usage, "input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        prompt += cache_read + cache_creation
        completion = getattr(usage, "output_tokens", 0) or 0
        return {
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
