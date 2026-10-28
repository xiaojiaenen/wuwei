"""OpenAI Responses API 适配器

基于 openai SDK 的 client.responses 接口（GPT-5/o 系列推荐通道），
支持流式事件与工具调用（function call）。
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
)
from .base import BaseAdapter, make_client
from .usage import parse_openai_usage


class ResponsesAdapter(BaseAdapter):
    """OpenAI Responses API 适配器。"""

    provider_name = "responses"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        prompt_cache_key: str | None = None,
        max_output_tokens: int = 4096,
        client=None,
        client_factory=None,
        owns_client=None,
        **kwargs,
    ):
        self.client, self.owns_client = make_client(
            AsyncOpenAI, api_key, base_url, kwargs, client, client_factory, owns_client
        )
        max_output_tokens = kwargs.pop("max_tokens", max_output_tokens)
        self.model = model
        self.default_params: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
        if prompt_cache_options is not None:
            self.default_params["prompt_cache_options"] = dict(prompt_cache_options)
        if prompt_cache_key is not None:
            self.default_params["prompt_cache_key"] = prompt_cache_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    # ── 消息/工具格式转换 ─────────────────────────────────────────────

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """wuwei 消息 → Responses API 的 (instructions, input items)。"""
        instructions_parts: list[str] = []
        items: list[dict[str, Any]] = []

        for msg in messages:
            items.extend(msg.metadata.get("responses_reasoning", []))
            if msg.role == "system":
                if msg.content:
                    instructions_parts.append(msg.content)
            elif msg.role == "tool" and msg.tool_call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content or "",
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                if msg.content:
                    items.append({"role": "assistant", "content": msg.content})
                for tc in msg.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.dumps(tc.function.arguments, ensure_ascii=False),
                        }
                    )
            elif msg.content:
                items.append({"role": msg.role, "content": msg.content})

        return "\n\n".join(instructions_parts), items

    def _convert_tools(self, tools: list) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            schema = tool.to_schema() if hasattr(tool, "to_schema") else tool
            if isinstance(schema, dict) and "function" in schema:
                func = schema["function"]
                converted.append(
                    {
                        "type": "function",
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        return converted

    # ── 请求构建/调用 ─────────────────────────────────────────────────

    def build_request(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool | None = False,
        **kwargs,
    ) -> dict[str, Any]:
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        instructions, input_items = self._convert_messages(messages)

        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": bool(stream),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            **self.default_params,
            **kwargs,
        }
        if instructions:
            request["instructions"] = instructions
        if tools:
            request["tools"] = self._convert_tools(tools)
        return {k: v for k, v in request.items() if v is not None}

    async def call(self, request: dict[str, Any]) -> Any:
        return await self.client.responses.create(**request)

    # ── 非流式解析 ───────────────────────────────────────────────────

    def parse_response(self, raw_response: Any) -> LLMResponse:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in getattr(raw_response, "output", None) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for block in getattr(item, "content", None) or []:
                    if getattr(block, "type", None) == "output_text":
                        content_parts.append(getattr(block, "text", ""))
            elif (
                item_type == "function_call"
                and getattr(raw_response, "status", "completed") == "completed"
            ):
                try:
                    arguments = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError("Malformed tool arguments") from exc
                tool_calls.append(
                    ToolCall(
                        id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                        type="function",
                        function=FunctionCall(name=item.name, arguments=arguments),
                    )
                )

        usage = self._parse_usage(getattr(raw_response, "usage", None))

        return LLMResponse(
            message=Message(
                role="assistant",
                content="".join(content_parts),
                tool_calls=tool_calls,
                metadata={
                    "responses_reasoning": [
                        item.model_dump() if hasattr(item, "model_dump") else vars(item)
                        for item in getattr(raw_response, "output", [])
                        if getattr(item, "type", None) == "reasoning"
                    ]
                },
            ),
            finish_reason=("tool_calls" if tool_calls else "stop")
            if getattr(raw_response, "status", "completed") == "completed"
            else getattr(raw_response, "status", "unknown"),
            raw_finish_reason=getattr(raw_response, "status", None),
            usage=usage,
            model=getattr(raw_response, "model", self.model),
            usage_available=getattr(raw_response, "usage", None) is not None,
            latency_ms=0,  # 由网关填充
        )

    # ── 流式解析 ─────────────────────────────────────────────────────
    #
    # Responses API 流式事件（节选）：
    #   response.output_text.delta            → 文本增量
    #   response.output_item.added            → 新输出项（function_call 开始）
    #   response.function_call_arguments.delta→ 工具参数增量
    #   response.completed                    → 最终响应（含 usage）

    def parse_stream_chunk(self, chunk: Any) -> AdapterEvent | None:
        event_type = getattr(chunk, "type", None)

        if event_type == "response.reasoning_summary_text.delta":
            return {"reasoning_content": getattr(chunk, "delta", "") or ""}

        if event_type == "response.output_text.delta":
            return {"content": getattr(chunk, "delta", "") or ""}

        if event_type == "response.output_item.added":
            item = getattr(chunk, "item", None)
            if item is not None and getattr(item, "type", None) == "function_call":
                return {
                    "tool_calls_delta": [
                        {
                            "index": getattr(chunk, "output_index", 0),
                            "id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                            "name": getattr(item, "name", ""),
                        }
                    ],
                }
            return None

        if event_type == "response.function_call_arguments.delta":
            return {
                "tool_calls_delta": [
                    {
                        "index": getattr(chunk, "output_index", 0),
                        "arguments": getattr(chunk, "delta", "") or "",
                    }
                ],
            }

        if event_type in ("response.completed", "response.incomplete", "response.failed"):
            if event_type != "response.completed":
                return {"finish_reason": event_type.split(".")[1]}
            response = getattr(chunk, "response", None)
            result: dict[str, Any] = {}
            output = getattr(response, "output", None) or []
            has_tool_calls = any(getattr(item, "type", None) == "function_call" for item in output)
            result["finish_reason"] = "tool_calls" if has_tool_calls else "stop"
            result["provider_metadata"] = {
                "responses_reasoning": [
                    item.model_dump() if hasattr(item, "model_dump") else vars(item)
                    for item in output
                    if getattr(item, "type", None) == "reasoning"
                ]
            }
            usage = self._parse_usage(getattr(response, "usage", None))
            if getattr(response, "usage", None) is not None:
                result["usage"] = usage
            return result

        return None

    # ── 公共工具 ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_usage(usage: Any) -> dict[str, int]:
        return parse_openai_usage(usage, responses=True)
