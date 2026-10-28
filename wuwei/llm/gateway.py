"""LLM 网关 — 统一的模型接入层

支持的 provider（注册表驱动，新增 provider 只需加一行）：
- openai:     OpenAI Chat Completions（含任意 OpenAI 兼容端点，传 base_url 即可，
              如 DeepSeek / vLLM / Ollama / 聚合网关）
- responses:  OpenAI Responses API
- anthropic:  Anthropic Claude 系列

能力：分类重试（按状态码）、fallback 链、流式 tool_calls 增量聚合、
reasoning 透传、.env 自动加载（不依赖 python-dotenv）。
"""

import asyncio
import json
import os
import random
import time
import warnings
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from ..tools import Tool
from .adapters import AnthropicAdapter, OpenAIAdapter, ResponsesAdapter
from .adapters.base import BaseAdapter
from .output import STRUCTURED_TOOL_NAME, OutputSpec, apply_output
from .profile import ModelProfile, default_profile
from .types import (
    FunctionCall,
    LLMResponse,
    LLMResponseChunk,
    Message,
    ToolCall,
    normalize_finish_reason,
)

# provider 注册表：adapter 类 / 默认模型 / 环境变量前缀 / 是否需要 api_key
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "openai": {
        "adapter": OpenAIAdapter,
        "default_model": "gpt-4o",
        "env_prefix": "OPENAI",
        "requires_api_key": True,
    },
    "responses": {
        "adapter": ResponsesAdapter,
        "default_model": "gpt-4o",
        "env_prefix": "OPENAI",
        "requires_api_key": True,
    },
    "anthropic": {
        "adapter": AnthropicAdapter,
        "default_model": "claude-sonnet-4-6",
        "env_prefix": "ANTHROPIC",
        "requires_api_key": True,
    },
}

# Legacy names retain their own environment namespaces during migration.
for _alias, _prefix in (("zhipu", "ZHIPU"), ("dashscope", "DASHSCOPE"), ("ollama", "OLLAMA")):
    _PROVIDER_SPECS[_alias] = {
        **_PROVIDER_SPECS["openai"],
        "env_prefix": _prefix,
        "requires_api_key": _alias != "ollama",
    }

# 按优先级排列的 provider 自动探测规则：环境变量 → provider
_PROVIDER_DETECTION = [
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("OPENAI_API_KEY", "openai"),
]


class LLMGateway:
    _DEFAULT_ENV_SEARCH_DEPTH = 3
    _DEFAULT_ENV_FILES = (".env", "env")
    _DEFAULT_ENV_PREFIX = "OPENAI"

    def __init__(self, config: dict[str, Any]):
        """根据显式配置初始化模型网关。

        config 关键字段：
        - provider: openai | responses | anthropic
        - api_key / base_url / model / temperature / max_tokens
        - retry: {"max_attempts": 3}（可选）
        - timeout: 秒（默认 60）
        - fallback: {"provider": ..., "api_key": ..., ...}（可选）
        """
        config = dict(config)
        options = dict(config.pop("provider_options", {}) or {})
        overlap = options.keys() & config.keys()
        if overlap:
            raise ValueError(f"Duplicate provider options: {sorted(overlap)}")
        config.update(options)
        provider = config.get("provider", "openai")
        if provider in {"zhipu", "dashscope", "ollama"}:
            warnings.warn(
                f"provider={provider} is deprecated; use openai with base_url",
                DeprecationWarning,
                stacklevel=2,
            )
            defaults = {
                "zhipu": ("https://open.bigmodel.cn/api/paas/v4/", "glm-4"),
                "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
                "ollama": ("http://localhost:11434/v1", "llama3"),
            }
            url, model = defaults[provider]
            config.setdefault("base_url", url)
            config.setdefault("model", model)
            provider = "openai"
        spec = _PROVIDER_SPECS.get(provider)
        if spec is None:
            raise ValueError(f"不支持的 provider: {provider}（可选: {', '.join(_PROVIDER_SPECS)}）")

        self.model_profile = (
            ModelProfile.model_validate(config["model_profile"])
            if config.get("model_profile")
            else (
                spec.get("model_profile") or default_profile(provider, bool(config.get("base_url")))
            )
        )
        adapter_kwargs: dict[str, Any] = {
            "api_key": config.get("api_key") or "missing",
            "model": config.get("model", spec["default_model"]),
            "temperature": config.get(
                "temperature", None if provider in {"responses", "anthropic"} else 0.2
            ),
        }
        if provider == "responses":
            adapter_kwargs["max_output_tokens"] = config.get(
                "max_output_tokens", config.get("max_tokens", 4096)
            )
        else:
            adapter_kwargs["max_tokens"] = config.get("max_tokens", 4096)
        if config.get("base_url"):
            adapter_kwargs["base_url"] = config["base_url"]
        # 透传其余 provider 专属参数（如 enable_prompt_caching）
        known = {
            "provider",
            "api_key",
            "model",
            "base_url",
            "temperature",
            "max_tokens",
            "max_output_tokens",
            "retry",
            "timeout",
            "fallback",
            "adapter",
            "adapter_factory",
            "model_profile",
            "artifact_store",
            "artifact_owner",
        }
        adapter_kwargs.update({k: v for k, v in config.items() if k not in known})

        self.adapter: BaseAdapter = config.get("adapter") or (
            config.get("adapter_factory") or spec["adapter"]
        )(**adapter_kwargs)
        self.provider = provider
        self.artifact_store = config.get("artifact_store")
        self.artifact_owner = config.get("artifact_owner")
        self.retry_policy = config.get("retry") or {"max_attempts": 3}
        if not isinstance(self.retry_policy, dict):
            raise ValueError("retry must be a configuration mapping")
        for key in ("base_delay", "max_delay"):
            if float(self.retry_policy.get(key, 1)) < 0:
                raise ValueError(f"retry.{key} must be nonnegative")
        if int(self.retry_policy.get("max_attempts", 3)) < 1:
            raise ValueError("retry.max_attempts must be positive")
        self.timeout = float(config.get("timeout", 60))
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self._fallback_gateway = None
        self._owns_fallback = False
        self._fallback_adapter: BaseAdapter | None = None

        fallback_config = config.get("fallback")
        if fallback_config:
            self._fallback_gateway = LLMGateway(fallback_config)
            self._fallback_adapter = self._fallback_gateway.adapter
            self._owns_fallback = True

    # ── from_env ─────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        env_prefix: str | None = None,
        env_file: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **config: Any,
    ) -> "LLMGateway":
        """从环境变量创建 LLMGateway。

        环境变量命名：{PREFIX}_API_KEY / {PREFIX}_BASE_URL / {PREFIX}_MODEL
        - provider 缺省时按 ANTHROPIC_API_KEY → OPENAI_API_KEY 顺序自动探测
        - env_prefix 用于任意 OpenAI 兼容端点，例如：
            LLMGateway.from_env(provider="openai", env_prefix="DEEPSEEK")
          读取 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
        """
        if not provider:
            if env_prefix:
                provider = env_prefix.lower().replace("_compatible", "")
                if provider not in _PROVIDER_SPECS:
                    provider = "openai"
            else:
                provider = "openai"
                for env_name, detected in _PROVIDER_DETECTION:
                    if os.getenv(env_name) or env_file_has(env_name, env_file):
                        provider = detected
                        break

        if not env_prefix:
            env_prefix = _PROVIDER_SPECS.get(provider, {}).get("env_prefix", "OPENAI")

        prefix = env_prefix.upper()
        api_key_env = f"{prefix}_API_KEY"
        base_url_env = f"{prefix}_BASE_URL"
        model_env = f"{prefix}_MODEL"

        file_values = cls._load_env_file(env_file=env_file)

        def _get(name: str) -> str | None:
            return os.getenv(name) or file_values.get(name)

        api_key = _get(api_key_env)
        if not api_key and _PROVIDER_SPECS[provider]["requires_api_key"]:
            raise ValueError(f"缺少环境变量: {api_key_env}")

        env_config: dict[str, Any] = {"provider": provider, "api_key": api_key or ""}
        resolved_base_url = base_url or _get(base_url_env)
        if resolved_base_url:
            env_config["base_url"] = resolved_base_url
        resolved_model = model or _get(model_env)
        if resolved_model:
            env_config["model"] = resolved_model

        env_config.update(config)
        return cls(env_config)

    @staticmethod
    def _load_env_file(env_file: str | None = None) -> dict[str, str]:
        """轻量 .env 解析（不依赖 python-dotenv，不修改 os.environ）。

        显式传入 env_file 时只读该文件；否则向上搜索 3 层父目录。
        """
        candidate_paths: list[Path] = []
        if env_file:
            candidate_paths.append(Path(env_file))
        else:
            directories = [Path.cwd(), *Path.cwd().parents[: LLMGateway._DEFAULT_ENV_SEARCH_DEPTH]]
            for directory in directories:
                for filename in LLMGateway._DEFAULT_ENV_FILES:
                    candidate_paths.append(directory / filename)

        for path in candidate_paths:
            if not path.exists() or not path.is_file():
                continue

            values: dict[str, str] = {}
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key:
                    values[key] = value
            return values

        return {}

    # ── 生成入口 ─────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | AsyncIterator[LLMResponseChunk]:
        """统一处理单次和流式生成请求。"""
        if stream:
            return self._generate_stream(messages=messages, tools=tools, **kwargs)
        return await self._generate_single(messages=messages, tools=tools, **kwargs)

    def build_request(self, messages, tools=None, stream=False, output_spec=None, **kwargs):
        if tools and not self.model_profile.capabilities.tool_calling:
            raise ValueError("Profile does not support tool calling")
        if stream and not self.model_profile.capabilities.streaming:
            raise ValueError("Profile does not support streaming")
        import base64

        from ..core.message import FilePart, ImagePart

        resolved_messages = []
        for message in messages:
            parts = []
            for part in message.content_parts:
                if isinstance(part, ImagePart) and not self.model_profile.capabilities.image_input:
                    raise ValueError("Profile does not support image input")
                if isinstance(part, FilePart) and not self.model_profile.capabilities.file_input:
                    raise ValueError("Profile does not support file input")
                if isinstance(part, (ImagePart, FilePart)) and part.artifact:
                    if not self.artifact_store or not self.artifact_owner:
                        raise ValueError(
                            "Artifact input requires artifact_store and artifact_owner"
                        )
                    data = self.artifact_store.read(part.artifact, owner=self.artifact_owner)
                    url = (
                        "data:"
                        + part.artifact.media_type
                        + ";base64,"
                        + base64.b64encode(data).decode("ascii")
                    )
                    part = part.model_copy(update={"artifact": None, "url": url})
                parts.append(part)
            resolved_messages.append(message.model_copy(update={"content_parts": parts}))
        messages = resolved_messages
        spec = (
            OutputSpec.model_validate(output_spec) if isinstance(output_spec, dict) else output_spec
        )
        mode = spec.resolve(self.model_profile) if spec else None
        protocol = getattr(self.adapter, "provider_name", self.provider)
        if mode == "prompted":
            from ..core.message import SystemMessage

            messages = [
                *messages,
                SystemMessage("Return JSON matching this schema: " + json.dumps(spec.schema_)),
            ]
        if mode == "tool":
            from .output import strict_json_schema

            if tools:
                raise ValueError("Tool output cannot be combined with executable tools")
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": STRUCTURED_TOOL_NAME,
                        "description": "Return structured output",
                        "parameters": (
                            strict_json_schema(spec.schema_)
                            if protocol in {"openai", "responses"}
                            else spec.schema_
                        ),
                        "strict": True,
                    },
                }
            ]
            kwargs["tool_choice"] = (
                {"type": "tool", "name": STRUCTURED_TOOL_NAME}
                if protocol == "anthropic"
                else (
                    {"type": "function", "name": STRUCTURED_TOOL_NAME}
                    if protocol == "responses"
                    else {"type": "function", "function": {"name": STRUCTURED_TOOL_NAME}}
                )
            )
        request = self.adapter.build_request(messages, tools, stream=stream, **kwargs)
        if mode == "native":
            apply_output(request, spec, protocol)
        return request

    # ── 错误分类 ─────────────────────────────────────────────────────

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """按状态码/异常类型分类错误，返回 retryable | fatal | unknown。

        优先读 SDK 异常的 status_code 属性，避免脆弱的字符串匹配。
        """
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            if status in (408, 409, 429):
                return "rate_limit"
            if 500 <= status < 600:
                return "server_error"
            if status >= 400:
                return "fatal"
            return "unknown"
        name = type(exc).__name__
        if name in (
            "APIConnectionError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
            "RemoteProtocolError",
        ):
            return "connection_error"
        if name in (
            "APITimeoutError",
            "TimeoutError",
            "asyncio.TimeoutError",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
        ):
            return "timeout"
        return "unknown"

    # ── 非流式 ───────────────────────────────────────────────────────

    async def _generate_single(
        self,
        messages: list[Message],
        tools: list[Tool] | None,
        **kwargs,
    ) -> LLMResponse:
        """发送一次非流式请求。

        重试策略（按状态码分类）：
        - 429 → 指数退避（上限 60s）
        - 5xx / 连接错误 / 超时 → 指数退避
        - 其它 4xx → 不重试
        重试耗尽后若配置了 fallback 则切换备用 adapter。
        """
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout):
                response = await self._single_attempts(messages, tools, **kwargs)
        except Exception as exc:
            if self._fallback_gateway and self._retryable(exc):
                remaining = self.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Model request overall deadline exceeded") from exc
                async with asyncio.timeout(remaining):
                    response = await self._fallback_gateway._generate_single(
                        messages, tools, **kwargs
                    )
            else:
                raise
        response.latency_ms = int((time.monotonic() - start) * 1000)
        return response

    @classmethod
    def register_provider(
        cls,
        name,
        adapter_factory,
        *,
        default_model="",
        env_prefix="OPENAI",
        requires_api_key=True,
        model_profile=None,
    ):
        _PROVIDER_SPECS[name] = {
            "adapter": adapter_factory,
            "model_profile": ModelProfile.model_validate(model_profile)
            if model_profile
            else default_profile(name, True),
            "default_model": default_model,
            "env_prefix": env_prefix,
            "requires_api_key": requires_api_key,
        }

    @classmethod
    def _retryable(cls, exc):
        return cls._classify_error(exc) in {
            "rate_limit",
            "server_error",
            "connection_error",
            "timeout",
        }

    async def _backoff(self, exc, attempt):
        headers = getattr(getattr(exc, "response", None), "headers", {})
        retry_after = headers.get("retry-after") if headers else None
        if retry_after is not None:
            try:
                delay = max(0, float(retry_after))
            except (TypeError, ValueError):
                try:
                    date = parsedate_to_datetime(retry_after)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=UTC)
                    delay = max(0, (date - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    delay = float(self.retry_policy.get("base_delay", 1)) * 2**attempt
        else:
            delay = float(self.retry_policy.get("base_delay", 1)) * 2**attempt
            if self.retry_policy.get("jitter", True):
                delay = random.uniform(delay / 2, delay)
        await asyncio.sleep(min(delay, float(self.retry_policy.get("max_delay", 60))))

    async def _single_attempts(self, messages, tools, **kwargs):
        request = self.build_request(messages=messages, tools=tools, stream=False, **kwargs)
        for attempt in range(int(self.retry_policy.get("max_attempts", 3))):
            try:
                raw = await self.adapter.call(request)
                response = self.adapter.parse_response(raw)
                response.metadata.update(
                    provider=self.provider, model=response.model, retry_count=attempt
                )
                return response
            except Exception as exc:
                if not self._retryable(exc) or attempt + 1 >= int(
                    self.retry_policy.get("max_attempts", 3)
                ):
                    raise
                await self._backoff(exc, attempt)

    def _create_fallback_adapter(self, config):
        return LLMGateway(config).adapter

    def set_fallback(self, fallback_gateway):
        """Borrow the complete fallback policy; its caller retains lifecycle ownership."""
        self._fallback_gateway = fallback_gateway
        self._fallback_adapter = fallback_gateway.adapter
        self._owns_fallback = False

    async def aclose(self):
        close = getattr(self.adapter, "aclose", None)
        if close:
            await close()
        if self._owns_fallback and self._fallback_gateway:
            await self._fallback_gateway.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def _generate_stream(self, messages, tools, **kwargs):
        observed = False
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout):
                for attempt in range(int(self.retry_policy.get("max_attempts", 3))):
                    try:
                        async for chunk in self._stream_once(messages, tools, **kwargs):
                            # Publishing even usage/tool metadata commits this request.
                            chunk.metadata.update(
                                provider=self.provider,
                                model=getattr(self.adapter, "model", ""),
                                retry_count=attempt,
                            )
                            observed = True
                            yield chunk
                        return
                    except Exception as exc:
                        if (
                            observed
                            or not self._retryable(exc)
                            or attempt + 1 >= int(self.retry_policy.get("max_attempts", 3))
                        ):
                            raise
                        await self._backoff(exc, attempt)
        except Exception as exc:
            if not observed and self._fallback_gateway and self._retryable(exc):
                remaining = self.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Model request overall deadline exceeded") from exc
                async with asyncio.timeout(remaining):
                    async for chunk in self._fallback_gateway._generate_stream(
                        messages, tools, **kwargs
                    ):
                        yield chunk
            else:
                raise

    async def _stream_once(self, messages, tools, **kwargs):
        request = self.build_request(messages, tools, stream=True, **kwargs)
        stream = await self.adapter.call(request)
        pending = {}
        usage = {}
        terminal = False
        reasoning_blocks = {}
        active_reasoning_index = None
        provider_metadata = {}
        try:
            async for raw in stream:
                data = self.adapter.parse_stream_chunk(raw)
                if not data:
                    continue
                if terminal and (
                    data.get("content")
                    or data.get("reasoning_content")
                    or data.get("tool_calls_delta")
                ):
                    raise ValueError("Model produced content after terminal event")
                if data.get("usage") is not None:
                    for key, value in data["usage"].items():
                        usage[key] = max(usage.get(key, 0), value)
                    usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                        "completion_tokens", 0
                    )
                deltas = data.get("tool_calls_delta")
                if deltas:
                    for delta in deltas:
                        item = pending.setdefault(
                            delta["index"], {"id": "", "name": "", "arguments": ""}
                        )
                        for key in ("id", "name"):
                            if delta.get(key):
                                item[key] = delta[key]
                        item["arguments"] += delta.get("arguments", "")
                event_metadata = data.get("provider_metadata", {})
                refusal_delta = event_metadata.get("refusal_delta")
                if refusal_delta:
                    provider_metadata["refusal"] = (
                        provider_metadata.get("refusal", "") + refusal_delta
                    )
                event = event_metadata.get("provider_event")
                if event:
                    provider_metadata.setdefault("provider_events", []).append(event)
                provider_metadata.update(
                    {
                        k: v
                        for k, v in event_metadata.items()
                        if k not in {"refusal_delta", "provider_event"} and v is not None
                    }
                )
                start_block = event_metadata.get("anthropic_reasoning_start")
                if start_block:
                    active_reasoning_index = start_block["index"]
                    reasoning_blocks[active_reasoning_index] = dict(start_block["block"])
                if data.get("reasoning_content") and active_reasoning_index is not None:
                    block = reasoning_blocks[active_reasoning_index]
                    block["thinking"] = block.get("thinking", "") + data["reasoning_content"]
                signature = event_metadata.get("anthropic_signature_delta")
                if signature:
                    block = reasoning_blocks[signature["index"]]
                    block["signature"] = block.get("signature", "") + signature["signature"]
                if reasoning_blocks:
                    provider_metadata["anthropic_reasoning"] = list(reasoning_blocks.values())
                if "responses_reasoning" in event_metadata:
                    provider_metadata["responses_reasoning"] = event_metadata["responses_reasoning"]
                raw_reason = data.get("finish_reason")
                reason = normalize_finish_reason(raw_reason) if raw_reason is not None else None
                complete = None
                if reason is not None:
                    terminal = True
                if reason == "tool_calls" and not pending:
                    raise ValueError("Tool-call termination without a tool call")
                if reason in ("stop", "tool_calls") and pending:
                    complete = []
                    for item in pending.values():
                        if not item["id"] or not item["name"]:
                            raise ValueError("Incomplete tool call identity")
                        # Never replace corrupt or truncated JSON with executable empty arguments.
                        arguments = json.loads(item["arguments"])
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be a JSON object")
                        complete.append(
                            ToolCall(
                                id=item["id"],
                                function=FunctionCall(name=item["name"], arguments=arguments),
                            )
                        )
                    reason = "tool_calls"
                yield LLMResponseChunk(
                    content=data.get("content", ""),
                    reasoning_content=data.get("reasoning_content") or None,
                    metadata=dict(provider_metadata),
                    tool_calls_delta=deltas,
                    tool_calls_complete=complete,
                    finish_reason=reason,
                    raw_finish_reason=raw_reason,
                    usage=dict(usage) if usage else None,
                    usage_available=bool(usage),
                )
            if not terminal:
                raise ValueError("Model stream ended without a terminal event")
        finally:
            close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
            if close:
                await close()


def env_file_has(env_name: str, env_file: str | None = None) -> bool:
    """检查 env 文件中是否定义了某个变量（供 provider 探测使用）。"""
    values = LLMGateway._load_env_file(env_file)
    return env_name in values
