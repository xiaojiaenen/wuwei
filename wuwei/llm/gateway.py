import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Union

from .adapters import OpenAIAdapter, AnthropicAdapter, ZhipuAdapter, DashScopeAdapter, OllamaAdapter
from .adapters.base import BaseAdapter
from .types import FunctionCall, LLMResponse, LLMResponseChunk, Message, ToolCall
from ..tools import Tool


class LLMGateway:
    _DEFAULT_ENV_SEARCH_DEPTH = 3
    _DEFAULT_ENV_FILES = (".env", "env")
    _DEFAULT_ENV_PREFIX = "OPENAI"

    def __init__(self, config: dict[str, Any]):
        """根据显式配置初始化模型网关。

        支持的 provider:
        - openai: OpenAI GPT 系列
        - anthropic: Anthropic Claude 系列
        - zhipu: 智谱 AI GLM-4 系列
        - dashscope: 阿里云通义千问系列
        - ollama: 本地模型（通过 Ollama）
        """
        self.adapter: BaseAdapter
        provider = config.get("provider", "openai")

        if provider == "openai":
            adapter_kwargs = {
                "api_key": config["api_key"],
                "model": config.get("model", "gpt-4o"),
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 4096),
            }
            if config.get("base_url"):
                adapter_kwargs["base_url"] = config["base_url"]
            if config.get("extra_body"):
                adapter_kwargs["extra_body"] = config["extra_body"]

            self.adapter = OpenAIAdapter(**adapter_kwargs)

        elif provider == "anthropic":
            adapter_kwargs = {
                "api_key": config["api_key"],
                "model": config.get("model", "claude-sonnet-4-6"),
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 4096),
            }
            self.adapter = AnthropicAdapter(**adapter_kwargs)

        elif provider == "zhipu":
            adapter_kwargs = {
                "api_key": config["api_key"],
                "model": config.get("model", "glm-4"),
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 4096),
            }
            if config.get("base_url"):
                adapter_kwargs["base_url"] = config["base_url"]
            self.adapter = ZhipuAdapter(**adapter_kwargs)

        elif provider == "dashscope":
            adapter_kwargs = {
                "api_key": config["api_key"],
                "model": config.get("model", "qwen-max"),
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 4096),
            }
            if config.get("base_url"):
                adapter_kwargs["base_url"] = config["base_url"]
            self.adapter = DashScopeAdapter(**adapter_kwargs)

        elif provider == "ollama":
            adapter_kwargs = {
                "model": config.get("model", "llama3"),
                "base_url": config.get("base_url", "http://localhost:11434"),
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 4096),
            }
            self.adapter = OllamaAdapter(**adapter_kwargs)

        else:
            raise ValueError(f"不支持的 provider: {provider}")

        self.retry_policy = config.get("retry", {"max_attempts": 3})
        self.timeout = config.get("timeout", 60)
        self._fallback_adapter: BaseAdapter | None = None

        # 配置 fallback 链
        fallback_config = config.get("fallback")
        if fallback_config:
            self._fallback_adapter = self._create_fallback_adapter(fallback_config)

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
        """
        从环境变量创建 LLMGateway。

        支持的 provider：
        - openai: 环境变量前缀 OPENAI_（默认）
        - anthropic: 环境变量前缀 ANTHROPIC_
        - zhipu: 环境变量前缀 ZHIPU_
        - dashscope: 环境变量前缀 DASHSCOPE_
        - ollama: 环境变量前缀 OLLAMA_

        环境变量命名固定为：
        - `{PREFIX}_API_KEY`
        - `{PREFIX}_BASE_URL`
        - `{PREFIX}_MODEL`
        """
        # 自动检测 provider
        if not provider:
            # 根据环境变量前缀自动检测
            if env_prefix:
                provider = env_prefix.lower()
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("ZHIPU_API_KEY"):
                provider = "zhipu"
            elif os.getenv("DASHSCOPE_API_KEY"):
                provider = "dashscope"
            else:
                provider = "openai"

        # 设置默认前缀
        if not env_prefix:
            prefix_map = {
                "openai": "OPENAI",
                "anthropic": "ANTHROPIC",
                "zhipu": "ZHIPU",
                "dashscope": "DASHSCOPE",
                "ollama": "OLLAMA",
            }
            env_prefix = prefix_map.get(provider, "OPENAI")

        prefix = env_prefix.upper()
        api_key_env = f"{prefix}_API_KEY"
        base_url_env = f"{prefix}_BASE_URL"
        model_env = f"{prefix}_MODEL"

        file_values = cls._load_env_file(env_file=env_file)

        api_key = os.getenv(api_key_env) or file_values.get(api_key_env)

        # Ollama 不需要 API key
        if not api_key and provider != "ollama":
            raise ValueError(f"缺少环境变量: {api_key_env}")

        env_config: dict[str, Any] = {
            "provider": provider,
            "api_key": api_key or "ollama",
        }

        resolved_base_url = base_url or os.getenv(base_url_env) or file_values.get(base_url_env)
        if resolved_base_url:
            env_config["base_url"] = resolved_base_url

        resolved_model = model or os.getenv(model_env) or file_values.get(model_env)
        if resolved_model:
            env_config["model"] = resolved_model

        env_config.update(config)
        return cls(env_config)

    @staticmethod
    def _load_env_file(env_file: str | None = None) -> dict[str, str]:
        """
        尝试从 env 文件读取变量。

        规则：
        1. 如果显式传入 `env_file`，只读取这个文件
        2. 否则自动查找当前目录和最多 3 层父目录中的 `.env` / `env`

        注意：
        - 这里只返回解析结果，不会修改 `os.environ`
        - 这是一个轻量实现，不依赖 `python-dotenv`
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
                value = value.strip()

                if not key:
                    continue

                if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]

                values[key] = value

            return values

        return {}

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

    async def _generate_single(
        self,
        messages: list[Message],
        tools: list[Tool] | None,
        **kwargs,
    ) -> LLMResponse:
        """发送一次非流式请求。

        分类重试策略：
        - 429 Rate Limit → 指数退避，使用 retry-after header
        - 5xx Server Error → 指数退避
        - 4xx (except 429) → 不重试（客户端错误）
        - Timeout → 指数退避
        - Connection Error → 指数退避

        如果配置了 fallback 链，在耗尽重试后会尝试 fallback。
        """
        request = self.adapter.build_request(messages=messages, tools=tools, stream=False, **kwargs)
        start = time.monotonic()
        last_exception = None

        for attempt in range(self.retry_policy["max_attempts"]):
            try:
                raw = await asyncio.wait_for(self.adapter.call(request), timeout=self.timeout)
                response = self.adapter.parse_response(raw)
                response.latency_ms = int((time.monotonic() - start) * 1000)
                return response
            except asyncio.TimeoutError as exc:
                last_exception = exc
                wait_time = 2 ** attempt
                if attempt < self.retry_policy["max_attempts"] - 1:
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as exc:
                last_exception = exc
                error_str = str(exc).lower()

                # 429 — 可重试
                if "429" in error_str or "rate limit" in error_str:
                    if attempt < self.retry_policy["max_attempts"] - 1:
                        wait_time = min(2 ** (attempt + 1), 60)  # cap at 60s for rate limits
                        await asyncio.sleep(wait_time)
                        continue
                # 5xx — 可重试
                elif any(f"{code}" in error_str for code in ["500", "502", "503", "504"]):
                    if attempt < self.retry_policy["max_attempts"] - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                # 4xx (except 429) — 不可重试
                elif any(f"{code}" in error_str for code in ["400", "401", "403", "404"]):
                    raise
                # 连接错误 — 可重试
                elif "connection" in error_str or "timeout" in error_str:
                    if attempt < self.retry_policy["max_attempts"] - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # 未知错误也重试几次
                    if attempt < self.retry_policy["max_attempts"] - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue

        # 尝试 fallback
        if self._fallback_adapter and last_exception:
            try:
                fallback_request = self._fallback_adapter.build_request(
                    messages=messages, tools=tools, stream=False, **kwargs
                )
                raw = await asyncio.wait_for(
                    self._fallback_adapter.call(fallback_request),
                    timeout=self.timeout,
                )
                response = self._fallback_adapter.parse_response(raw)
                response.latency_ms = int((time.monotonic() - start) * 1000)
                return response
            except Exception:
                pass

        raise last_exception

    def _create_fallback_adapter(self, config: dict) -> "BaseAdapter":
        """创建 fallback 适配器"""
        provider = config.get("provider", "openai")

        if provider == "openai":
            return OpenAIAdapter(
                api_key=config["api_key"],
                model=config.get("model", "gpt-4o-mini"),
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 4096),
                base_url=config.get("base_url"),
                **{k: v for k, v in config.items() if k not in ("provider", "api_key", "model", "temperature", "max_tokens", "base_url")},
            )
        elif provider == "anthropic":
            return AnthropicAdapter(
                api_key=config["api_key"],
                model=config.get("model", "claude-sonnet-4-6"),
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 4096),
            )
        elif provider == "ollama":
            return OllamaAdapter(
                model=config.get("model", "llama3"),
                base_url=config.get("base_url", "http://localhost:11434"),
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 4096),
            )
        else:
            raise ValueError(f"Unsupported fallback provider: {provider}")

    def set_fallback(self, fallback_gateway: "LLMGateway") -> None:
        """设置 fallback LLM 网关"""
        self._fallback_adapter = fallback_gateway.adapter

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
            reasoning_content = data.get("reasoning_content", "")
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

            out_chunk = LLMResponseChunk(
                content=content,
                reasoning_content=reasoning_content or None,
            )

            if finish_reason == "tool_calls":
                complete: list[ToolCall] = []
                for item in pending.values():
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
