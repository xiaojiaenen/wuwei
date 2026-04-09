import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Union

from .adapters import OpenAIAdapter
from .adapters.base import BaseAdapter
from .types import FunctionCall, LLMResponse, LLMResponseChunk, Message, ToolCall
from ..tools import Tool


class LLMGateway:
    _DEFAULT_ENV_SEARCH_DEPTH = 3
    _DEFAULT_ENV_FILES = (".env", "env")
    _DEFAULT_ENV_PREFIX = "OPENAI"

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
        *,
        env_prefix: str | None = None,
        env_file: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **config: Any,
    ) -> "LLMGateway":
        """
        从环境变量创建 LLMGateway。

        这个方法只保留少量高频参数：
        - `env_prefix`：决定读取哪组环境变量，默认是 `OPENAI`
        - `env_file`：显式指定 env 文件
        - `model` / `base_url`：作为显式覆盖值

        环境变量命名固定为：
        - `{PREFIX}_API_KEY`
        - `{PREFIX}_BASE_URL`
        - `{PREFIX}_MODEL`

        框架内部会自动在当前目录和最多 3 层父目录中查找 `.env` / `env`，
        这部分不再暴露给用户配置，保持接口简单。
        """
        prefix = (env_prefix or cls._DEFAULT_ENV_PREFIX).upper()
        api_key_env = f"{prefix}_API_KEY"
        base_url_env = f"{prefix}_BASE_URL"
        model_env = f"{prefix}_MODEL"

        file_values = cls._load_env_file(env_file=env_file)

        api_key = os.getenv(api_key_env) or file_values.get(api_key_env)
        if not api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")

        env_config: dict[str, Any] = {
            "provider": "openai",
            "api_key": api_key,
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
                wait_time = 2**attempt
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
