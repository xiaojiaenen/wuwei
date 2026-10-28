"""YAML 配置加载器

支持通过 YAML 文件定义 Agent。

示例 YAML：
    agent:
      name: my-agent
      system_prompt: 你是一个有用的助手
      load_builtins: [calc, time, file]

    llm:
      provider: openai
      model: gpt-4o
      api_key_env: OPENAI_API_KEY   # 也支持直接写 api_key

    tools: []        # 自定义 Tool 对象需要在代码里注册，此处仅声明名字做校验
    middleware:      # 支持的 type: logging / storage / hitl / tracing
      - type: logging
      - type: hitl
        auto_approve: [calculate]
"""

import logging
import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("wuwei.config")


class LLMConfig(BaseModel):
    """LLM 配置"""

    model_config = ConfigDict(extra="forbid")
    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None  # 从环境变量读取 api key
    base_url: str | None = None
    temperature: float | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    prompt_cache_options: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    reasoning: dict[str, Any] | None = None
    model_profile: dict[str, Any] | None = None
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    max_completion_tokens: int | None = None
    timeout: float = 60
    retry: dict = Field(default_factory=lambda: {"max_attempts": 3})


class ToolConfig(BaseModel):
    """工具配置（声明式校验用；自定义工具需在代码中注册）"""

    name: str
    description: str = ""
    type: str = "function"


class AgentConfig(BaseModel):
    """Agent 配置"""

    model_config = ConfigDict(extra="forbid")
    checkpoint_path: str | None = None
    name: str = "wuwei-agent"
    description: str = ""
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 20
    parallel_tool_calls: bool = False
    load_builtins: bool | str | list[str] = "minimal"

    # LLM 配置
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # 工具配置
    tools: list[ToolConfig] = Field(default_factory=list)

    # 中间件配置
    middleware: list[dict] = Field(default_factory=list)


def load_agent_config(config_path: str) -> AgentConfig:
    """从 YAML 文件加载 Agent 配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Agent YAML must be an object")
    if "agent" in data:
        section = data.pop("agent")
        if not isinstance(section, dict):
            raise ValueError("agent section must be an object")
        overlap = section.keys() & data.keys()
        if overlap:
            raise ValueError(f"Duplicate agent settings: {sorted(overlap)}")
        data = {**section, **data}
    return AgentConfig(**data)


def _build_middleware(spec: dict):
    """根据声明构建中间件实例。"""
    from wuwei.middleware import MiddlewareStack  # noqa: F401  (确保包初始化)

    if not isinstance(spec, dict):
        raise ValueError("Middleware configuration must be an object")
    mw_type = spec.get("type")
    allowed = {
        "logging": {"type", "log_level"},
        "storage": {"type", "storage_path"},
        "tracing": {"type", "service_name", "console_exporter"},
        "hitl": {
            "type",
            "auto_approve",
            "auto_reject",
            "auto_approve_tools",
            "auto_reject_tools",
            "policy",
            "approval_policy",
            "mode",
        },
    }
    if mw_type in allowed and spec.keys() - allowed[mw_type]:
        raise ValueError(f"Unknown {mw_type} settings: {sorted(spec.keys() - allowed[mw_type])}")
    if mw_type == "permission":
        from wuwei.security import PermissionConfig, PermissionMiddleware

        return PermissionMiddleware(
            PermissionConfig(**{k: v for k, v in spec.items() if k != "type"})
        )
    if mw_type == "budget":
        from wuwei.middleware import Budget, BudgetMiddleware

        return BudgetMiddleware(
            Budget(**{k: v for k, v in spec.items() if k not in {"type", "model"}}),
            model=spec.get("model", ""),
        )
    if mw_type == "logging":
        from wuwei.middleware.logging import LoggingMiddleware

        return LoggingMiddleware(log_level=spec.get("log_level", logging.INFO))
    if mw_type == "storage":
        from wuwei.middleware.storage import StorageMiddleware

        return StorageMiddleware(storage_path=spec.get("storage_path", ".wuwei/sessions"))
    if mw_type == "tracing":
        from wuwei.observability.tracing import TracingMiddleware

        return TracingMiddleware(
            service_name=spec.get("service_name", "wuwei"),
            console_exporter=bool(spec.get("console_exporter", False)),
        )
    if mw_type == "hitl":
        from wuwei.middleware.hitl import HitlMiddleware

        async def _console_approval(tool_call) -> bool:
            import asyncio

            answer = await asyncio.to_thread(
                input, f"是否允许执行工具 {tool_call.function.name}? (y/N): "
            )
            return answer.strip().lower() in ("y", "yes")

        return HitlMiddleware(
            approval_provider=_console_approval,
            auto_approve_tools=spec.get("auto_approve_tools", spec.get("auto_approve")) or [],
            auto_reject_tools=spec.get("auto_reject_tools", spec.get("auto_reject")) or [],
            approval_policy=spec.get("approval_policy", spec.get("policy", "all")),
            mode=spec.get("mode", "callback"),
        )
    raise ValueError(
        f"不支持的中间件类型: {mw_type}（支持: logging/storage/tracing/hitl/permission/budget；"
        "自定义中间件请在代码中构造后传入 Agent(middleware=...)）"
    )


def effective_llm_config(config: LLMConfig) -> dict[str, Any]:
    """Only explicitly supplied model settings override gateway defaults (including null)."""
    values = config.model_dump(exclude={"api_key_env", "provider_options"}, exclude_unset=True)
    options = dict(config.provider_options)
    reserved = {
        "provider",
        "model",
        "api_key",
        "api_key_env",
        "base_url",
        "retry",
        "timeout",
        "fallback",
        "adapter",
        "adapter_factory",
        "model_profile",
        "artifact_store",
        "artifact_owner",
    }
    if options.keys() & reserved:
        raise ValueError(f"Reserved provider options: {sorted(options.keys() & reserved)}")
    overlap = values.keys() & options.keys()
    if overlap:
        raise ValueError(f"Duplicate provider options: {sorted(overlap)}")
    return {**values, "provider_options": options}


def create_agent_from_config(config: AgentConfig, **extra):
    """从配置创建 Agent。

    支持 config.llm.api_key_env 从环境变量读 key；两者都缺省时
    由 LLMGateway.from_env 的规则兜底（读 {PROVIDER}_API_KEY 或 .env 文件）。
    """
    from wuwei.agent.agent import Agent
    from wuwei.llm.gateway import LLMGateway
    from wuwei.middleware.stack import MiddlewareStack

    # 创建 LLM
    llm_config: dict[str, Any] = effective_llm_config(config.llm)
    if config.llm.api_key:
        llm_config["api_key"] = config.llm.api_key
    elif config.llm.api_key_env:
        api_key = os.getenv(config.llm.api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {config.llm.api_key_env} 未设置（llm.api_key_env）")
        llm_config["api_key"] = api_key
    if config.llm.base_url:
        llm_config["base_url"] = config.llm.base_url

    llm = LLMGateway(llm_config) if llm_config.get("api_key") else LLMGateway.from_env(**llm_config)

    # 构建中间件
    stack = MiddlewareStack()
    for spec in config.middleware:
        stack.add(_build_middleware(spec))

    if config.checkpoint_path and "checkpointer" not in extra:
        from wuwei.graph.checkpoint import SQLiteCheckpointer

        extra["checkpointer"] = SQLiteCheckpointer(config.checkpoint_path)
    agent = Agent(
        llm=llm,
        default_system_prompt=config.system_prompt,
        default_max_steps=config.max_steps,
        default_parallel_tool_calls=config.parallel_tool_calls,
        load_builtins=config.load_builtins,
        middleware=stack,
        **extra,
    )
    return agent
