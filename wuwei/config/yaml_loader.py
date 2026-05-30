"""YAML 配置加载器

支持通过 YAML 文件定义 Agent。
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
import yaml
import os


class LLMConfig(BaseModel):
    """LLM 配置"""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096


class ToolConfig(BaseModel):
    """工具配置"""
    name: str
    description: str = ""
    type: str = "function"  # function/builtin


class AgentConfig(BaseModel):
    """Agent 配置

    示例 YAML：
        agent:
          name: my-agent
          description: 我的 Agent
          system_prompt: 你是一个有用的助手

        llm:
          provider: zhipu
          model: glm-4

        tools:
          - name: search
            description: 搜索文档
          - name: get_weather
            description: 获取天气

        middleware:
          - type: logging
          - type: hitl
            auto_approve: [safe_tool]
    """
    name: str = "wuwei-agent"
    description: str = ""
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 20
    parallel_tool_calls: bool = False

    # LLM 配置
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # 工具配置
    tools: list[ToolConfig] = Field(default_factory=list)

    # 中间件配置
    middleware: list[dict] = Field(default_factory=list)


def load_agent_config(config_path: str) -> AgentConfig:
    """从 YAML 文件加载 Agent 配置

    示例：
        config = load_agent_config("agent.yaml")
        agent = create_agent_from_config(config)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AgentConfig(**data)


def create_agent_from_config(config: AgentConfig):
    """从配置创建 Agent

    需要安装相应的扩展包。
    """
    from wuwei.llm.gateway import LLMGateway
    from wuwei.agent.agent import Agent

    # 创建 LLM
    llm_config = {
        "provider": config.llm.provider,
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
    }
    if config.llm.api_key:
        llm_config["api_key"] = config.llm.api_key
    if config.llm.base_url:
        llm_config["base_url"] = config.llm.base_url

    llm = LLMGateway(llm_config)

    # 创建 Agent
    agent = Agent(
        llm=llm,
        default_system_prompt=config.system_prompt,
        default_max_steps=config.max_steps,
        default_parallel_tool_calls=config.parallel_tool_calls,
    )

    return agent
