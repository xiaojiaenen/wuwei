"""Agent 模块

注意：旧版 Agent 类使用 Hook 系统，新版 Agent 类使用 Middleware 系统。
"""

# 新版：使用 Middleware
from wuwei.agent.agent_v2 import Agent

# 旧版：使用 Hook（已弃用，保留用于向后兼容）
from wuwei.agent.agent import Agent as LegacyAgent
from wuwei.agent.base import BaseAgent, BaseSessionAgent
from wuwei.agent.plan_agent import PlanAgent
from wuwei.agent.session import AgentSession
from wuwei.agent.multi_agent import Swarm, TeamMember, SubTask

__all__ = [
    # 新版
    "Agent",
    # 旧版（已弃用）
    "LegacyAgent",
    "BaseAgent",
    "BaseSessionAgent",
    "PlanAgent",
    "AgentSession",
    "Swarm",
    "TeamMember",
    "SubTask",
]
