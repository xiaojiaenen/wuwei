"""Agent 模块"""

from wuwei.agent.agent import Agent
from wuwei.agent.base import BaseAgent, BaseSessionAgent
from wuwei.agent.session import AgentSession
from wuwei.agent.multi_agent import MultiAgentGraph, TeamMember, HandoffMiddleware
from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware
from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware

# 向后兼容别名
Swarm = MultiAgentGraph

__all__ = [
    "Agent",
    "AgentSession",
    "BaseAgent",
    "BaseSessionAgent",
    "MultiAgentGraph",
    "Swarm",  # 向后兼容
    "TeamMember",
    "SubAgent",
    "SubAgentMiddleware",
    "AsyncSubAgent",
    "AsyncSubAgentMiddleware",
    "HandoffMiddleware",
]
