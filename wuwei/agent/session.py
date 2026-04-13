from dataclasses import dataclass, field

from wuwei.memory import Context
from wuwei.skill.skill import Skill


@dataclass
class AgentSession:
    """保存一次会话的配置和上下文。"""

    session_id: str
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 10
    parallel_tool_calls: bool = False
    context: Context = field(init=False)
    active_skills: list[Skill] = field(init=False)

    def __post_init__(self) -> None:
        """初始化后立刻重置上下文。"""
        self.reset()

    def reset(self) -> None:
        """清空上下文，并重新写入 system prompt。"""
        self.context = Context()
        self.context.add_system_message(self.system_prompt)
