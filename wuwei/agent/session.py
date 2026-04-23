from dataclasses import dataclass, field

from wuwei.memory import Context


@dataclass
class AgentSession:
    """保存一次会话的配置和上下文。"""

    session_id: str
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 10
    parallel_tool_calls: bool = False
    last_usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    last_latency_ms: int = 0
    last_llm_calls: int = 0
    context: Context = field(init=False)

    def __post_init__(self) -> None:
        """初始化后立刻重置上下文。"""
        self.reset()

    def reset(self) -> None:
        """清空上下文，并重新写入 system prompt。"""
        self.context = Context()
        self.context.add_system_message(self.system_prompt)
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_latency_ms = 0
        self.last_llm_calls = 0
