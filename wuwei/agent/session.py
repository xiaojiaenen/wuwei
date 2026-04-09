from dataclasses import dataclass, field

from wuwei.core.context import Context


@dataclass
class AgentSession:
    session_id: str
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 10
    parallel_tool_calls: bool = False
    context: Context = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.context = Context()
        self.context.add_system_message(self.system_prompt)
