from dataclasses import dataclass, field
from typing import Any

from wuwei.memory import Context


@dataclass
class AgentSession:
    """保存一次会话的配置和上下文。"""

    session_id: str
    system_prompt: str = "你是一个有用的助手"
    max_steps: int = 10
    parallel_tool_calls: bool = False
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
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

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "max_steps": self.max_steps,
            "parallel_tool_calls": self.parallel_tool_calls,
            "summary": self.summary,
            "metadata": self.metadata,
            "last_usage": self.last_usage,
            "last_latency_ms": self.last_latency_ms,
            "last_llm_calls": self.last_llm_calls,
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        session = cls(
            session_id=data["session_id"],
            system_prompt=data.get("system_prompt", "你是一个有用的助手"),
            max_steps=data.get("max_steps", 10),
            parallel_tool_calls=data.get("parallel_tool_calls", False),
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
            last_usage=data.get("last_usage", {}),
            last_latency_ms=data.get("last_latency_ms", 0),
            last_llm_calls=data.get("last_llm_calls", 0),
        )
        session.context = Context.from_dict(data.get("context", {}))
        return session
