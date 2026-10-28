import asyncio
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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        """初始化后立刻重置上下文。"""
        self.reset()

    def reset(self) -> None:
        """清空上下文，并重新写入 system prompt。"""
        self.metadata.pop("_pending_interrupt", None)
        self.metadata.pop("_blocked_run", None)
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
        from wuwei.runtime.action_log import ACTION_LOG_KEY, ActionLog

        metadata = dict(self.metadata)
        if isinstance(metadata.get(ACTION_LOG_KEY), ActionLog):
            metadata[ACTION_LOG_KEY] = {
                "__type__": "ActionLog",
                "value": metadata[ACTION_LOG_KEY].to_dict(),
            }
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "max_steps": self.max_steps,
            "parallel_tool_calls": self.parallel_tool_calls,
            "summary": self.summary,
            "metadata": metadata,
            "last_usage": self.last_usage,
            "last_latency_ms": self.last_latency_ms,
            "last_llm_calls": self.last_llm_calls,
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        if data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported session snapshot version")
        session = cls(
            session_id=data["session_id"],
            system_prompt=data.get("system_prompt", "你是一个有用的助手"),
            max_steps=data.get("max_steps", 10),
            parallel_tool_calls=data.get("parallel_tool_calls", False),
            summary=data.get("summary"),
            metadata={},
            last_usage=data.get("last_usage", {}),
            last_latency_ms=data.get("last_latency_ms", 0),
            last_llm_calls=data.get("last_llm_calls", 0),
        )
        session.metadata = dict(data.get("metadata") or {})
        session.context = Context.from_dict(data.get("context", {}))
        session.last_usage = data.get("last_usage") or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        session.last_latency_ms = data.get("last_latency_ms", 0)
        session.last_llm_calls = data.get("last_llm_calls", 0)
        from wuwei.runtime.action_log import ACTION_LOG_KEY, ActionLog

        log = session.metadata.get(ACTION_LOG_KEY)
        if isinstance(log, dict) and log.get("__type__") == "ActionLog":
            session.metadata[ACTION_LOG_KEY] = ActionLog.from_dict(log["value"])
        return session
