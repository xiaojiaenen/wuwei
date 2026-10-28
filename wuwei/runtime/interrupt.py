"""中断信号（interrupt）— 可持久化挂起的执行控制流

v3.2 内核能力：agent 在等待人工审批等场景下，把完整状态存入 checkpointer
后优雅挂起（而不是异常炸穿），随后用 Agent.resume() 从断点恢复。

与普通异常的区别：InterruptSignal 是控制流信号，AgentRunner 会捕获它、
保存检查点、发出 approval_required 事件并正常结束本次 run。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wuwei.core.message import ToolCall


@dataclass
class InterruptPayload:
    """一次挂起的完整描述（可 JSON 序列化）。"""

    checkpoint_id: str
    session_id: str
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    run_id: str | None = None
    tool_call_id: str | None = None
    continuation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "reason": self.reason,
            "run_id": self.run_id,
            "tool_call_id": self.tool_call_id,
            "continuation": self.continuation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterruptPayload:
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args") or {},
            reason=data.get("reason", ""),
            run_id=data.get("run_id"),
            tool_call_id=data.get("tool_call_id"),
            continuation=data.get("continuation"),
        )


class InterruptSignal(Exception):
    """中间件抛出：请求挂起本次 run，等待外部恢复。

    Attributes:
        tool_call: 需要审批的工具调用
        reason: 挂起原因
        completed: 中断发生前已完成的工具调用（本步内）
        pending: 尚未执行的工具调用（本步内，含触发中断的那个）
    """

    def __init__(
        self,
        tool_call: ToolCall,
        reason: str = "等待人工审批",
        completed: list[ToolCall] | None = None,
        pending: list[ToolCall] | None = None,
    ):
        self.tool_call = tool_call
        self.reason = reason
        self.completed = completed or []
        self.pending = pending or [tool_call]
        super().__init__(reason)
