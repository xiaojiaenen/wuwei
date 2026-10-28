"""Trusted execution envelope: business content never controls run governance."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolOutcome(BaseModel):
    status: Literal["success", "error", "unknown"] = "success"
    content: Any = ""
    executed: bool = True
    side_effect: bool = False
    retryable: bool = False
    reconciliation: dict[str, Any] = Field(default_factory=dict)

    def to_message(self, call_id: str, name: str):
        from wuwei.core.message import ToolMessage
        from wuwei.tools.executor import ToolExecutor

        return ToolMessage(
            content=ToolExecutor.serialize_output(None, self.content),
            tool_call_id=call_id,
            name=name,
            status="success" if self.status == "success" else "error",
            metadata={"tool_outcome": self.model_dump(mode="json", exclude={"content"})},
        )

    @classmethod
    def from_delegation(cls, payload: dict):
        """Explicit adapter for framework-owned child run results only."""
        status = payload.get("status")
        return cls(
            content=payload,
            status="unknown"
            if status == "unknown"
            else "success"
            if status == "completed"
            else "error",
            reconciliation=payload.get("reconciliation") or {},
        )
