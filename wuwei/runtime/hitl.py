from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from wuwei.llm import ToolCall

ApprovalStatus = Literal["approved", "rejected", "pending"]


@dataclass
class ApprovalRequest:
    """A request for human approval before an action is executed."""

    id: str
    session_id: str
    action_type: str
    payload: dict[str, Any]
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalDecision:
    """The result returned by an approval provider."""

    status: ApprovalStatus
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ApprovalProvider(Protocol):
    """Backend-agnostic approval provider.

    Applications can implement this with a console prompt, Web UI, IM bot,
    database workflow, or any other approval mechanism.
    """

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class ApprovalPolicy:
    """Default approval policy based on tool names."""

    def __init__(
        self,
        *,
        require_approval_tools: set[str] | None = None,
        auto_approve_tools: set[str] | None = None,
    ) -> None:
        self.require_approval_tools = require_approval_tools or set()
        self.auto_approve_tools = auto_approve_tools or set()

    def requires_tool_approval(self, tool_call: ToolCall, *, session, task=None) -> bool:
        tool_name = tool_call.function.name

        if tool_name in self.auto_approve_tools:
            return False

        if tool_name in self.require_approval_tools:
            return True

        return False


class ToolApprovalRejected(Exception):
    """Raised when a human rejects a tool call."""


class ConsoleApprovalProvider:
    """Simple local provider for development and examples."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        tool_name = request.payload.get("tool_name", request.action_type)
        arguments = request.payload.get("arguments", {})

        print("\n[HITL] Human approval required")
        print(f"approval_id: {request.id}")
        print(f"session_id: {request.session_id}")
        print(f"action: {tool_name}")
        print(f"arguments: {arguments}")

        answer = await asyncio.to_thread(input, "Approve? Type y/N: ")
        if answer.strip().lower() == "y":
            return ApprovalDecision(status="approved", reason="approved from console")

        return ApprovalDecision(status="rejected", reason="rejected from console")
