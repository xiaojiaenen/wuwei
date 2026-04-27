from __future__ import annotations

from uuid import uuid4

from wuwei.llm import ToolCall
from wuwei.runtime.hitl import (
    ApprovalPolicy,
    ApprovalProvider,
    ApprovalRequest,
    ToolApprovalRejected,
)
from wuwei.runtime.hooks import RuntimeHook


class HitlHook(RuntimeHook):
    """Runtime hook that gates selected tool calls behind human approval."""

    def __init__(
        self,
        provider: ApprovalProvider,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy or ApprovalPolicy()

    async def before_tool(self, session, tool_call: ToolCall, *, step: int, task=None) -> None:
        if not self.policy.requires_tool_approval(tool_call, session=session, task=task):
            return

        request = ApprovalRequest(
            id=uuid4().hex,
            session_id=session.session_id,
            action_type="tool_call",
            tool_call=tool_call,
            payload={
                "tool_name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
                "tool_call_id": tool_call.id,
                "step": step,
            },
            metadata=dict(getattr(session, "metadata", {}) or {}),
        )

        decision = await self.provider.request_approval(request)
        if decision.status != "approved":
            raise ToolApprovalRejected(decision.reason or "tool call rejected by human")
