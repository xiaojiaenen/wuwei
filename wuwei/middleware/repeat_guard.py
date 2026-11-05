"""Stop a model that keeps issuing the same call instead of changing approach."""

import hashlib
import json

from wuwei.core.message import ToolCall, ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext

_STATE_KEY = "repeat_call_guard"


def _fingerprint(tool_call: ToolCall) -> str:
    payload = json.dumps(
        {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RepeatCallGuard(Middleware):
    """Block the (threshold+1)-th consecutive identical call.

    A failing model often retries the same call verbatim. The block is not an
    exception: the model receives an error result and can pick another strategy,
    which keeps the run inside its normal step budget. The counter resets as soon
    as a different call is attempted, so deliberate polling is not affected once
    the model varies the arguments.
    """

    def __init__(self, threshold: int = 3, exempt: tuple[str, ...] = ()):
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.threshold = threshold
        self.exempt = set(exempt)

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall | ToolMessage | None:
        if tool_call.function.name in self.exempt:
            return tool_call
        state = ctx.metadata.setdefault(_STATE_KEY, {})
        key = _fingerprint(tool_call)
        if state.get("key") == key:
            state["count"] = int(state.get("count", 0)) + 1
        else:
            state["key"] = key
            state["count"] = 1
        if state["count"] <= self.threshold:
            return tool_call
        return ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": "repeated_call",
                        "message": (
                            f"工具 {tool_call.function.name} 已用相同参数连续调用 "
                            f"{self.threshold} 次，已被拦截。请改变参数或策略，"
                            "不要继续重复同一调用。"
                        ),
                    },
                },
                ensure_ascii=False,
            ),
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            status="error",
            metadata={"tool_outcome": {"status": "error", "executed": False}},
        )
