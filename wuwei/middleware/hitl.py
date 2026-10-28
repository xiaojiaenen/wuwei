"""Human-in-the-Loop 中间件

v3.1 行为变更：审批拒绝不再抛异常炸穿整个 run，
而是返回一条结构化错误的 ToolMessage 回喂给模型——
模型能看到"被拒绝"及原因，从而调整策略继续对话。

v3.2 新增 interrupt 模式（mode="interrupt"，需 Agent 配置 checkpointer）：
需要审批时把完整状态存入检查点并优雅挂起，
用 Agent.resume(interrupt, approved=...) 从断点恢复执行。

默认审批策略：所有工具都会询问 approval_provider；
auto_approve_tools / auto_reject_tools 作为覆盖项；
approval_policy="declared" 时仅审 execution.requires_approval=True 的工具。
"""

import json
from collections.abc import Awaitable, Callable

from wuwei.core.errors import WuweiError
from wuwei.core.message import ToolCall, ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.tools import ToolExecutor


class ToolApprovalRejected(WuweiError):
    """工具审批被拒绝（仅在 raise_on_reject=True 时抛出）"""
    pass


def _rejection_message(tool_call: ToolCall, reason: str) -> ToolMessage:
    """构建回喂给模型的结构化拒绝结果。"""
    return ToolMessage(
        content=json.dumps(
            {
                "ok": False,
                "error": {
                    "type": "approval_rejected",
                    "message": f"工具 {tool_call.function.name} 被拒绝执行：{reason}",
                    "hint": "请不要重复执行被拒绝的操作，向用户说明情况或改用其他方案。",
                },
            },
            ensure_ascii=False,
        ),
        tool_call_id=tool_call.id,
        name=tool_call.function.name,
        status="error",
    )


class HitlMiddleware(Middleware):
    """Human-in-the-Loop 中间件

    在工具执行前请求用户审批。

    示例：
        async def approval_provider(tool_call: ToolCall) -> bool:
            print(f"是否允许执行 {tool_call.function.name}? (y/n)")
            return input().lower() == "y"

        middleware = HitlMiddleware(approval_provider)

    Args:
        approval_provider: 审批提供者，接收 ToolCall 返回是否批准
        auto_approve_tools: 自动批准的工具列表（优先级最高）
        auto_reject_tools: 自动拒绝的工具列表
        approval_policy: 审批范围——
            "all": 所有工具都询问 approval_provider（默认）
            "declared": 仅 execution.requires_approval=True 的工具询问
        raise_on_reject: 拒绝时抛异常而非回喂（默认 False）
    """

    is_tool_guard = True

    def __init__(
        self,
        approval_provider: Callable[[ToolCall], Awaitable[bool]] | None = None,
        auto_approve_tools: list[str] | None = None,
        auto_reject_tools: list[str] | None = None,
        approval_policy: str = "all",
        raise_on_reject: bool = False,
        mode: str = "callback",
    ):
        """
        Args:
            approval_provider: 审批提供者（callback 模式必填；interrupt 模式可省略）
            approval_policy: "all" 全部询问 | "declared" 仅 requires_approval 工具
            mode: "callback" 同步回调审批；"interrupt" 挂起等待 Agent.resume()
        """
        if approval_policy not in ("all", "declared"):
            raise ValueError(f"approval_policy 只支持 all/declared，收到: {approval_policy}")
        if mode not in ("callback", "interrupt"):
            raise ValueError(f"mode 只支持 callback/interrupt，收到: {mode}")
        if mode == "callback" and approval_provider is None:
            raise ValueError("callback 模式必须提供 approval_provider")
        self.approval_provider = approval_provider
        self.auto_approve_tools = auto_approve_tools or []
        self.auto_reject_tools = auto_reject_tools or []
        self.approval_policy = approval_policy
        self.mode = mode
        self.raise_on_reject = raise_on_reject
        self._executor: ToolExecutor | None = None

    def _tool_requires_approval(self, tool_call: ToolCall, ctx=None) -> bool:
        """查询工具声明位（惰性绑定 registry）。"""
        executor = ctx.config.get("tool_executor") if ctx is not None else self._executor
        if executor is None:
            return False
        tool = executor.registry.get(tool_call.function.name)
        return bool(tool and tool.execution.requires_approval)

    def bind_executor(self, executor: ToolExecutor) -> None:
        """绑定 ToolExecutor 以读取工具的 requires_approval 声明。"""
        self._executor = executor

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall | ToolMessage:
        """工具执行前请求审批。拒绝时返回 ToolMessage 回喂模型。"""
        tool_name = tool_call.function.name

        # 自动批准（白名单优先级最高）
        if tool_name in self.auto_approve_tools:
            return tool_call

        needs_approval = tool_name in self.auto_reject_tools or (
            self.approval_policy == "all"
            or self._tool_requires_approval(tool_call, ctx)
        )
        if not needs_approval:
            return tool_call

        # 自动拒绝
        if tool_name in self.auto_reject_tools:
            return self._reject(
                tool_call, f"工具 {tool_name} 在自动拒绝名单中"
            )

        # A decision authorizes exactly this call and argument revision, not its siblings.
        decision = ctx.metadata.get("approval_decisions", {}).get(tool_call.id)
        if decision and decision.get("name") == tool_name and decision.get("arguments") == tool_call.function.arguments:
            if decision["approved"]:
                return tool_call
            return self._reject(tool_call, decision.get("reason") or "用户拒绝了本次执行")

        # interrupt 模式：挂起等待外部恢复（需要 Agent 配置 checkpointer）
        if self.mode == "interrupt":
            if not ctx.config.get("checkpointer"):
                raise ValueError("interrupt mode requires checkpointer")
            from wuwei.runtime.interrupt import InterruptSignal

            raise InterruptSignal(tool_call=tool_call, reason=f"工具 {tool_name} 等待人工审批")

        # 请求用户审批
        try:
            approved = await self.approval_provider(tool_call)
        except Exception as exc:
            # 审批通道本身故障：fail-safe 拒绝并回喂
            return self._reject(tool_call, f"审批流程异常：{exc}")

        if not approved:
            return self._reject(tool_call, "用户拒绝了本次执行")

        return tool_call

    def _reject(self, tool_call: ToolCall, reason: str) -> ToolMessage:
        if self.raise_on_reject:
            raise ToolApprovalRejected(reason, details={"tool_name": tool_call.function.name})
        return _rejection_message(tool_call, reason)
