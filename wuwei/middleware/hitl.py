"""Human-in-the-Loop 中间件"""

from typing import Callable, Awaitable, Optional
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import ToolCall
from wuwei.core.errors import WuweiError


class ToolApprovalRejected(WuweiError):
    """工具审批被拒绝"""
    pass


class HitlMiddleware(Middleware):
    """Human-in-the-Loop 中间件

    在工具执行前请求用户审批。

    示例：
        async def approval_provider(tool_call: ToolCall) -> bool:
            print(f"是否允许执行 {tool_call.function.name}? (y/n)")
            return input().lower() == "y"

        middleware = HitlMiddleware(approval_provider)
    """

    def __init__(
        self,
        approval_provider: Callable[[ToolCall], Awaitable[bool]],
        auto_approve_tools: list[str] = None,
        auto_reject_tools: list[str] = None,
    ):
        """
        Args:
            approval_provider: 审批提供者，接收 ToolCall 返回是否批准
            auto_approve_tools: 自动批准的工具列表
            auto_reject_tools: 自动拒绝的工具列表
        """
        self.approval_provider = approval_provider
        self.auto_approve_tools = auto_approve_tools or []
        self.auto_reject_tools = auto_reject_tools or []

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """工具执行前请求审批"""
        tool_name = tool_call.function.name

        # 自动批准
        if tool_name in self.auto_approve_tools:
            return tool_call

        # 自动拒绝
        if tool_name in self.auto_reject_tools:
            raise ToolApprovalRejected(
                f"工具 {tool_name} 被自动拒绝",
                details={"tool_name": tool_name},
            )

        # 请求用户审批
        approved = await self.approval_provider(tool_call)
        if not approved:
            raise ToolApprovalRejected(
                f"用户拒绝了工具 {tool_name} 的执行",
                details={"tool_name": tool_name},
            )

        return tool_call
