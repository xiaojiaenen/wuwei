"""日志中间件"""

import logging
from typing import Optional
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import AIMessage, ToolCall, ToolMessage


logger = logging.getLogger("wuwei.middleware")


class LoggingMiddleware(Middleware):
    """日志中间件

    记录 LLM 调用和工具执行的日志。
    """

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """记录 LLM 调用前"""
        logger.log(
            self.log_level,
            f"[Step {ctx.step}] LLM 调用 - 消息数: {len(ctx.state.messages)}",
        )
        return ctx

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """记录 LLM 调用后"""
        tool_count = len(response.tool_calls) if response.tool_calls else 0
        logger.log(
            self.log_level,
            f"[Step {ctx.step}] LLM 响应 - 内容长度: {len(response.content)}, "
            f"工具调用: {tool_count}",
        )
        return ctx

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """记录工具执行前"""
        logger.log(
            self.log_level,
            f"[Step {ctx.step}] 工具调用: {tool_call.function.name}",
        )
        return tool_call

    async def after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """记录工具执行后"""
        logger.log(
            self.log_level,
            f"[Step {ctx.step}] 工具结果: {tool_message.name} "
            f"({tool_message.status})",
        )
        return tool_message
