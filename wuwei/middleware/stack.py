"""中间件栈管理"""

from typing import Optional
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import AIMessage, ToolCall, ToolMessage


class MiddlewareStack:
    """中间件栈管理器

    按顺序执行多个中间件。

    示例：
        stack = MiddlewareStack()
        stack.add(MemoryMiddleware())
        stack.add(LoggingMiddleware())
        stack.add(HitlMiddleware())

        ctx = await stack.execute_before_llm(ctx)
    """

    def __init__(self):
        self.middlewares: list[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewareStack":
        """添加中间件"""
        self.middlewares.append(middleware)
        return self

    def remove(self, middleware: Middleware) -> "MiddlewareStack":
        """移除中间件"""
        self.middlewares.remove(middleware)
        return self

    def clear(self) -> "MiddlewareStack":
        """清空中间件"""
        self.middlewares.clear()
        return self

    async def execute_before_llm(
        self,
        ctx: MiddlewareContext,
    ) -> MiddlewareContext:
        """执行所有 before_llm 中间件"""
        for mw in self.middlewares:
            ctx = await mw.before_llm(ctx)
        return ctx

    async def execute_after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """执行所有 after_llm 中间件"""
        for mw in self.middlewares:
            ctx = await mw.after_llm(ctx, response)
        return ctx

    async def execute_before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """执行所有 before_tool 中间件"""
        for mw in self.middlewares:
            tool_call = await mw.before_tool(ctx, tool_call)
        return tool_call

    async def execute_after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """执行所有 after_tool 中间件"""
        for mw in self.middlewares:
            tool_message = await mw.after_tool(ctx, tool_message)
        return tool_message

    async def execute_on_error(
        self,
        ctx: MiddlewareContext,
        error: Exception,
    ) -> Optional[Exception]:
        """执行错误处理中间件"""
        for mw in self.middlewares:
            result = await mw.on_error(ctx, error)
            if result is None:
                return None  # 已处理
            error = result
        return error

    def __len__(self) -> int:
        return len(self.middlewares)

    def __repr__(self) -> str:
        names = [mw.__class__.__name__ for mw in self.middlewares]
        return f"MiddlewareStack({', '.join(names)})"
