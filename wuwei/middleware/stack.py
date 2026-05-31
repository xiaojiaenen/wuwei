"""中间件栈管理

借鉴 DeepAgents 和 AgentScope 的设计：
- 支持预分拣：初始化时检测每个中间件实现了哪些钩子，
  未实现钩子的中间件在对应执行路径中零开销
- wrap_model_call：洋葱模式包装 LLM 调用
"""

from typing import Any, Callable, Optional

from wuwei.core.message import AIMessage, ToolCall, ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext


class MiddlewareStack:
    """中间件栈管理器

    预分拣机制：添加中间件时检测其实现的钩子，
    按钩子类型维护分拣列表，避免遍历未实现该钩子的中间件。

    wrap_model_call 钩子：使用洋葱模型递归包装，
    内层中间件包装外层。

    示例：
        stack = MiddlewareStack()
        stack.add(MemoryMiddleware())
        stack.add(LoggingMiddleware())
        stack.add(HitlMiddleware())

        ctx = await stack.execute_before_llm(ctx)
    """

    def __init__(self):
        self.middlewares: list[Middleware] = []
        # 预分拣列表：只包含实现了对应钩子的中间件
        self._before_llm_mws: list[Middleware] = []
        self._after_llm_mws: list[Middleware] = []
        self._before_tool_mws: list[Middleware] = []
        self._after_tool_mws: list[Middleware] = []
        self._on_error_mws: list[Middleware] = []
        self._wrap_model_call_mws: list[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewareStack":
        """添加中间件并自动分拣

        检测中间件是否覆写了各个钩子，将其加入对应分拣列表。
        """
        self.middlewares.append(middleware)

        # 检测钩子实现（通过检查子类是否覆写了基类方法）
        cls = type(middleware)
        base = Middleware

        if cls.before_llm is not base.before_llm:
            self._before_llm_mws.append(middleware)
        if cls.after_llm is not base.after_llm:
            self._after_llm_mws.append(middleware)
        if cls.before_tool is not base.before_tool:
            self._before_tool_mws.append(middleware)
        if cls.after_tool is not base.after_tool:
            self._after_tool_mws.append(middleware)
        if cls.on_error is not base.on_error:
            self._on_error_mws.append(middleware)
        if cls.wrap_model_call is not base.wrap_model_call:
            self._wrap_model_call_mws.append(middleware)

        return self

    def remove(self, middleware: Middleware) -> "MiddlewareStack":
        """移除中间件"""
        self.middlewares.remove(middleware)
        self._before_llm_mws = [m for m in self._before_llm_mws if m is not middleware]
        self._after_llm_mws = [m for m in self._after_llm_mws if m is not middleware]
        self._before_tool_mws = [m for m in self._before_tool_mws if m is not middleware]
        self._after_tool_mws = [m for m in self._after_tool_mws if m is not middleware]
        self._on_error_mws = [m for m in self._on_error_mws if m is not middleware]
        self._wrap_model_call_mws = [m for m in self._wrap_model_call_mws if m is not middleware]
        return self

    def clear(self) -> "MiddlewareStack":
        """清空所有中间件"""
        self.middlewares.clear()
        self._before_llm_mws.clear()
        self._after_llm_mws.clear()
        self._before_tool_mws.clear()
        self._after_tool_mws.clear()
        self._on_error_mws.clear()
        self._wrap_model_call_mws.clear()
        return self

    # ── 标准钩子执行 ──────────────────────────────────────────────

    async def execute_before_llm(
        self,
        ctx: MiddlewareContext,
    ) -> MiddlewareContext:
        """执行所有 before_llm 中间件（仅已实现该钩子的）"""
        for mw in self._before_llm_mws:
            ctx = await mw.before_llm(ctx)
        return ctx

    async def execute_after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """执行所有 after_llm 中间件"""
        for mw in self._after_llm_mws:
            ctx = await mw.after_llm(ctx, response)
        return ctx

    async def execute_before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> Optional[ToolCall]:
        """执行所有 before_tool 中间件

        返回 None 表示中间件拦截了该工具调用。
        """
        for mw in self._before_tool_mws:
            result = await mw.before_tool(ctx, tool_call)
            if result is None:
                return None
            tool_call = result
        return tool_call

    async def execute_after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """执行所有 after_tool 中间件"""
        for mw in self._after_tool_mws:
            tool_message = await mw.after_tool(ctx, tool_message)
        return tool_message

    async def execute_on_error(
        self,
        ctx: MiddlewareContext,
        error: Exception,
    ) -> Optional[Exception]:
        """执行错误处理中间件

        返回 None 表示错误已被处理。
        """
        for mw in self._on_error_mws:
            result = await mw.on_error(ctx, error)
            if result is None:
                return None
            error = result
        return error

    # ── wrap_model_call ─────────────────────────────────────────────

    async def execute_wrap_model_call(
        self,
        ctx: MiddlewareContext,
        messages: list[Any],
        tools: list[Any],
        model_caller: Callable,
    ) -> Any:
        """执行 wrap_model_call 洋葱模型

        使用递归洋葱模式包装 LLM 调用：
        最外层的中间件最先执行 wrapping 逻辑，
        最内层的中间件最后执行 wrapping 逻辑。

        Args:
            ctx: 中间件上下文
            messages: 待发送的消息列表
            tools: 可用工具列表
            model_caller: 实际的 LLM 调用函数 async (messages, tools) -> response

        Returns:
            LLM 响应
        """
        if not self._wrap_model_call_mws:
            return await model_caller(messages, tools)

        # 构建洋葱链
        async def _call_chain(
            index: int,
            _messages: list[Any],
            _tools: list[Any],
        ) -> Any:
            if index >= len(self._wrap_model_call_mws):
                return await model_caller(_messages, _tools)

            mw = self._wrap_model_call_mws[index]

            async def _next_handler(_m: list[Any], _t: list[Any]) -> Any:
                return await _call_chain(index + 1, _m, _t)

            return await mw.wrap_model_call(ctx, _messages, _tools, _next_handler)

        return await _call_chain(0, messages, tools)

    # ── utility ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.middlewares)

    def __repr__(self) -> str:
        names = [mw.__class__.__name__ for mw in self.middlewares]
        return f"MiddlewareStack({', '.join(names)})"
