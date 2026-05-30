"""中间件模块测试"""

import pytest
from wuwei.middleware import Middleware, MiddlewareContext, MiddlewareStack
from wuwei.middleware.logging import LoggingMiddleware
from wuwei.middleware.hitl import HitlMiddleware, ToolApprovalRejected
from wuwei.graph.state import State
from wuwei.core.message import AIMessage, ToolCall, FunctionCall, ToolMessage


class TestMiddleware:
    """Middleware 测试"""

    @pytest.mark.asyncio
    async def test_middleware_lifecycle(self):
        """测试中间件生命周期"""

        class TestMiddleware(Middleware):
            def __init__(self):
                self.calls = []

            async def before_llm(self, ctx):
                self.calls.append("before_llm")
                return ctx

            async def after_llm(self, ctx, response):
                self.calls.append("after_llm")
                return ctx

            async def before_tool(self, ctx, tool_call):
                self.calls.append("before_tool")
                return tool_call

            async def after_tool(self, ctx, tool_message):
                self.calls.append("after_tool")
                return tool_message

        mw = TestMiddleware()
        ctx = MiddlewareContext(state=State())

        ctx = await mw.before_llm(ctx)
        ctx = await mw.after_llm(ctx, AIMessage(content="test"))
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments={}),
        )
        await mw.before_tool(ctx, tool_call)
        tool_msg = ToolMessage(
            content="result",
            tool_call_id="call_123",
            name="test",
        )
        await mw.after_tool(ctx, tool_msg)

        assert mw.calls == [
            "before_llm",
            "after_llm",
            "before_tool",
            "after_tool",
        ]


class TestMiddlewareStack:
    """MiddlewareStack 测试"""

    @pytest.mark.asyncio
    async def test_add_and_execute(self):
        """测试添加和执行中间件"""

        class CounterMiddleware(Middleware):
            def __init__(self, name):
                self.name = name
                self.calls = []

            async def before_llm(self, ctx):
                self.calls.append(self.name)
                return ctx

        stack = MiddlewareStack()
        mw1 = CounterMiddleware("mw1")
        mw2 = CounterMiddleware("mw2")

        stack.add(mw1)
        stack.add(mw2)

        ctx = MiddlewareContext(state=State())
        ctx = await stack.execute_before_llm(ctx)

        assert mw1.calls == ["mw1"]
        assert mw2.calls == ["mw2"]
        assert len(stack) == 2

    @pytest.mark.asyncio
    async def test_remove(self):
        """测试移除中间件"""

        class TestMiddleware(Middleware):
            pass

        stack = MiddlewareStack()
        mw = TestMiddleware()
        stack.add(mw)
        assert len(stack) == 1

        stack.remove(mw)
        assert len(stack) == 0

    @pytest.mark.asyncio
    async def test_clear(self):
        """测试清空中间件"""

        class TestMiddleware(Middleware):
            pass

        stack = MiddlewareStack()
        stack.add(TestMiddleware())
        stack.add(TestMiddleware())
        assert len(stack) == 2

        stack.clear()
        assert len(stack) == 0


class TestLoggingMiddleware:
    """LoggingMiddleware 测试"""

    @pytest.mark.asyncio
    async def test_logging(self):
        """测试日志中间件"""
        import logging

        # 创建一个测试 handler
        test_handler = logging.StreamHandler()
        test_handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("wuwei.middleware")
        logger.addHandler(test_handler)
        logger.setLevel(logging.DEBUG)

        mw = LoggingMiddleware(log_level=logging.DEBUG)
        ctx = MiddlewareContext(state=State())

        ctx = await mw.before_llm(ctx)
        ctx = await mw.after_llm(ctx, AIMessage(content="test"))

        # 清理
        logger.removeHandler(test_handler)


class TestHitlMiddleware:
    """HitlMiddleware 测试"""

    @pytest.mark.asyncio
    async def test_auto_approve(self):
        """测试自动批准"""
        approval_calls = []

        async def approval_provider(tool_call):
            approval_calls.append(tool_call)
            return True

        mw = HitlMiddleware(
            approval_provider=approval_provider,
            auto_approve_tools=["safe_tool"],
        )
        ctx = MiddlewareContext(state=State())

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="safe_tool", arguments={}),
        )

        result = await mw.before_tool(ctx, tool_call)
        assert result == tool_call
        assert len(approval_calls) == 0  # 自动批准，不调用审批

    @pytest.mark.asyncio
    async def test_auto_reject(self):
        """测试自动拒绝"""
        mw = HitlMiddleware(
            approval_provider=lambda tc: True,
            auto_reject_tools=["dangerous_tool"],
        )
        ctx = MiddlewareContext(state=State())

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="dangerous_tool", arguments={}),
        )

        with pytest.raises(ToolApprovalRejected):
            await mw.before_tool(ctx, tool_call)

    @pytest.mark.asyncio
    async def test_user_approval(self):
        """测试用户审批"""

        async def approval_provider(tool_call):
            return True  # 批准

        mw = HitlMiddleware(approval_provider=approval_provider)
        ctx = MiddlewareContext(state=State())

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test_tool", arguments={}),
        )

        result = await mw.before_tool(ctx, tool_call)
        assert result == tool_call

    @pytest.mark.asyncio
    async def test_user_rejection(self):
        """测试用户拒绝"""

        async def approval_provider(tool_call):
            return False  # 拒绝

        mw = HitlMiddleware(approval_provider=approval_provider)
        ctx = MiddlewareContext(state=State())

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test_tool", arguments={}),
        )

        with pytest.raises(ToolApprovalRejected):
            await mw.before_tool(ctx, tool_call)
