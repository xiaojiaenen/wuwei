"""核心模块测试"""

import pytest
from wuwei.core import (
    Runnable,
    RunnableSequence,
    RunnableConfig,
    BaseMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    ToolCall,
    FunctionCall,
    WuweiError,
    ToolError,
)


class TestRunnable:
    """Runnable 测试"""

    def test_runnable_sequence(self):
        """测试 Runnable 序列"""

        class UpperRunnable(Runnable):
            async def invoke(self, input, config=None):
                return input.upper()

        class ExclaimRunnable(Runnable):
            async def invoke(self, input, config=None):
                return f"{input}!"

        upper = UpperRunnable()
        exclaim = ExclaimRunnable()

        # 使用 |
        chain = upper | exclaim
        assert isinstance(chain, RunnableSequence)
        assert len(chain.runnables) == 2

    def test_runnable_config(self):
        """测试 RunnableConfig"""
        config = RunnableConfig(
            tags=["test"],
            metadata={"key": "value"},
        )
        assert config.tags == ["test"]
        assert config.metadata == {"key": "value"}


class TestMessages:
    """消息类型测试"""

    def test_human_message(self):
        """测试 HumanMessage"""
        msg = HumanMessage(content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.to_openai() == {"role": "user", "content": "hello"}

    def test_ai_message(self):
        """测试 AIMessage"""
        msg = AIMessage(content="world")
        assert msg.role == "assistant"
        assert msg.content == "world"

    def test_tool_call(self):
        """测试 ToolCall"""
        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="search", arguments={"query": "test"}),
        )
        assert tc.id == "call_123"
        openai_format = tc.to_openai()
        assert openai_format["id"] == "call_123"
        assert openai_format["function"]["name"] == "search"

    def test_tool_message(self):
        """测试 ToolMessage"""
        msg = ToolMessage(
            content="result",
            tool_call_id="call_123",
            name="search",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"


class TestErrors:
    """错误类型测试"""

    def test_wuwei_error(self):
        """测试 WuweiError"""
        err = WuweiError("test error", details={"key": "value"})
        assert str(err) == "test error"
        assert err.details == {"key": "value"}

    def test_tool_error(self):
        """测试 ToolError"""
        err = ToolError("tool failed")
        assert isinstance(err, WuweiError)
