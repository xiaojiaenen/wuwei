"""中间件基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from wuwei.core.message import AIMessage, ToolCall, ToolMessage
from wuwei.graph.state import State


@dataclass
class MiddlewareContext:
    """中间件上下文

    在中间件栈中传递的数据容器。
    """
    state: State
    config: dict = field(default_factory=dict)
    step: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Middleware(ABC):
    """中间件基类

    借鉴 Deep Agents 的中间件栈设计。
    每个中间件可以拦截和修改请求/响应。

    生命周期：
    - before_llm: LLM 调用前
    - after_llm: LLM 调用后
    - before_tool: 工具执行前
    - after_tool: 工具执行后
    """

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前

        可以修改消息、工具列表等。
        """
        return ctx

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """LLM 调用后

        可以修改响应、记录日志等。
        """
        return ctx

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """工具执行前

        可以修改工具参数、拦截执行等。
        """
        return tool_call

    async def after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """工具执行后

        可以修改结果、记录日志等。
        """
        return tool_message

    async def on_error(
        self,
        ctx: MiddlewareContext,
        error: Exception,
    ) -> Optional[Exception]:
        """错误处理

        返回 None 表示已处理，否则继续抛出。
        """
        return error

    async def wrap_model_call(
        self,
        ctx: MiddlewareContext,
        messages: list[Any],
        tools: list[Any],
        next_handler: Any,
    ) -> Any:
        """包装 LLM 调用（洋葱模型）

        借鉴 DeepAgents 的 wrap_model_call 模式。
        中间件可以在 LLM 调用前后注入逻辑：
        - 修改 system prompt（注入 skill 定义、memory 等）
        - 过滤工具列表
        - 添加 Anthropic cache_control
        - 后处理响应

        使用方式：
            async def wrap_model_call(self, ctx, messages, tools, next_handler):
                # 前置处理
                messages = self._inject_context(messages)
                # 调用下一个中间件（或实际 LLM）
                response = await next_handler(messages, tools)
                # 后置处理
                return response

        Args:
            ctx: 中间件上下文
            messages: 消息列表
            tools: 工具列表
            next_handler: 下一层处理函数 async (messages, tools) -> response
        """
        return await next_handler(messages, tools)
