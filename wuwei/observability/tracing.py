"""追踪中间件"""

import logging
from typing import Optional
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import AIMessage, ToolCall, ToolMessage


logger = logging.getLogger("wuwei.observability")


class TracingMiddleware(Middleware):
    """追踪中间件

    记录 LLM 调用和工具执行的追踪信息。
    可以集成 OpenTelemetry 或其他追踪系统。

    示例：
        middleware = TracingMiddleware(service_name="my-agent")
    """

    def __init__(self, service_name: str = "wuwei"):
        self.service_name = service_name
        self._tracer = None

        # 尝试初始化 OpenTelemetry
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider = TracerProvider()
            # 添加 Console exporter，span 数据输出到 stdout
            exporter = ConsoleSpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            logger.info(f"[{service_name}] OpenTelemetry tracing enabled (console exporter)")
        except ImportError:
            logger.debug(
                "OpenTelemetry 未安装，使用日志追踪"
            )

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """开始 LLM 调用追踪"""
        if self._tracer:
            span = self._tracer.start_span("llm_call")
            ctx.metadata["span"] = span
        else:
            logger.info(
                f"[{self.service_name}] LLM 调用开始 - "
                f"消息数: {len(ctx.state.messages)}"
            )
        return ctx

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """结束 LLM 调用追踪"""
        span = ctx.metadata.get("span")
        if span:
            span.set_attribute(
                "llm.content_length",
                len(response.content),
            )
            span.set_attribute(
                "llm.tool_calls_count",
                len(response.tool_calls) if response.tool_calls else 0,
            )
            if response.usage:
                span.set_attribute(
                    "llm.tokens_used",
                    response.usage.get("total_tokens", 0),
                )
            span.end()
        else:
            logger.info(
                f"[{self.service_name}] LLM 调用完成 - "
                f"内容长度: {len(response.content)}"
            )
        return ctx

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """开始工具执行追踪"""
        if self._tracer:
            span = self._tracer.start_span(
                f"tool_{tool_call.function.name}"
            )
            ctx.metadata["tool_span"] = span
        else:
            logger.info(
                f"[{self.service_name}] 工具调用: "
                f"{tool_call.function.name}"
            )
        return tool_call

    async def after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """结束工具执行追踪"""
        span = ctx.metadata.get("tool_span")
        if span:
            span.set_attribute(
                "tool.status",
                tool_message.status,
            )
            span.end()
        else:
            logger.info(
                f"[{self.service_name}] 工具完成: "
                f"{tool_message.name} ({tool_message.status})"
            )
        return tool_message
