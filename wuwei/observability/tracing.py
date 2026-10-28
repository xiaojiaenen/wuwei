"""追踪中间件

v3.1 修复：
- 使用 start_as_current_span 上下文管理器 → span 有上下文传播，
  能形成调用树（旧实现全是平级根 span）
- 不再无条件全局 set_tracer_provider（避免多实例互踩），
  仅在尚无 provider 时注册，console exporter 默认关闭
- span 生命周期用 try/finally 保证异常路径也会结束
"""

import logging

from wuwei.core.message import AIMessage, ToolCall, ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext

logger = logging.getLogger("wuwei.observability")


class TracingMiddleware(Middleware):
    """追踪中间件

    记录 LLM 调用和工具执行的追踪信息。
    安装了 OpenTelemetry 时导出标准 span（可接 Langfuse / Logfire / Jaeger）；
    未安装时退化为结构化日志。

    示例：
        middleware = TracingMiddleware(service_name="my-agent")
    """

    def __init__(self, service_name: str = "wuwei", console_exporter: bool = False):
        self.service_name = service_name
        self._tracer = None

        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            if provider is None or type(provider).__name__ == "ProxyTracerProvider":
                from opentelemetry.sdk.trace import TracerProvider

                new_provider = TracerProvider()
                if console_exporter:
                    from opentelemetry.sdk.trace.export import (
                        ConsoleSpanExporter,
                        SimpleSpanProcessor,
                    )

                    new_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(new_provider)
                provider = new_provider
            self._tracer = provider.get_tracer(service_name)
            logger.info("[%s] OpenTelemetry tracing enabled", service_name)
        except ImportError:
            logger.debug("OpenTelemetry 未安装，使用日志追踪")

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """开始 LLM 调用追踪"""
        if self._tracer:
            cm = self._tracer.start_as_current_span("llm_call")
            cm_span = cm.__enter__()
            self._resources(ctx)["llm"] = (cm, cm_span)
        else:
            msg_count = (
                len(ctx.state.messages) if ctx.state is not None and ctx.state.messages else 0
            )
            logger.info("[%s] LLM 调用开始 - 消息数: %s", self.service_name, msg_count)
        return ctx

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """结束 LLM 调用追踪"""
        resource = self._resources(ctx).pop("llm", None)
        span_cm, span = resource if resource else (None, None)
        if span_cm is not None:
            try:
                if span is not None:
                    span.set_attribute("llm.content_length", len(response.content))
                    span.set_attribute(
                        "llm.tool_calls_count",
                        len(response.tool_calls) if response.tool_calls else 0,
                    )
                    usage = getattr(response, "usage", None)
                    if usage:
                        span.set_attribute("llm.tokens_used", usage.get("total_tokens", 0))
            finally:
                span_cm.__exit__(None, None, None)
        else:
            logger.info(
                "[%s] LLM 调用完成 - 内容长度: %s",
                self.service_name,
                len(response.content),
            )
        return ctx

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """开始工具执行追踪"""
        if self._tracer:
            cm = self._tracer.start_as_current_span(f"tool_{tool_call.function.name}")
            cm_span = cm.__enter__()
            self._resources(ctx)[f"tool_{tool_call.id}"] = (cm, cm_span)
        else:
            logger.info("[%s] 工具调用: %s", self.service_name, tool_call.function.name)
        return tool_call

    async def after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """结束工具执行追踪"""
        key = f"tool_{tool_message.tool_call_id}"
        resource = self._resources(ctx).pop(key, None)
        span_cm, span = resource if resource else (None, None)
        if span_cm is not None:
            try:
                if span is not None:
                    status = getattr(tool_message, "status", "success")
                    span.set_attribute("tool.status", status)
            finally:
                span_cm.__exit__(None, None, None)
        else:
            logger.info(
                "[%s] 工具完成: %s (%s)",
                self.service_name,
                tool_message.name,
                getattr(tool_message, "status", "unknown"),
            )
        return tool_message

    def _resources(self, ctx):
        return ctx.config.setdefault("_runtime_resources", {}).setdefault(("tracing", id(self)), {})

    async def aclose(self, ctx):
        """Close transient spans before terminal, interrupt or cancellation checkpoint."""
        resources = self._resources(ctx)
        while resources:
            _, (cm, _) = resources.popitem()
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to close tracing span")

    async def on_error(self, ctx, error):
        await self.aclose(ctx)
        return error


def trace_current_span(span_cm):
    """从 start_as_current_span 的上下文管理器取当前 span（尽力而为）。"""
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except ImportError:  # pragma: no cover
        return None
