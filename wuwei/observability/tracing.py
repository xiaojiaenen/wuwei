"""追踪中间件

v3.1 修复：
- 使用 start_as_current_span 上下文管理器 → span 有上下文传播，
  能形成调用树（旧实现全是平级根 span）
- 不再无条件全局 set_tracer_provider（避免多实例互踩），
  仅在尚无 provider 时注册，console exporter 默认关闭
- span 生命周期用 try/finally 保证异常路径也会结束
"""

import logging
import re
import time
from collections import deque

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

    def __init__(
        self,
        service_name: str = "wuwei",
        console_exporter: bool = False,
        *,
        capture_content: bool = False,
        content_limit: int = 512,
        record_limit: int = 1000,
    ):
        self.service_name = service_name
        self._tracer = None
        self.capture_content = capture_content
        self.content_limit = max(0, content_limit)
        self.records = deque(maxlen=max(1, record_limit))

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

    def _facts(self, ctx):
        state = ctx.config.get("run_state") or ctx.metadata or {}
        if hasattr(state, "model_dump"):
            state = state.model_dump()
        facts = {"schema_version": "wuwei.trace.v1"}
        for key in ("run_id", "session_id", "parent_run_id", "trace_id", "provider", "model"):
            value = state.get(key, ctx.config.get(key))
            if value is not None:
                facts[key] = value
        session = ctx.config.get("session")
        if session is not None:
            facts.setdefault("session_id", session.session_id)
        parent = ctx.config.get("parent_context")
        if parent:
            parent_id = (
                parent.get("run_id")
                if isinstance(parent, dict)
                else (getattr(parent, "metadata", {}) or {}).get(
                    "run_id", getattr(parent, "run_id", None)
                )
            )
            if parent_id:
                facts.setdefault("parent_run_id", parent_id)
        return facts

    def _record(self, ctx, event, **values):
        record = {**self._facts(ctx), "event": event, **values}
        self.records.append(record)
        logger.info("wuwei.trace %s", record)
        return record

    def _content(self, text):
        # Opt-in content still removes common secret assignments and bearer tokens.
        text = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", str(text))
        text = re.sub(
            r"(?i)(api[_-]?key|password|token|secret)([\s\"':=]+)[^,\s}]+", r"\1\2[REDACTED]", text
        )
        return text[: self.content_limit]

    async def before_run(self, ctx):
        self._resources(ctx)["started"] = time.monotonic()
        if self._tracer:
            cm = self._tracer.start_as_current_span("agent_run")
            span = cm.__enter__()
            context_fn = getattr(span, "get_span_context", None)
            if callable(context_fn):
                span_context = context_fn()
                if getattr(span_context, "is_valid", False):
                    ctx.metadata["trace_id"] = f"{span_context.trace_id:032x}"
            for key, value in self._facts(ctx).items():
                span.set_attribute(key, value)
            self._resources(ctx)["run"] = (cm, span)
        self._record(ctx, "run.started")
        return ctx

    async def after_run(self, ctx, result):
        get = (
            result.get
            if isinstance(result, dict)
            else lambda key, default=None: getattr(result, key, default)
        )
        started = self._resources(ctx).pop("started", time.monotonic())
        values = {
            "status": get("status", "unknown"),
            "termination_reason": get("termination_reason"),
            "duration_ms": (time.monotonic() - started) * 1000,
            "usage": dict(get("usage") or {}),
            "usage_available": get("usage_available", bool(get("usage"))),
        }
        if self.capture_content:
            values["content"] = self._content(get("content", ""))
        self._record(ctx, "run.finished", **values)
        resource = self._resources(ctx).pop("run", None)
        if resource:
            cm, span = resource
            span.set_attribute("run.status", values["status"])
            cm.__exit__(None, None, None)
        return ctx

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
        facts = {"usage": dict(getattr(response, "usage", None) or {})}
        for key in ("provider", "model", "retry_count", "usage_available"):
            value = getattr(response, key, None)
            if value is not None:
                facts[key] = value
        metadata = getattr(response, "metadata", {}) or {}
        for key in ("retry_count", "provider", "model", "usage_available"):
            if key in metadata:
                facts[key] = metadata[key]
        self._record(ctx, "llm.finished", **facts)
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
                    for key in ("provider", "model", "retry_count", "usage_available"):
                        if key in facts:
                            span.set_attribute(f"llm.{key}", facts[key])
                    for key, value in facts["usage"].items():
                        if isinstance(value, (int, float)):
                            span.set_attribute(f"llm.usage.{key}", value)
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
        outcome = (getattr(tool_message, "metadata", {}) or {}).get("tool_outcome", {})
        status = outcome.get("status", getattr(tool_message, "status", "unknown"))
        self._record(
            ctx,
            "tool.finished",
            tool_name=tool_message.name,
            tool_call_id=tool_message.tool_call_id,
            status=status,
            executed=outcome.get("executed"),
        )
        key = f"tool_{tool_message.tool_call_id}"
        resource = self._resources(ctx).pop(key, None)
        span_cm, span = resource if resource else (None, None)
        if span_cm is not None:
            try:
                if span is not None:
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
        resources.pop("started", None)
        while resources:
            _, (cm, _) = resources.popitem()
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to close tracing span")

    async def on_error(self, ctx, error):
        self._record(ctx, "run.error", error_type=type(error).__name__)
        for value in self._resources(ctx).values():
            if isinstance(value, tuple):
                _, span = value
                # Do not attach exception text: it may contain secrets.
                try:
                    from opentelemetry.trace import Status, StatusCode

                    if hasattr(span, "set_status"):
                        span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                except ImportError:
                    pass
        await self.aclose(ctx)
        return error


def trace_current_span(span_cm):
    """从 start_as_current_span 的上下文管理器取当前 span（尽力而为）。"""
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except ImportError:  # pragma: no cover
        return None
