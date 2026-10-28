"""事件总线 — 统一的事件出口

AgentRunner 产生的每个 AgentEvent 在 yield 的同时发布到总线，
订阅者（控制台美化输出、OTel exporter、前端 WebSocket 等）并行消费，
互不影响主流程。订阅者异常只记日志，不会打断 agent 运行。

示例：
    bus = EventBus()
    bus.subscribe(ConsoleReporter(verbose=True))
    agent = Agent(llm=llm, event_bus=bus)

    @bus.subscribe
    async def collect(event: AgentEvent):
        metrics.record(event.type)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from wuwei.llm import AgentEvent

logger = logging.getLogger("wuwei.events")

Handler = Callable[[AgentEvent], Any]


class EventBus:
    """同步/异步订阅者统一的事件总线（fire-and-forget）。"""

    def __init__(self) -> None:
        self._handlers: list[Handler] = []
        self._tasks: set[asyncio.Task] = set()
        self._tail = None
        self._closed = False

    def subscribe(self, handler: Handler) -> Callable[[], None]:
        """注册订阅者（支持 sync 与 async），返回取消订阅函数。"""
        if self._closed:
            raise RuntimeError("EventBus is closed")
        self._handlers.append(handler)

        def _unsub() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return _unsub

    async def publish(self, event: AgentEvent) -> None:
        """顺序投递给所有订阅者；单个订阅者异常不影响其它订阅者。"""
        if self._closed:
            raise RuntimeError("EventBus is closed")
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except Exception:
                logger.exception("事件订阅者处理失败: %s", event.type)

    def publish_nowait(self, event: AgentEvent) -> None:
        """调度异步投递（不阻塞调用方）。无运行中的事件循环时同步执行。"""
        if self._closed:
            raise RuntimeError("EventBus is closed")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # 无事件循环（同步上下文）：直接同步消费
            for handler in list(self._handlers):
                try:
                    if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(
                        type(handler).__call__
                    ):
                        logger.warning("同步上下文中跳过异步订阅者: %s", handler)
                        continue
                    result = handler(event)
                    if inspect.iscoroutine(result):
                        result.close()
                        logger.warning("同步上下文中跳过异步订阅者: %s", handler)
                except Exception:
                    logger.exception("事件订阅者处理失败: %s", event.type)
            return
        previous = self._tail

        async def ordered():
            if previous is not None:
                await previous
            await self._safe_publish(event)

        task = loop.create_task(ordered())
        self._tail = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self):
        """Wait for every currently queued delivery, including deliveries queued by handlers."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self):
        await self.drain()
        self._closed = True
        self._tail = None
        self._handlers.clear()

    async def _safe_publish(self, event: AgentEvent) -> None:
        try:
            await self.publish(event)
        except Exception:
            logger.exception("事件投递失败: %s", event.type)


class ConsoleReporter:
    """控制台美化输出订阅者（对标 Claude Code 的终端体验）。"""

    _ICONS = {
        "run_start": "▶",
        "run_end": "■",
        "llm_start": "🧠",
        "llm_end": "✓",
        "text_delta": None,  # 流式文本单独处理
        "tool_start": "🔧",
        "tool_end": "✓",
        "tool_error": "✗",
        "approval_required": "⏸",
        "error": "✗",
        "done": "✔",
    }

    def __init__(self, verbose: bool = False, stream_text: bool = True) -> None:
        self.verbose = verbose
        self.stream_text = stream_text
        self._in_text = False

    def __call__(self, event: AgentEvent) -> None:
        icon = self._ICONS.get(event.type, "·")
        if event.type == "text_delta":
            if self.stream_text:
                print(event.data.get("content", ""), end="", flush=True)
                self._in_text = True
            return
        if self._in_text:
            print()
            self._in_text = False
        if icon is None:
            return
        detail = ""
        if event.type in ("tool_start", "tool_end", "tool_error"):
            detail = str(event.data.get("tool_name", ""))
        elif event.type == "approval_required":
            detail = f"{event.data.get('tool_name')} 等待审批"
        elif event.type == "run_end":
            usage = event.data.get("usage") or {}
            detail = f"tokens={usage.get('total_tokens', 0)} status={event.data.get('status', '')}"
        elif event.type == "error":
            detail = str(event.data.get("message", ""))
        elif self.verbose:
            detail = str(event.data)
        suffix = f" {detail}" if detail else ""
        print(f"{icon} [{event.type}]{suffix}")
