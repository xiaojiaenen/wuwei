"""新的 AgentRunner - 使用 Middleware 替代 Hook"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from wuwei.agent.session import AgentSession
from wuwei.llm import (
    AgentEvent,
    AgentRunResult,
    LLMGateway,
    LLMResponse,
    LLMResponseChunk,
    Message,
    ToolCall,
)
from wuwei.middleware.base import MiddlewareContext
from wuwei.middleware.stack import MiddlewareStack
from wuwei.tools import Tool, ToolExecutor

if TYPE_CHECKING:
    from wuwei.planning import Task


class AgentRunner:
    """单个 agent 会话的运行时执行器（使用 Middleware）。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool],
        tool_executor: ToolExecutor,
        session: AgentSession,
        middleware: MiddlewareStack | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = tool_executor
        self.session = session
        self.middleware = middleware or MiddlewareStack()

    async def run(
        self,
        user_input: str,
        stream: bool = False,
        task: "Task | None" = None,
    ):
        """执行一次 agent 运行。"""
        if stream:
            return self._run_stream(user_input, task=task)
        return await self._run_non_stream(user_input, task=task)

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _merge_usage(self, total: dict[str, int], usage: dict[str, int] | None) -> None:
        if not usage:
            return

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = total.get(key, 0) + usage.get(key, 0)

    def _set_session_run_stats(
        self,
        *,
        usage: dict[str, int],
        latency_ms: int,
        llm_calls: int,
    ) -> None:
        self.session.last_usage = dict(usage)
        self.session.last_latency_ms = latency_ms
        self.session.last_llm_calls = llm_calls

    def _build_run_result(
        self,
        *,
        content: str,
        usage: dict[str, int],
        latency_ms: int,
        llm_calls: int,
    ) -> AgentRunResult:
        self._set_session_run_stats(
            usage=usage,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
        )
        return AgentRunResult(
            content=content,
            usage=dict(usage),
            latency_ms=latency_ms,
            llm_calls=llm_calls,
        )

    def _build_event(
        self,
        event_type: str,
        *,
        step: int,
        run_id: str | None = None,
        data: dict | None = None,
    ) -> AgentEvent:
        return AgentEvent(
            type=event_type,
            session_id=self.session.session_id,
            step=step,
            run_id=run_id,
            data=data or {},
        )

    async def _emit_event(
        self,
        event_type: str,
        *,
        step: int,
        run_id: str | None = None,
        data: dict | None = None,
    ) -> AgentEvent:
        event = self._build_event(event_type, step=step, run_id=run_id, data=data)
        # Middleware 不直接支持 emit_event，跳过
        return event

    async def stream_events(
        self,
        user_input: str,
        *,
        task: "Task | None" = None,
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流的形式执行一次 agent 运行。"""
        step_count = 0
        llm_calls = 0
        total_latency_ms = 0
        total_usage = self._empty_usage()
        run_id = uuid4().hex
        context = self.session.context
        context.add_user_message(user_input)
        await self._emit_event("run_start", step=0, run_id=run_id, data={"input": user_input})

        try:
            while step_count < self.session.max_steps:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                full_tool_calls = None
                messages = self._copy_messages()
                tools = list(self.tools)

                # 使用 Middleware
                ctx = MiddlewareContext(
                    state=None,  # 简化：不使用 State
                    config={},
                    step=step_count,
                )
                ctx.state = type('State', (), {'messages': messages, 'metadata': {}})()
                ctx = await self.middleware.execute_before_llm(ctx)
                messages = ctx.state.messages

                llm_start = time.monotonic()
                await self._emit_event(
                    "llm_start",
                    step=step_count,
                    run_id=run_id,
                    data={"tools": [tool.name for tool in tools]},
                )
                stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                    messages,
                    tools=tools,
                    stream=True,
                )
                llm_calls += 1

                async for chunk in stream:
                    if chunk.reasoning_content:
                        reasoning_parts.append(chunk.reasoning_content)
                        event = await self._emit_event(
                            "reasoning_delta",
                            step=step_count,
                            run_id=run_id,
                            data={"content": chunk.reasoning_content},
                        )
                        yield event

                    if chunk.content:
                        content_parts.append(chunk.content)
                        event = await self._emit_event(
                            "text_delta",
                            step=step_count,
                            run_id=run_id,
                            data={"content": chunk.content},
                        )
                        yield event

                    self._merge_usage(total_usage, chunk.usage)

                    if chunk.tool_calls_complete:
                        full_tool_calls = chunk.tool_calls_complete

                total_latency_ms += int((time.monotonic() - llm_start) * 1000)
                await self._emit_event(
                    "llm_end",
                    step=step_count,
                    run_id=run_id,
                    data={
                        "latency_ms": total_latency_ms,
                        "usage": dict(total_usage),
                        "has_tool_calls": bool(full_tool_calls),
                    },
                )
                ai_message = context.add_ai_message(
                    "".join(content_parts),
                    tool_calls=full_tool_calls,
                    reasoning_content="".join(reasoning_parts) or None,
                )

                if full_tool_calls:
                    for tool_call in full_tool_calls:
                        tool = self.tool_executor.registry.get(tool_call.function.name)
                        event = self._build_event(
                            "tool_start",
                            step=step_count,
                            run_id=run_id,
                            data={
                                "tool_name": tool_call.function.name,
                                "display_name": tool.display_name if tool else None,
                                "args": tool_call.function.arguments,
                                "tool_call_id": tool_call.id,
                            },
                        )
                        yield event

                    tool_messages = await self._execute_tool_calls(
                        full_tool_calls,
                        step=step_count,
                        task=task,
                        run_id=run_id,
                    )
                    self._append_tool_messages(tool_messages)

                    for tool_call, tool_message in zip(
                        full_tool_calls,
                        tool_messages,
                        strict=False,
                    ):
                        event = self._build_event(
                            "tool_end",
                            step=step_count,
                            run_id=run_id,
                            data={
                                "tool_name": tool_call.function.name,
                                "tool_call_id": tool_call.id,
                                "output": tool_message.content,
                            },
                        )
                        yield event

                        error_message = self.tool_executor.extract_error_message(
                            tool_message.content
                        )
                        if error_message:
                            event = self._build_event(
                                "error",
                                step=step_count,
                                run_id=run_id,
                                data={
                                    "message": error_message,
                                    "tool_name": tool_call.function.name,
                                    "tool_call_id": tool_call.id,
                                },
                            )
                            yield event

                    step_count += 1
                    continue

                done_event = await self._emit_event(
                    "done",
                    step=step_count,
                    run_id=run_id,
                    data={
                        "usage": dict(total_usage),
                        "latency_ms": total_latency_ms,
                        "llm_calls": llm_calls,
                    },
                )
                yield done_event
                await self._emit_event(
                    "run_end", step=step_count, run_id=run_id, data=dict(done_event.data)
                )
                self._set_session_run_stats(
                    usage=total_usage,
                    latency_ms=total_latency_ms,
                    llm_calls=llm_calls,
                )
                return

            limit_message = "任务未完成，已达到最大步骤限制。"
            ai_message = context.add_ai_message(limit_message)
            text_event = await self._emit_event(
                "text_delta",
                step=step_count,
                run_id=run_id,
                data={"content": limit_message},
            )
            yield text_event
            done_event = await self._emit_event(
                "done",
                step=step_count,
                run_id=run_id,
                data={
                    "reason": "max_steps",
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
            )
            yield done_event
            await self._emit_event(
                "run_end", step=step_count, run_id=run_id, data=dict(done_event.data)
            )
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
        except Exception as exc:
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            error_event = await self._emit_event(
                "error",
                step=step_count,
                run_id=run_id,
                data={
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
            )
            yield error_event

    def _copy_messages(self) -> list[Message]:
        return [message.model_copy(deep=True) for message in self.session.context.get_messages()]

    def _append_tool_messages(self, tool_messages: list[Message]) -> None:
        """把工具输出写回当前会话上下文。"""
        for tool_message in tool_messages:
            self.session.context.add_tool_message(
                tool_message.content or "", tool_message.tool_call_id
            )

    def _iter_tool_feedback_chunks(self, tool_messages: list[Message]):
        """把工具错误转换成流式 chunk，便于上层统一消费。"""
        for tool_message in tool_messages:
            error_message = self.tool_executor.extract_error_message(tool_message.content)
            if error_message:
                yield LLMResponseChunk(content=f"\n[工具执行错误] {error_message}\n")

    async def _execute_one_tool_call(
        self,
        tool_call: ToolCall,
        *,
        step: int,
        task: "Task | None" = None,
        run_id: str | None = None,
    ) -> Message:
        tool = self.tool_executor.registry.get(tool_call.function.name)
        await self._emit_event(
            "tool_start",
            step=step,
            run_id=run_id,
            data={
                "tool_name": tool_call.function.name,
                "display_name": tool.display_name if tool else None,
                "args": tool_call.function.arguments,
                "tool_call_id": tool_call.id,
                "side_effect": bool(tool and tool.execution.side_effect),
                "requires_approval": bool(tool and tool.execution.requires_approval),
            },
        )
        try:
            tool_message = await self.tool_executor.execute_one(tool_call)
        except Exception as exc:
            tool_message = Message(
                role="tool",
                content=f"工具执行失败: {exc}",
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
            )

        await self._emit_event(
            "tool_end",
            step=step,
            run_id=run_id,
            data={
                "tool_name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "output": tool_message.content,
            },
        )
        return tool_message

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        step: int,
        task: "Task | None" = None,
        run_id: str | None = None,
    ) -> list[Message]:
        """执行所有工具调用"""
        tool_messages = []
        for tool_call in tool_calls:
            tool_message = await self._execute_one_tool_call(
                tool_call,
                step=step,
                task=task,
                run_id=run_id,
            )
            tool_messages.append(tool_message)
        return tool_messages

    async def _run_non_stream(
        self,
        user_input: str,
        task: "Task | None" = None,
    ) -> AgentRunResult:
        """非流式执行"""
        start_time = time.monotonic()
        llm_calls = 0
        total_usage = self._empty_usage()
        context = self.session.context
        context.add_user_message(user_input)

        try:
            step_count = 0
            while step_count < self.session.max_steps:
                messages = self._copy_messages()
                tools = list(self.tools)

                # 使用 Middleware
                ctx = MiddlewareContext(
                    state=None,
                    config={},
                    step=llm_calls,
                )
                ctx.state = type('State', (), {'messages': messages, 'metadata': {}})()
                ctx = await self.middleware.execute_before_llm(ctx)
                messages = ctx.state.messages

                response = await self.llm.generate(messages, tools=tools)
                llm_calls += 1
                self._merge_usage(total_usage, response.usage)

                ai_message = context.add_ai_message(
                    response.message.content,
                    tool_calls=response.message.tool_calls or [],
                )

                if response.message.tool_calls:
                    tool_messages = await self._execute_tool_calls(
                        response.message.tool_calls,
                        step=step_count,
                        task=task,
                    )
                    self._append_tool_messages(tool_messages)
                    step_count += 1
                else:
                    latency_ms = int((time.monotonic() - start_time) * 1000)
                    return self._build_run_result(
                        content=response.message.content or "",
                        usage=total_usage,
                        latency_ms=latency_ms,
                        llm_calls=llm_calls,
                    )

            latency_ms = int((time.monotonic() - start_time) * 1000)
            return self._build_run_result(
                content="任务未完成，已达到最大步骤限制。",
                usage=total_usage,
                latency_ms=latency_ms,
                llm_calls=llm_calls,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            return self._build_run_result(
                content=f"执行出错: {exc}",
                usage=total_usage,
                latency_ms=latency_ms,
                llm_calls=llm_calls,
            )
