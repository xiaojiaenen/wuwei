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
from wuwei.runtime.hitl import ToolApprovalRejected
from wuwei.runtime.hooks import HookManager
from wuwei.tools import Tool, ToolExecutor

if TYPE_CHECKING:
    from wuwei.planning import Task


class AgentRunner:
    """单个 agent 会话的运行时执行器。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool],
        tool_executor: ToolExecutor,
        session: AgentSession,
        hooks: HookManager | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = tool_executor
        self.session = session
        self.hooks = hooks or HookManager()

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
        await self.hooks.emit_event(event)
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
                messages, tools = await self.hooks.before_llm(
                    self.session,
                    messages,
                    tools,
                    step=step_count,
                    task=task,
                )

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
                await self.hooks.after_ai_message(
                    self.session,
                    ai_message,
                    step=step_count,
                    task=task,
                )

                if full_tool_calls:
                    for tool_call in full_tool_calls:
                        event = self._build_event(
                            "tool_start",
                            step=step_count,
                            run_id=run_id,
                            data={
                                "tool_name": tool_call.function.name,
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
            await self.hooks.after_ai_message(
                self.session,
                ai_message,
                step=step_count,
                task=task,
            )
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
                "args": tool_call.function.arguments,
                "tool_call_id": tool_call.id,
                "side_effect": bool(tool and tool.execution.side_effect),
                "requires_approval": bool(tool and tool.execution.requires_approval),
            },
        )
        try:
            await self.hooks.before_tool(self.session, tool_call, step=step, task=task, tool=tool)
        except ToolApprovalRejected as exc:
            tool_message = self._build_tool_error_message(
                tool_call,
                error_type=type(exc).__name__,
                message=str(exc),
                metadata={
                    "tool_executed": False,
                    "instruction": (
                        "The human rejected this tool call. The tool was not executed. "
                        "Do not claim the action succeeded; tell the user it was not completed."
                    ),
                },
            )
            await self._emit_tool_result_event(tool_call, tool_message, step=step, run_id=run_id)
            return tool_message

        tool_message = await self.tool_executor.execute_one(tool_call)
        await self.hooks.after_tool(
            self.session,
            tool_call,
            tool_message,
            step=step,
            task=task,
            tool=tool,
        )
        await self._emit_tool_result_event(tool_call, tool_message, step=step, run_id=run_id)
        return tool_message

    async def _emit_tool_result_event(
        self,
        tool_call: ToolCall,
        tool_message: Message,
        *,
        step: int,
        run_id: str | None,
    ) -> None:
        data = {
            "tool_name": tool_call.function.name,
            "tool_call_id": tool_call.id,
            "output": tool_message.content,
        }
        await self._emit_event("tool_end", step=step, run_id=run_id, data=data)
        error_message = self.tool_executor.extract_error_message(tool_message.content)
        if error_message:
            await self._emit_event(
                "tool_error",
                step=step,
                run_id=run_id,
                data={
                    "message": error_message,
                    "tool_name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                },
            )

    def _build_tool_error_message(
        self,
        tool_call: ToolCall,
        *,
        error_type: str,
        message: str,
        metadata: dict | None = None,
    ) -> Message:
        payload = {
            "ok": False,
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        if metadata:
            payload.update(metadata)

        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=json.dumps(payload, ensure_ascii=False),
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        step: int,
        task: "Task | None" = None,
        run_id: str | None = None,
    ) -> list[Message]:
        """执行工具调用，并在每次调用前后触发 hook。"""
        if self.session.parallel_tool_calls and len(tool_calls) > 1:
            return await asyncio.gather(
                *(
                    self._execute_one_tool_call(
                        tool_call,
                        step=step,
                        task=task,
                        run_id=run_id,
                    )
                    for tool_call in tool_calls
                )
            )

        results: list[Message] = []
        for tool_call in tool_calls:
            results.append(
                await self._execute_one_tool_call(
                    tool_call,
                    step=step,
                    task=task,
                    run_id=run_id,
                )
            )
        return results

    async def _run_non_stream(
        self,
        user_input: str,
        *,
        task: "Task | None" = None,
    ):
        """非流式执行路径。"""
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
                messages = self._copy_messages()
                tools = list(self.tools)
                messages, tools = await self.hooks.before_llm(
                    self.session,
                    messages,
                    tools,
                    step=step_count,
                    task=task,
                )

                await self._emit_event(
                    "llm_start",
                    step=step_count,
                    run_id=run_id,
                    data={"tools": [tool.name for tool in tools]},
                )
                response: LLMResponse = await self.llm.generate(
                    messages,
                    tools=tools,
                )
                llm_calls += 1
                total_latency_ms += response.latency_ms
                self._merge_usage(total_usage, response.usage)
                await self._emit_event(
                    "llm_end",
                    step=step_count,
                    run_id=run_id,
                    data={
                        "finish_reason": response.finish_reason,
                        "latency_ms": response.latency_ms,
                        "usage": dict(response.usage),
                    },
                )

                await self.hooks.after_llm(
                    self.session,
                    response,
                    step=step_count,
                    task=task,
                )

                context.add_ai_message(
                    response.message.content,
                    response.message.tool_calls,
                    reasoning_content=response.message.reasoning_content,
                )

                if response.finish_reason == "tool_calls" and response.message.tool_calls:
                    tool_messages = await self._execute_tool_calls(
                        response.message.tool_calls,
                        step=step_count,
                        task=task,
                        run_id=run_id,
                    )
                    self._append_tool_messages(tool_messages)
                    step_count += 1
                    continue

                result = self._build_run_result(
                    content=response.message.content or "",
                    usage=total_usage,
                    latency_ms=total_latency_ms,
                    llm_calls=llm_calls,
                )
                await self._emit_event(
                    "run_end",
                    step=step_count,
                    run_id=run_id,
                    data=result.model_dump(),
                )
                return result

            limit_message = "任务未完成，已达到最大步骤限制。"
            context.add_ai_message(limit_message)
            result = self._build_run_result(
                content=limit_message,
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            await self._emit_event(
                "run_end",
                step=step_count,
                run_id=run_id,
                data={**result.model_dump(), "reason": "max_steps"},
            )
            return result
        except Exception as exc:
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            await self._emit_event(
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
            raise

    async def _run_stream(
        self,
        user_input: str,
        *,
        task: "Task | None" = None,
    ) -> AsyncIterator[LLMResponseChunk]:
        """流式执行路径。"""
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
                messages, tools = await self.hooks.before_llm(
                    self.session,
                    messages,
                    tools,
                    step=step_count,
                    task=task,
                )

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
                        await self._emit_event(
                            "reasoning_delta",
                            step=step_count,
                            run_id=run_id,
                            data={"content": chunk.reasoning_content},
                        )

                    if chunk.content:
                        content_parts.append(chunk.content)
                        await self._emit_event(
                            "text_delta",
                            step=step_count,
                            run_id=run_id,
                            data={"content": chunk.content},
                        )
                        yield chunk
                    elif chunk.reasoning_content:
                        yield chunk

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
                await self.hooks.after_ai_message(
                    self.session,
                    ai_message,
                    step=step_count,
                    task=task,
                )

                if full_tool_calls:
                    tool_messages = await self._execute_tool_calls(
                        full_tool_calls,
                        step=step_count,
                        task=task,
                        run_id=run_id,
                    )
                    self._append_tool_messages(tool_messages)
                    for chunk in self._iter_tool_feedback_chunks(tool_messages):
                        yield chunk
                    step_count += 1
                    continue

                self._set_session_run_stats(
                    usage=total_usage,
                    latency_ms=total_latency_ms,
                    llm_calls=llm_calls,
                )
                await self._emit_event(
                    "run_end",
                    step=step_count,
                    run_id=run_id,
                    data={
                        "usage": dict(total_usage),
                        "latency_ms": total_latency_ms,
                        "llm_calls": llm_calls,
                    },
                )
                return

            limit_message = "任务未完成，已达到最大步骤限制。"
            ai_message = context.add_ai_message(limit_message)
            await self.hooks.after_ai_message(
                self.session,
                ai_message,
                step=step_count,
                task=task,
            )
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            await self._emit_event(
                "run_end",
                step=step_count,
                run_id=run_id,
                data={
                    "reason": "max_steps",
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
            )
            yield LLMResponseChunk(content=limit_message, finish_reason="stop")
        except Exception as exc:
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            await self._emit_event(
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
            raise
