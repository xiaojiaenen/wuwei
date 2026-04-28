import asyncio
import json
import time
from typing import AsyncIterator, TYPE_CHECKING

from wuwei.agent.session import AgentSession
from wuwei.llm import AgentEvent, AgentRunResult, LLMGateway, LLMResponse, LLMResponseChunk, Message, ToolCall
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
        context = self.session.context
        context.add_user_message(user_input)

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
                stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                    messages,
                    tools=tools,
                    stream=True,
                )
                llm_calls += 1

                async for chunk in stream:
                    if chunk.reasoning_content:
                        reasoning_parts.append(chunk.reasoning_content)
                        yield AgentEvent(
                            type="reasoning_delta",
                            session_id=self.session.session_id,
                            step=step_count,
                            data={"content": chunk.reasoning_content},
                        )

                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield AgentEvent(
                            type="text_delta",
                            session_id=self.session.session_id,
                            step=step_count,
                            data={"content": chunk.content},
                        )

                    self._merge_usage(total_usage, chunk.usage)

                    if chunk.tool_calls_complete:
                        full_tool_calls = chunk.tool_calls_complete

                total_latency_ms += int((time.monotonic() - llm_start) * 1000)
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
                        yield AgentEvent(
                            type="tool_start",
                            session_id=self.session.session_id,
                            step=step_count,
                            data={
                                "tool_name": tool_call.function.name,
                                "args": tool_call.function.arguments,
                                "tool_call_id": tool_call.id,
                            },
                        )

                    tool_messages = await self._execute_tool_calls(
                        full_tool_calls,
                        step=step_count,
                        task=task,
                    )
                    self._append_tool_messages(tool_messages)

                    for tool_call, tool_message in zip(full_tool_calls, tool_messages):
                        yield AgentEvent(
                            type="tool_end",
                            session_id=self.session.session_id,
                            step=step_count,
                            data={
                                "tool_name": tool_call.function.name,
                                "tool_call_id": tool_call.id,
                                "output": tool_message.content,
                            },
                        )

                        error_message = self.tool_executor.extract_error_message(tool_message.content)
                        if error_message:
                            yield AgentEvent(
                                type="error",
                                session_id=self.session.session_id,
                                step=step_count,
                                data={
                                    "message": error_message,
                                    "tool_name": tool_call.function.name,
                                    "tool_call_id": tool_call.id,
                                },
                            )

                    step_count += 1
                    continue

                yield AgentEvent(
                    type="done",
                    session_id=self.session.session_id,
                    step=step_count,
                    data={
                        "usage": dict(total_usage),
                        "latency_ms": total_latency_ms,
                        "llm_calls": llm_calls,
                    },
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
            yield AgentEvent(
                type="text_delta",
                session_id=self.session.session_id,
                step=step_count,
                data={"content": limit_message},
            )
            yield AgentEvent(
                type="done",
                session_id=self.session.session_id,
                step=step_count,
                data={
                    "reason": "max_steps",
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
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
            yield AgentEvent(
                type="error",
                session_id=self.session.session_id,
                step=step_count,
                data={
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
            )

    def _copy_messages(self) -> list[Message]:
        return [message.model_copy(deep=True) for message in self.session.context.get_messages()]

    def _append_tool_messages(self, tool_messages: list[Message]) -> None:
        """把工具输出写回当前会话上下文。"""
        for tool_message in tool_messages:
            self.session.context.add_tool_message(tool_message.content or "", tool_message.tool_call_id)

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
    ) -> Message:
        try:
            await self.hooks.before_tool(self.session, tool_call, step=step, task=task)
        except ToolApprovalRejected as exc:
            return self._build_tool_error_message(
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

        tool_message = await self.tool_executor.execute_one(tool_call)
        await self.hooks.after_tool(self.session, tool_call, tool_message, step=step, task=task)
        return tool_message

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
    ) -> list[Message]:
        """执行工具调用，并在每次调用前后触发 hook。"""
        if self.session.parallel_tool_calls and len(tool_calls) > 1:
            return await asyncio.gather(
                *(self._execute_one_tool_call(tool_call, step=step, task=task) for tool_call in tool_calls)
            )

        results: list[Message] = []
        for tool_call in tool_calls:
            results.append(await self._execute_one_tool_call(tool_call, step=step, task=task))
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
        context = self.session.context
        context.add_user_message(user_input)

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

                response: LLMResponse = await self.llm.generate(
                    messages,
                    tools=tools,
                )
                llm_calls += 1
                total_latency_ms += response.latency_ms
                self._merge_usage(total_usage, response.usage)

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
                    )
                    self._append_tool_messages(tool_messages)
                    step_count += 1
                    continue

                return self._build_run_result(
                    content=response.message.content or "",
                    usage=total_usage,
                    latency_ms=total_latency_ms,
                    llm_calls=llm_calls,
                )

            limit_message = "任务未完成，已达到最大步骤限制。"
            context.add_ai_message(limit_message)
            return self._build_run_result(
                content=limit_message,
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
        except Exception:
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
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
        context = self.session.context
        context.add_user_message(user_input)

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
                stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                    messages,
                    tools=tools,
                    stream=True,
                )
                llm_calls += 1

                async for chunk in stream:
                    if chunk.reasoning_content:
                        reasoning_parts.append(chunk.reasoning_content)

                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield chunk
                    elif chunk.reasoning_content:
                        yield chunk

                    self._merge_usage(total_usage, chunk.usage)

                    if chunk.tool_calls_complete:
                        full_tool_calls = chunk.tool_calls_complete

                total_latency_ms += int((time.monotonic() - llm_start) * 1000)
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
            yield LLMResponseChunk(content=limit_message, finish_reason="stop")
        except Exception:
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            raise
