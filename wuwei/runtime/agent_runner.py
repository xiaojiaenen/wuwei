import asyncio
from typing import AsyncIterator, TYPE_CHECKING

from wuwei.agent.session import AgentSession
from wuwei.llm import AgentEvent, LLMGateway, LLMResponse, LLMResponseChunk, Message, ToolCall
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

    async def stream_events(
        self,
        user_input: str,
        *,
        task: "Task | None" = None,
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流的形式执行一次 agent 运行。"""
        step_count = 0
        context = self.session.context
        context.add_user_message(user_input)

        try:
            while step_count < self.session.max_steps:
                content_parts: list[str] = []
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

                stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                    messages,
                    tools=tools,
                    stream=True,
                )

                async for chunk in stream:
                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield AgentEvent(
                            type="text_delta",
                            session_id=self.session.session_id,
                            step=step_count,
                            data={"content": chunk.content},
                        )

                    if chunk.tool_calls_complete:
                        full_tool_calls = chunk.tool_calls_complete

                context.add_ai_message("".join(content_parts), tool_calls=full_tool_calls)

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
                )
                return

            limit_message = "任务未完成，已达到最大步骤限制。"
            context.add_ai_message(limit_message)
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
                data={"reason": "max_steps"},
            )
        except Exception as exc:
            yield AgentEvent(
                type="error",
                session_id=self.session.session_id,
                step=step_count,
                data={
                    "message": str(exc),
                    "error_type": type(exc).__name__,
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
        await self.hooks.before_tool(self.session, tool_call, step=step, task=task)
        tool_message = await self.tool_executor.execute_one(tool_call)
        await self.hooks.after_tool(self.session, tool_call, tool_message, step=step, task=task)
        return tool_message

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
        context = self.session.context
        context.add_user_message(user_input)

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
            await self.hooks.after_llm(
                self.session,
                response,
                step=step_count,
                task=task,
            )

            context.add_ai_message(
                response.message.content,
                response.message.tool_calls,
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

            return response.message.content

        limit_message = "任务未完成，已达到最大步骤限制。"
        context.add_ai_message(limit_message)
        return limit_message

    async def _run_stream(
        self,
        user_input: str,
        *,
        task: "Task | None" = None,
    ) -> AsyncIterator[LLMResponseChunk]:
        """流式执行路径。"""
        step_count = 0
        context = self.session.context
        context.add_user_message(user_input)

        while step_count < self.session.max_steps:
            content_parts: list[str] = []
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

            stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                messages,
                tools=tools,
                stream=True,
            )

            async for chunk in stream:
                if chunk.content:
                    content_parts.append(chunk.content)
                    yield chunk

                if chunk.tool_calls_complete:
                    full_tool_calls = chunk.tool_calls_complete

            context.add_ai_message("".join(content_parts), tool_calls=full_tool_calls)

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

            return

        limit_message = "任务未完成，已达到最大步骤限制。"
        context.add_ai_message(limit_message)
        yield LLMResponseChunk(content=limit_message, finish_reason="stop")
