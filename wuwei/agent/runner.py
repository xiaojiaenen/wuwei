from typing import AsyncIterator

from wuwei.agent.session import AgentSession
from wuwei.llm import LLMGateway, LLMResponse, LLMResponseChunk
from wuwei.tools import Tool, ToolExecutor


class AgentRunner:
    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool],
        tool_executor: ToolExecutor,
        session: AgentSession,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = tool_executor
        self.session = session

    async def run(self, user_input: str, stream: bool = False):
        if stream:
            return self._run_stream(user_input)
        return await self._run_non_stream(user_input)

    def _append_tool_messages(self, tool_messages) -> None:
        for tool_message in tool_messages:
            self.session.context.add_tool_message(tool_message.content or "", tool_message.tool_call_id)

    def _iter_tool_feedback_chunks(self, tool_messages):
        for tool_message in tool_messages:
            error_message = self.tool_executor.extract_error_message(tool_message.content)
            if error_message:
                yield LLMResponseChunk(content=f"\n[工具执行错误] {error_message}\n")

    async def _execute_tool_calls(self, tool_calls):
        return await self.tool_executor.execute(
            tool_calls,
            concurrent=self.session.parallel_tool_calls,
        )

    async def _run_non_stream(self, user_input: str):
        step_count = 0
        context = self.session.context
        context.add_user_message(user_input)

        while step_count < self.session.max_steps:
            response: LLMResponse = await self.llm.generate(
                context.get_messages(),
                tools=self.tools,
            )
            context.add_ai_message(
                response.message.content,
                response.message.tool_calls,
            )

            if response.finish_reason == "tool_calls" and response.message.tool_calls:
                tool_messages = await self._execute_tool_calls(response.message.tool_calls)
                self._append_tool_messages(tool_messages)
                step_count += 1
                continue

            return response.message.content

        limit_message = "任务未完成，已达到最大步骤限制"
        context.add_ai_message(limit_message)
        return limit_message

    async def _run_stream(self, user_input: str) -> AsyncIterator[LLMResponseChunk]:
        step_count = 0
        context = self.session.context
        context.add_user_message(user_input)

        while step_count < self.session.max_steps:
            content_parts: list[str] = []
            full_tool_calls = None

            stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                context.get_messages(),
                tools=self.tools,
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
                tool_messages = await self._execute_tool_calls(full_tool_calls)
                self._append_tool_messages(tool_messages)
                for chunk in self._iter_tool_feedback_chunks(tool_messages):
                    yield chunk
                step_count += 1
                continue

            return

        limit_message = "任务未完成，已达到最大步骤限制"
        context.add_ai_message(limit_message)
        yield LLMResponseChunk(content=limit_message, finish_reason="stop")
