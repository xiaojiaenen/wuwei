from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.llm import LLMResponse, Message, ToolCall
    from wuwei.planning import Task
    from wuwei.tools import Tool
    from wuwei.llm import Message as ToolMessage

class RuntimeHook:
    async def before_llm(
        self,
        session: "AgentSession",
        messages: list["Message"],
        tools: list["Tool"],
        *,
        step: int,
        task: "Task | None" = None,
    ) -> tuple[list["Message"], list["Tool"]]:
        return messages, tools

    async def after_llm(
        self,
        session: "AgentSession",
        response: "LLMResponse",
        *,
        step: int,
        task: "Task | None" = None,
    ) -> None:
        pass

    async def before_tool(
        self,
        session: "AgentSession",
        tool_call: "ToolCall",
        *,
        step: int,
        task: "Task | None" = None,
    ) -> None:
        pass

    async def after_tool(
        self,
        session: "AgentSession",
        tool_call: "ToolCall",
        tool_message,
        *,
        step: int,
        task: "Task | None" = None,
    ) -> None:
        pass

    async def on_task_start(self, session: "AgentSession", task: "Task") -> None:
        pass

    async def on_task_end(self, session: "AgentSession", task: "Task") -> None:
        pass

class HookManager:
    def __init__(self, hooks: Iterable[RuntimeHook] | None = None) -> None:
        self._hooks = list(hooks or [])

    def register(self, hook: RuntimeHook) -> None:
        self._hooks.append(hook)

    async def before_llm(self, session, messages, tools, *, step: int, task=None):
        current_messages = messages
        current_tools = tools
        for hook in self._hooks:
            current_messages, current_tools = await hook.before_llm(
                session,
                current_messages,
                current_tools,
                step=step,
                task=task,
            )
        return current_messages, current_tools

    async def after_llm(self, session, response, *, step: int, task=None) -> None:
        for hook in self._hooks:
            await hook.after_llm(session, response, step=step, task=task)

    async def before_tool(self, session, tool_call, *, step: int, task=None) -> None:
        for hook in self._hooks:
            await hook.before_tool(session, tool_call, step=step, task=task)

    async def after_tool(self, session, tool_call, tool_message, *, step: int, task=None) -> None:
        for hook in self._hooks:
            await hook.after_tool(session, tool_call, tool_message, step=step, task=task)

    async def on_task_start(self, session, task) -> None:
        for hook in self._hooks:
            await hook.on_task_start(session, task)

    async def on_task_end(self, session, task) -> None:
        for hook in self._hooks:
            await hook.on_task_end(session, task)