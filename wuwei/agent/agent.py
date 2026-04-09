from typing import Any
from uuid import uuid4

from wuwei.agent.base import BaseAgent
from wuwei.agent.runner import AgentRunner
from wuwei.agent.session import AgentSession
from wuwei.llm import LLMGateway
from wuwei.tools import Tool, ToolExecutor, ToolRegistry


class Agent(BaseAgent):
    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        default_system_prompt: str = "你是一个有用的助手",
        default_max_steps: int = 10,
        default_parallel_tool_calls: bool = False,
    ) -> None:
        self.llm = llm
        self.default_system_prompt = default_system_prompt
        self.default_max_steps = default_max_steps
        self.default_parallel_tool_calls = default_parallel_tool_calls

        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        else:
            self.tool_registry = ToolRegistry()
            for tool in tools or []:
                self.tool_registry.register(tool)

        self.tool_executor = ToolExecutor(self.tool_registry)
        self._sessions: dict[str, AgentSession] = {}

    def create_session(
        self,
        session_id: str | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AgentSession:
        session = AgentSession(
            session_id=session_id or uuid4().hex,
            system_prompt=system_prompt or self.default_system_prompt,
            max_steps=max_steps if max_steps is not None else self.default_max_steps,
            parallel_tool_calls=(
                parallel_tool_calls
                if parallel_tool_calls is not None
                else self.default_parallel_tool_calls
            ),
        )
        self._sessions[session.session_id] = session
        return session

    def create_or_get_session(
        self,
        session_id: str | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AgentSession:
        if session_id is not None and session_id in self._sessions:
            return self._sessions[session_id]

        return self.create_session(
            session_id=session_id,
            system_prompt=system_prompt,
            max_steps=max_steps,
            parallel_tool_calls=parallel_tool_calls,
        )


    def create_runner(self, session: AgentSession) -> AgentRunner:
        return AgentRunner(
            llm=self.llm,
            tools=self.tool_registry.list_tools(),
            tool_executor=self.tool_executor,
            session=session,
        )

    async def run(
        self,
        user_input: str,
        session: AgentSession | None = None,
        stream: bool = False,
    ) -> Any:
        current_session = session or self.create_or_get_session()
        runner = self.create_runner(current_session)
        return await runner.run(user_input, stream=stream)
