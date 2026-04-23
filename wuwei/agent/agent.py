from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from wuwei.agent.base import BaseSessionAgent
from wuwei.agent.session import AgentSession
from wuwei.llm import AgentEvent, LLMGateway
from wuwei.runtime import AgentRunner
from wuwei.tools import Tool, ToolRegistry

ToolLike = Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]


class Agent(BaseSessionAgent):
    """普通单 agent 门面对象。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        default_system_prompt: str = "你是一个有用的助手",
        default_max_steps: int = 10,
        default_parallel_tool_calls: bool = False,
        hooks = None,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            default_system_prompt=default_system_prompt,
            default_max_steps=default_max_steps,
            default_parallel_tool_calls=default_parallel_tool_calls,
            hooks=hooks
        )

    def create_runner(self, session: AgentSession) -> AgentRunner:
        """为普通 agent 会话创建执行器。"""
        return AgentRunner(
            llm=self.llm,
            tools=self.tool_registry.list_tools(),
            tool_executor=self.tool_executor,
            session=session,
            hooks=self.hooks
        )

    async def run(
        self,
        user_input: str,
        session: AgentSession | None = None,
        stream: bool = False,
    ) -> Any:
        """运行一次普通 agent。"""
        current_session = session or self.create_or_get_session()
        runner = self.create_runner(current_session)
        return await runner.run(user_input, stream=stream)

    def stream_events(
        self,
        user_input: str,
        session: AgentSession | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流运行一次普通 agent。"""
        current_session = session or self.create_or_get_session(session_id=session_id)
        runner = self.create_runner(current_session)
        return runner.stream_events(user_input)

    @classmethod
    def from_env(
            cls,
            *,
            builtin_tools: list[str] | None = None,
            tools: list[ToolLike] | None = None,
            system_prompt: str = "你是一个有用的助手",
            max_steps: int = 10,
            parallel_tool_calls: bool = False,
            hooks=None,
            **llm_kwargs,
    ) -> "Agent":
        llm = LLMGateway.from_env(**llm_kwargs)
        registry = ToolRegistry.from_builtin(builtin_tools)

        for item in tools or []:
            if isinstance(item, Tool):
                registry.register(item)
                continue
            if callable(item):
                registry.register_callable(item)
                continue
            raise TypeError(f"tools 只支持 Tool 或可调用对象，收到: {type(item).__name__}")

        return cls(
            llm=llm,
            tools=registry,
            default_system_prompt=system_prompt,
            default_max_steps=max_steps,
            default_parallel_tool_calls=parallel_tool_calls,
            hooks=hooks,
        )
