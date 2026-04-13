from typing import Any

from wuwei.agent.base import BaseSessionAgent
from wuwei.agent.session import AgentSession
from wuwei.llm import LLMGateway
from wuwei.runtime import AgentRunner
from wuwei.tools import Tool, ToolRegistry


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
