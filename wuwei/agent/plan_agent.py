from typing import Any

from wuwei.agent.base import BaseSessionAgent
from wuwei.agent.session import AgentSession
from wuwei.llm import LLMGateway
from wuwei.planning import Planner, Task
from wuwei.runtime import PlannerExecutorRunner
from wuwei.tools import Tool, ToolRegistry


class PlanAgent(BaseSessionAgent):
    """Plan-and-execute 门面对象。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        planner: Planner | None = None,
        default_system_prompt: str = "你是一个有用的助手",
        default_max_steps: int = 10,
        default_parallel_tool_calls: bool = False,
        hooks=None,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            default_system_prompt=default_system_prompt,
            default_max_steps=default_max_steps,
            default_parallel_tool_calls=default_parallel_tool_calls,
            hooks=hooks,
        )
        self.planner = planner or Planner.create_planner(llm=self.llm)

    def create_runner(self, session: AgentSession) -> PlannerExecutorRunner:
        """为 plan-and-execute 会话创建执行器。"""
        return PlannerExecutorRunner(
            llm=self.llm,
            tools=self.tool_registry.list_tools(),
            tool_executor=self.tool_executor,
            session=session,
            planner=self.planner,
            hooks=self.hooks,
        )

    async def plan(
        self,
        goal: str,
        session: AgentSession | None = None,
    ) -> list[Task]:
        """只做任务规划，不执行任务。"""
        current_session = session or self.create_or_get_session()
        runner = self.create_runner(current_session)
        return await runner.plan(goal)

    async def execute(
        self,
        goal: str,
        tasks: list[Task],
        session: AgentSession | None = None,
        stream: bool = False,
    ) -> Any:
        """执行已经规划好的任务列表。"""
        current_session = session or self.create_or_get_session()
        runner = self.create_runner(current_session)
        return await runner.execute(goal, tasks, stream=stream)

    async def run(
        self,
        user_input: str,
        session: AgentSession | None = None,
        stream: bool = False,
    ) -> Any:
        """对外统一入口：先规划，再执行。"""
        current_session = session or self.create_or_get_session()
        runner = self.create_runner(current_session)
        return await runner.run(user_input, stream=stream)
