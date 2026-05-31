"""多 Agent 协作模块"""

from dataclasses import dataclass, field
from typing import Any, Callable
from wuwei.agent.agent import Agent
from wuwei.core.message import HumanMessage, AIMessage


@dataclass
class SubTask:
    """子任务"""
    id: int
    description: str
    assigned_to: str = ""
    result: str = ""
    status: str = "pending"  # pending/in_progress/completed/failed


@dataclass
class TeamMember:
    """团队成员"""
    name: str
    agent: Agent
    role: str
    tools: list[str] = field(default_factory=list)


class Swarm:
    """多 Agent 协作（Swarm）

    借鉴 Claude Code 的 Swarm 机制，支持：
    - 领导者分解任务
    - 成员执行子任务
    - 结果汇总

    示例：
        leader = Agent(llm=llm, tools=[...])
        members = [
            TeamMember(name="researcher", agent=researcher, role="研究员"),
            TeamMember(name="writer", agent=writer, role="写手"),
        ]

        swarm = Swarm(leader=leader, members=members)
        result = await swarm.run("写一篇关于 AI 的文章")
    """

    def __init__(
        self,
        leader: Agent,
        members: list[TeamMember] = None,
    ):
        self.leader = leader
        self.members = {m.name: m for m in (members or [])}
        self.handoff_history: list[dict] = []

    async def run(self, task: str) -> str:
        """执行任务

        Args:
            task: 任务描述

        Returns:
            任务结果
        """
        # 1. 领导者分解任务
        subtasks = await self._decompose_task(task)

        # 2. 执行子任务
        results = {}
        for subtask in subtasks:
            # 分配任务
            assigned_to = await self._assign_task(subtask, results)
            subtask.assigned_to = assigned_to
            subtask.status = "in_progress"

            if assigned_to == "leader":
                # 领导者自己执行
                result = await self.leader.run(
                    subtask.description,
                )
            else:
                # 委派给成员
                member = self.members[assigned_to]
                result = await member.agent.run(
                    subtask.description,
                )

            subtask.result = result
            subtask.status = "completed"
            results[subtask.id] = subtask

            # 记录协作历史
            self.handoff_history.append({
                "task": subtask.description,
                "assigned_to": assigned_to,
                "result": result,
            })

        # 3. 领导者汇总结果
        summary = await self._synthesize_results(results)
        return summary

    async def _decompose_task(self, task: str) -> list[SubTask]:
        """分解任务

        使用领导者 LLM 将任务分解为子任务。
        """
        prompt = f"""请将以下任务分解为 2-5 个子任务：

任务：{task}

可用的团队成员：
{self._format_members()}

请以 JSON 格式返回子任务列表：
[
  {{"description": "子任务描述", "suggested_assignee": "成员名或leader"}}
]"""

        response = await self.leader.run(prompt)

        # 简单解析（实际应该用结构化输出）
        import json
        try:
            # 尝试从响应中提取 JSON
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end != 0:
                tasks_data = json.loads(response[start:end])
                return [
                    SubTask(
                        id=i,
                        description=t["description"],
                        assigned_to=t.get("suggested_assignee", "leader"),
                    )
                    for i, t in enumerate(tasks_data)
                ]
        except (json.JSONDecodeError, KeyError):
            pass

        # 如果解析失败，创建单个子任务
        return [SubTask(id=0, description=task, assigned_to="leader")]

    async def _assign_task(
        self,
        subtask: SubTask,
        completed: dict[int, SubTask],
    ) -> str:
        """分配任务

        根据子任务描述和成员能力分配任务。
        """
        # 使用建议的分配
        if subtask.assigned_to in self.members or subtask.assigned_to == "leader":
            return subtask.assigned_to

        # 默认分配给领导者
        return "leader"

    def _format_members(self) -> str:
        """格式化成员信息"""
        lines = []
        for name, member in self.members.items():
            lines.append(f"- {name}: {member.role}")
        return "\n".join(lines)

    async def _synthesize_results(
        self,
        results: dict[int, SubTask],
    ) -> str:
        """汇总结果

        使用领导者 LLM 汇总所有子任务结果。
        """
        results_text = "\n\n".join(
            f"子任务 {subtask.id} ({subtask.assigned_to}):\n{subtask.result}"
            for subtask in results.values()
        )

        prompt = f"""请汇总以下子任务的结果，形成最终输出：

{results_text}"""

        return await self.leader.run(prompt)
