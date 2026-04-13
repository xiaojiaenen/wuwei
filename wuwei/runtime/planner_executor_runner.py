from typing import Any, AsyncIterator

from wuwei.agent.session import AgentSession
from wuwei.llm import LLMGateway, LLMResponseChunk
from wuwei.planning import Planner, Task
from wuwei.runtime.agent_runner import AgentRunner
from wuwei.runtime.hooks import HookManager
from wuwei.tools import Tool, ToolExecutor


class PlannerExecutorRunner:
    """负责在任务 DAG 上执行 plan-and-execute 流程。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool],
        tool_executor: ToolExecutor,
        session: AgentSession,
        planner: Planner | None = None,
        hooks: HookManager | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = tool_executor
        self.session = session
        self.planner = planner or Planner.create_planner(llm=self.llm)
        self.hooks = hooks or HookManager()
        self.last_tasks: list[Task] = []

    async def run(self, user_input: str, stream: bool = False) -> Any:
        """先规划，再执行。"""
        tasks = await self.plan(user_input)
        return await self.execute(user_input, tasks, stream=stream)

    async def plan(self, goal: str) -> list[Task]:
        """只做任务规划，不执行任务。"""
        tasks = await self.planner.plan_task(goal)
        self.last_tasks = tasks
        return tasks

    async def execute(
        self,
        goal: str,
        tasks: list[Task],
        stream: bool = False,
    ) -> Any:
        """执行已经规划好的任务列表。"""
        self.last_tasks = tasks
        if stream:
            return self._execute_stream(goal, tasks)
        return await self._execute_non_stream(goal, tasks)

    async def _execute_non_stream(self, goal: str, tasks: list[Task]) -> list[Task]:
        """按依赖顺序非流式执行任务图。"""
        tasks_by_id, dependencies = self._index_tasks(tasks)

        while True:
            self._mark_blocked_tasks(tasks_by_id, dependencies)
            ready_tasks = self._get_ready_tasks(tasks_by_id, dependencies)
            if not ready_tasks:
                break

            for task in ready_tasks:
                await self._execute_task_non_stream(goal, task, tasks_by_id, dependencies)

        self._mark_unresolved_tasks(tasks_by_id)
        return tasks

    async def _execute_stream(
        self,
        goal: str,
        tasks: list[Task],
    ) -> AsyncIterator[LLMResponseChunk]:
        """按依赖顺序流式执行任务图。"""
        tasks_by_id, dependencies = self._index_tasks(tasks)

        while True:
            self._mark_blocked_tasks(tasks_by_id, dependencies)
            ready_tasks = self._get_ready_tasks(tasks_by_id, dependencies)
            if not ready_tasks:
                break

            for task in ready_tasks:
                async for chunk in self._execute_task_stream(goal, task, tasks_by_id, dependencies):
                    yield chunk

        self._mark_unresolved_tasks(tasks_by_id)

    def _index_tasks(self, tasks: list[Task]) -> tuple[dict[int, Task], dict[int, list[int]]]:
        """建立任务索引，并把 next 反转成依赖列表。"""
        tasks_by_id = {task.id: task for task in tasks}
        if len(tasks_by_id) != len(tasks):
            raise ValueError("Planner 返回了重复的 task.id")

        dependencies: dict[int, list[int]] = {task.id: [] for task in tasks}
        for task in tasks:
            for child_id in task.next:
                if child_id not in tasks_by_id:
                    raise ValueError(f"任务 {task.id} 指向了不存在的下游任务 {child_id}")
                dependencies[child_id].append(task.id)

        return tasks_by_id, dependencies

    def _get_ready_tasks(
        self,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> list[Task]:
        """找出当前这一轮所有可执行的任务。"""
        ready: list[Task] = []

        for task in tasks_by_id.values():
            if task.status not in {"pending", "in_progress"}:
                continue

            parent_ids = dependencies[task.id]
            if all(tasks_by_id[parent_id].status == "completed" for parent_id in parent_ids):
                ready.append(task)

        return sorted(
            ready,
            key=lambda item: (0 if item.status == "in_progress" else 1, item.id),
        )

    def _mark_blocked_tasks(
        self,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> None:
        """把被失败上游阻塞的任务标记为 blocked。"""
        for task in tasks_by_id.values():
            if task.status not in {"pending", "in_progress"}:
                continue

            blocked_by = [
                parent_id
                for parent_id in dependencies[task.id]
                if tasks_by_id[parent_id].status in {"failed", "blocked"}
            ]
            if not blocked_by:
                continue

            task.status = "blocked"
            task.error = f"Blocked by upstream tasks: {', '.join(map(str, blocked_by))}"

    def _mark_unresolved_tasks(self, tasks_by_id: dict[int, Task]) -> None:
        """主循环结束后，把仍无法推进的任务统一标记为 blocked。"""
        for task in tasks_by_id.values():
            if task.status in {"pending", "in_progress"}:
                task.status = "blocked"
                task.error = "没有可继续执行的路径；任务图可能存在非法依赖。"

    def _create_task_session(self, task_id: int) -> AgentSession:
        """为单个任务创建隔离会话。"""
        return AgentSession(
            session_id=f"{self.session.session_id}:task:{task_id}",
            system_prompt=self.session.system_prompt,
            max_steps=self.session.max_steps,
            parallel_tool_calls=self.session.parallel_tool_calls,
        )

    def _create_runner(self, task_session: AgentSession) -> AgentRunner:
        """基于任务会话创建单任务执行器。"""
        return AgentRunner(
            llm=self.llm,
            tools=self.tools,
            tool_executor=self.tool_executor,
            session=task_session,
            hooks=self.hooks,
        )

    def _get_dependency_tasks(
        self,
        task: Task,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> list[Task]:
        """取出当前任务的所有上游任务对象。"""
        return [tasks_by_id[parent_id] for parent_id in dependencies[task.id]]

    def _format_completed_task_results(self, tasks: list[Task]) -> str:
        """把已完成上游任务结果整理成 prompt 文本。"""
        completed_tasks = [task for task in tasks if task.status == "completed"]
        if not completed_tasks:
            return "无已完成的上游任务结果。"

        parts: list[str] = []
        for task in sorted(completed_tasks, key=lambda item: item.id):
            parts.append(
                f"Task {task.id}\n"
                f"描述：{task.description}\n"
                f"结果：{task.result or ''}"
            )
        return "\n\n".join(parts)

    async def _execute_task_non_stream(
        self,
        goal: str,
        task: Task,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> None:
        """以非流式方式执行单个任务。"""
        dependency_tasks = self._get_dependency_tasks(task, tasks_by_id, dependencies)
        completed_task_results = self._format_completed_task_results(dependency_tasks)
        prompt = self._build_prompt(goal, task, completed_task_results)

        task.status = "in_progress"
        task.error = None
        await self.hooks.on_task_start(self.session, task)

        task_session = self._create_task_session(task.id)
        runner = self._create_runner(task_session)

        try:
            task.result = await runner.run(prompt, stream=False, task=task)
            task.status = "completed"
        except Exception as exc:
            task.result = None
            task.error = str(exc)
            task.status = "failed"
        finally:
            await self.hooks.on_task_end(self.session, task)

    async def _execute_task_stream(
        self,
        goal: str,
        task: Task,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> AsyncIterator[LLMResponseChunk]:
        """以流式方式执行单个任务。"""
        dependency_tasks = self._get_dependency_tasks(task, tasks_by_id, dependencies)
        completed_task_results = self._format_completed_task_results(dependency_tasks)
        prompt = self._build_prompt(goal, task, completed_task_results)

        task.status = "in_progress"
        task.error = None
        await self.hooks.on_task_start(self.session, task)

        task_session = self._create_task_session(task.id)
        runner = self._create_runner(task_session)

        try:
            stream = await runner.run(prompt, stream=True, task=task)
            async for chunk in stream:
                yield chunk

            last_message = task_session.context.get_last_message()
            if last_message and last_message.role == "assistant":
                task.result = last_message.content
            else:
                task.result = None
            task.status = "completed"
        except Exception as exc:
            task.result = None
            task.error = str(exc)
            task.status = "failed"
            yield LLMResponseChunk(content=f"\n[任务失败] Task {task.id}: {task.error}\n")
        finally:
            await self.hooks.on_task_end(self.session, task)

    def _build_prompt(self, goal: str, task: Task, completed_task_results: str) -> str:
        """构造执行单个任务时发给模型的 prompt。"""
        return f"""
# 角色
你是一个任务执行代理，只负责执行当前分配给你的单个任务。
你可以使用工具完成任务，但不要重新规划整个目标。

# 总目标
{goal}

# 当前任务
当前任务 ID:
{task.id}

当前任务描述：
{task.description}

# 上下文
已完成上游任务结果：
{completed_task_results}

# 执行规则
1. 只执行当前任务，不要改写任务图，不要自行新增长期计划。
2. 优先复用已完成的上游任务结果。
3. 需要精确数据或外部能力时，主动调用工具，不要凭空编造。
4. 如果当前任务已经可以直接完成，就直接输出结果。
5. 如果工具失败一次，可以根据错误调整后再试一次。
6. 输出尽量直接作为 task.result，并保持简洁。
7. 不要输出规划过程类废话。
8. 如果无法完成当前任务，请明确说明阻塞原因。

# 输出要求
1. 只返回当前任务结果。
2. 不要输出 JSON。
3. 不要输出 Markdown 代码块。
4. 不要重复整段任务描述。
5. 不要暴露内部思考过程。
""".strip()
