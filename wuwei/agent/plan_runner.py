from typing import Any, AsyncIterator

from wuwei.agent.runner import AgentRunner
from wuwei.agent.session import AgentSession
from wuwei.core.planner import Planner
from wuwei.core.task import Task
from wuwei.llm import LLMGateway, LLMResponseChunk
from wuwei.tools import Tool, ToolExecutor


class PlannerExecutorRunner:
    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool],
        tool_executor: ToolExecutor,
        session: AgentSession,
        planner: Planner | None = None,
    ) -> None:
        """初始化 plan-and-execute 运行器。"""
        self.llm = llm
        self.tools = tools
        self.tool_executor = tool_executor
        self.session = session
        self.planner = planner or Planner.create_planner(llm=self.llm)
        self.last_tasks: list[Task] = []

    async def run(self, user_input: str, stream: bool = False) -> Any:
        """对外统一入口：先规划，再执行。"""
        tasks = await self.plan(user_input)
        return await self.execute(user_input, tasks, stream=stream)

    async def plan(self, goal: str) -> list[Task]:
        """只做规划，不执行任务。"""
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
        """
        非流式执行任务图。

        实现思路：
        1. 先把任务列表整理成索引和依赖图。
        2. 进入调度循环，每轮找出当前可执行的任务。
        3. 逐个执行这些任务。
        4. 如果上游失败，及时把下游标记为 blocked。
        5. 循环结束后，对仍未完成的任务做兜底处理。
        """
        tasks_by_id, dependencies = self._index_tasks(tasks)

        while True:
            self._mark_blocked_tasks(tasks_by_id, dependencies)
            ready_tasks = self._get_ready_tasks(tasks_by_id, dependencies)
            if not ready_tasks:
                break

            # 这里先保持串行，逻辑最清晰，也最容易调试。
            for task in ready_tasks:
                await self._execute_task_non_stream(goal, task, tasks_by_id, dependencies)

        self._mark_unresolved_tasks(tasks_by_id)
        return tasks

    async def _execute_stream(
        self,
        goal: str,
        tasks: list[Task],
    ) -> AsyncIterator[LLMResponseChunk]:
        """
        流式执行任务图。

        整体调度流程和非流式一致，不同点在于：
        - 单个任务执行过程中，需要把流式 chunk 持续向上游转发。
        - 任务结束后，仍然要把最终结果写回 Task.result。
        """
        tasks_by_id, dependencies = self._index_tasks(tasks)

        while True:
            self._mark_blocked_tasks(tasks_by_id, dependencies)
            ready_tasks = self._get_ready_tasks(tasks_by_id, dependencies)
            if not ready_tasks:
                break

            # 流式模式下先保持串行输出，否则多个任务的 chunk 会混在一起。
            for task in ready_tasks:
                async for chunk in self._execute_task_stream(goal, task, tasks_by_id, dependencies):
                    yield chunk

        self._mark_unresolved_tasks(tasks_by_id)

    def _index_tasks(self, tasks: list[Task]) -> tuple[dict[int, Task], dict[int, list[int]]]:
        """
        把任务列表转换成便于调度的数据结构。

        返回两个结果：
        - tasks_by_id: task_id -> Task
        - dependencies: task_id -> 该任务依赖的上游 task_id 列表

        注意：
        Task.next 表示“下游任务”，所以这里要做一次反向映射。
        """
        # 建立 id 到任务对象的映射，后续取任务时会更方便。
        tasks_by_id = {task.id: task for task in tasks}

        # 如果字典长度变短，说明存在重复 task.id。
        if len(tasks_by_id) != len(tasks):
            raise ValueError("Planner 返回了重复的 task.id")

        # 先为每个任务初始化一个“上游依赖列表”。
        dependencies: dict[int, list[int]] = {task.id: [] for task in tasks}

        # task.next 是“我完成后，谁可以继续执行”。
        # 这里把它反转成“某个任务依赖哪些上游任务”。
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
        """
        找出当前这一轮可以执行的任务。

        满足以下条件的任务可以执行：
        - 当前状态仍可执行，比如 pending 或 in_progress
        - 它的所有上游依赖都已经 completed
        """
        ready: list[Task] = []

        for task in tasks_by_id.values():
            if task.status not in {"pending", "in_progress"}:
                continue

            parent_ids = dependencies[task.id]
            if all(tasks_by_id[parent_id].status == "completed" for parent_id in parent_ids):
                ready.append(task)

        # 排序后调试体验更稳定；初始 in_progress 的任务优先执行。
        return sorted(
            ready,
            key=lambda task: (0 if task.status == "in_progress" else 1, task.id),
        )

    def _mark_blocked_tasks(
        self,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> None:
        """
        把被上游失败任务阻塞的任务标记为 blocked。

        规则很简单：
        只要某个任务依赖的任意上游任务已经 failed 或 blocked，
        它自己就不可能继续执行了。
        """
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
        """
        对调度结束后仍未完成的任务做兜底处理。

        如果主循环已经结束，但任务还停留在 pending 或 in_progress，
        说明任务图无法继续推进，此时统一标成 blocked。
        """
        for task in tasks_by_id.values():
            if task.status in {"pending", "in_progress"}:
                task.status = "blocked"
                task.error = "No executable path remained; the task graph may contain invalid dependencies"

    def _create_task_session(self, task_id: int) -> AgentSession:
        """
        为单个任务创建独立会话。

        这样每个 task 都有自己的上下文，不会把别的 task 对话历史混进来。
        """
        return AgentSession(
            session_id=f"{self.session.session_id}:task:{task_id}",
            system_prompt=self.session.system_prompt,
            max_steps=self.session.max_steps,
            parallel_tool_calls=self.session.parallel_tool_calls,
        )

    def _create_runner(self, task_session: AgentSession) -> AgentRunner:
        """基于 task session 创建单任务执行器。"""
        return AgentRunner(
            llm=self.llm,
            tools=self.tools,
            tool_executor=self.tool_executor,
            session=task_session,
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
        """
        把上游已完成任务结果整理成 prompt 文本。

        这里只组织“已完成任务”的结果，避免把失败或未完成任务的噪音带进去。
        """
        completed_tasks = [task for task in tasks if task.status == "completed"]
        if not completed_tasks:
            return "无"

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
        """
        执行单个任务的非流式版本。

        主要流程：
        1. 收集上游结果
        2. 构造执行 prompt
        3. 创建独立 session 和 runner
        4. 执行任务
        5. 回写 task.result / task.error / task.status
        """
        dependency_tasks = self._get_dependency_tasks(task, tasks_by_id, dependencies)
        completed_task_results = self._format_completed_task_results(dependency_tasks)
        prompt = self._build_prompt(goal, task, completed_task_results)

        task.status = "in_progress"
        task.error = None

        task_session = self._create_task_session(task.id)
        runner = self._create_runner(task_session)

        try:
            task.result = await runner.run(prompt, stream=False)
            task.status = "completed"
        except Exception as exc:
            task.result = None
            task.error = str(exc)
            task.status = "failed"

    async def _execute_task_stream(
        self,
        goal: str,
        task: Task,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> AsyncIterator[LLMResponseChunk]:
        """
        执行单个任务的流式版本。

        执行时持续向外转发 chunk；执行完成后，再把最终结果落回 task.result。
        """
        dependency_tasks = self._get_dependency_tasks(task, tasks_by_id, dependencies)
        completed_task_results = self._format_completed_task_results(dependency_tasks)
        prompt = self._build_prompt(goal, task, completed_task_results)

        task.status = "in_progress"
        task.error = None

        task_session = self._create_task_session(task.id)
        runner = self._create_runner(task_session)

        try:
            stream = await runner.run(prompt, stream=True)
            async for chunk in stream:
                yield chunk

            # 流式执行结束后，从 task 自己的 session 里取最终 assistant 消息。
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

    def _build_prompt(self, goal: str, task: Task, completed_task_results: str) -> str:
        """
        构造“执行单个任务”时发给模型的 prompt。

        prompt 里只强调四类信息：
        - 总目标
        - 当前任务
        - 已完成上游结果
        - 执行约束
        """
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
2. 优先使用“已完成上游任务结果”作为当前任务输入。
3. 需要精确数据或外部能力时，主动调用工具，不要凭空编造。
4. 如果当前任务已经可以直接完成，就直接输出结果。
5. 如果工具出错，可以先根据错误调整一次；仍然失败再明确说明原因。
6. 输出尽量直接作为 task.result，不要写成长篇过程日志。
7. 不要输出“我将开始执行”“下一步计划是”这类规划废话。
8. 如果无法完成当前任务，请明确说明阻塞原因。

# 输出要求
1. 直接返回当前任务的执行结果。
2. 不要输出 JSON。
3. 不要输出 Markdown 代码块。
4. 不要重复整段任务描述。
5. 不要暴露内部思考过程。
"""
