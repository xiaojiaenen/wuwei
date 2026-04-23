from collections.abc import AsyncIterator
from typing import Any

from wuwei.agent.session import AgentSession
from wuwei.llm import AgentEvent, AgentRunResult, LLMGateway, LLMResponseChunk
from wuwei.planning import PlanRunResult, Planner, Task
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
        self.last_plan_usage = self._empty_usage()
        self.last_plan_latency_ms = 0
        self.last_plan_llm_calls = 0
        self.last_execution_usage = self._empty_usage()
        self.last_execution_latency_ms = 0
        self.last_execution_llm_calls = 0

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _merge_usage(self, total: dict[str, int], usage: dict[str, int] | None) -> None:
        if not usage:
            return

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = total.get(key, 0) + usage.get(key, 0)

    def _capture_plan_stats(self) -> None:
        self.last_plan_usage = dict(getattr(self.planner, "last_usage", self._empty_usage()))
        self.last_plan_latency_ms = int(getattr(self.planner, "last_latency_ms", 0))
        self.last_plan_llm_calls = int(getattr(self.planner, "last_llm_calls", 0))

    def _set_execution_stats(
        self,
        *,
        usage: dict[str, int],
        latency_ms: int,
        llm_calls: int,
    ) -> None:
        self.last_execution_usage = dict(usage)
        self.last_execution_latency_ms = latency_ms
        self.last_execution_llm_calls = llm_calls

    def _build_plan_run_result(self, tasks: list[Task]) -> PlanRunResult:
        total_usage = self._empty_usage()
        self._merge_usage(total_usage, self.last_plan_usage)
        self._merge_usage(total_usage, self.last_execution_usage)

        total_latency_ms = self.last_plan_latency_ms + self.last_execution_latency_ms
        total_llm_calls = self.last_plan_llm_calls + self.last_execution_llm_calls

        self.session.last_usage = dict(total_usage)
        self.session.last_latency_ms = total_latency_ms
        self.session.last_llm_calls = total_llm_calls

        return PlanRunResult(
            tasks=tasks,
            usage=total_usage,
            latency_ms=total_latency_ms,
            llm_calls=total_llm_calls,
            planner_usage=dict(self.last_plan_usage),
            planner_latency_ms=self.last_plan_latency_ms,
            planner_llm_calls=self.last_plan_llm_calls,
            execution_usage=dict(self.last_execution_usage),
            execution_latency_ms=self.last_execution_latency_ms,
            execution_llm_calls=self.last_execution_llm_calls,
        )

    async def run(self, user_input: str, stream: bool = False) -> Any:
        """先规划，再执行。"""
        tasks = await self.plan(user_input)
        if stream:
            return await self.execute(user_input, tasks, stream=stream)

        tasks = await self.execute(user_input, tasks, stream=False)
        return self._build_plan_run_result(tasks)

    async def stream_events(self, goal: str) -> AsyncIterator[AgentEvent]:
        """先规划，再以结构化事件流执行。"""
        tasks = await self.plan(goal)
        async for event in self.execute_events(goal, tasks):
            yield event

    async def plan(self, goal: str) -> list[Task]:
        """只做任务规划，不执行任务。"""
        tasks = await self.planner.plan_task(goal)
        self.last_tasks = tasks
        self._capture_plan_stats()
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

    async def execute_events(
        self,
        goal: str,
        tasks: list[Task],
    ) -> AsyncIterator[AgentEvent]:
        """按依赖顺序输出任务执行事件流。"""
        self.last_tasks = tasks
        execution_usage = self._empty_usage()
        execution_latency_ms = 0
        execution_llm_calls = 0
        tasks_by_id, dependencies = self._index_tasks(tasks)

        while True:
            self._mark_blocked_tasks(tasks_by_id, dependencies)
            ready_tasks = self._get_ready_tasks(tasks_by_id, dependencies)
            if not ready_tasks:
                break

            for task in ready_tasks:
                async for event in self._execute_task_events(goal, task, tasks_by_id, dependencies):
                    if event.type == "done":
                        self._merge_usage(
                            execution_usage,
                            event.data.get("usage") if isinstance(event.data, dict) else None,
                        )
                        execution_latency_ms += int(event.data.get("latency_ms", 0))
                        execution_llm_calls += int(event.data.get("llm_calls", 0))
                    yield event

        self._mark_unresolved_tasks(tasks_by_id)
        self._set_execution_stats(
            usage=execution_usage,
            latency_ms=execution_latency_ms,
            llm_calls=execution_llm_calls,
        )
        total_usage = self._empty_usage()
        self._merge_usage(total_usage, self.last_plan_usage)
        self._merge_usage(total_usage, execution_usage)
        self.session.last_usage = total_usage
        self.session.last_latency_ms = self.last_plan_latency_ms + execution_latency_ms
        self.session.last_llm_calls = self.last_plan_llm_calls + execution_llm_calls

    async def _execute_non_stream(self, goal: str, tasks: list[Task]) -> list[Task]:
        """按依赖顺序非流式执行任务图。"""
        execution_usage = self._empty_usage()
        execution_latency_ms = 0
        execution_llm_calls = 0
        tasks_by_id, dependencies = self._index_tasks(tasks)

        while True:
            self._mark_blocked_tasks(tasks_by_id, dependencies)
            ready_tasks = self._get_ready_tasks(tasks_by_id, dependencies)
            if not ready_tasks:
                break

            for task in ready_tasks:
                result = await self._execute_task_non_stream(goal, task, tasks_by_id, dependencies)
                if result is not None:
                    self._merge_usage(execution_usage, result.usage)
                    execution_latency_ms += result.latency_ms
                    execution_llm_calls += result.llm_calls

        self._mark_unresolved_tasks(tasks_by_id)
        self._set_execution_stats(
            usage=execution_usage,
            latency_ms=execution_latency_ms,
            llm_calls=execution_llm_calls,
        )
        self.session.last_usage = dict(execution_usage)
        self.session.last_latency_ms = execution_latency_ms
        self.session.last_llm_calls = execution_llm_calls
        return tasks

    async def _execute_stream(
        self,
        goal: str,
        tasks: list[Task],
    ) -> AsyncIterator[LLMResponseChunk]:
        """按依赖顺序流式执行任务图。"""
        self._set_execution_stats(
            usage=self._empty_usage(),
            latency_ms=0,
            llm_calls=0,
        )
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
        self.session.last_usage = dict(self.last_execution_usage)
        self.session.last_latency_ms = self.last_execution_latency_ms
        self.session.last_llm_calls = self.last_execution_llm_calls

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
                f"Task {task.id}\n" f"描述：{task.description}\n" f"结果：{task.result or ''}"
            )
        return "\n\n".join(parts)

    async def _execute_task_non_stream(
        self,
        goal: str,
        task: Task,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> AgentRunResult | None:
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
            result: AgentRunResult = await runner.run(prompt, stream=False, task=task)
            task.result = result.content
            task.status = "completed"
            return result
        except Exception as exc:
            task.result = None
            task.error = str(exc)
            task.status = "failed"
            return None
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
            self._merge_usage(self.last_execution_usage, task_session.last_usage)
            self.last_execution_latency_ms += task_session.last_latency_ms
            self.last_execution_llm_calls += task_session.last_llm_calls
            await self.hooks.on_task_end(self.session, task)

    async def _execute_task_events(
        self,
        goal: str,
        task: Task,
        tasks_by_id: dict[int, Task],
        dependencies: dict[int, list[int]],
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流方式执行单个任务。"""
        dependency_tasks = self._get_dependency_tasks(task, tasks_by_id, dependencies)
        completed_task_results = self._format_completed_task_results(dependency_tasks)
        prompt = self._build_prompt(goal, task, completed_task_results)

        task.status = "in_progress"
        task.error = None
        await self.hooks.on_task_start(self.session, task)

        task_session = self._create_task_session(task.id)
        runner = self._create_runner(task_session)

        try:
            async for event in runner.stream_events(prompt, task=task):
                event_data = dict(event.data)
                event_data["task_id"] = task.id
                event_data["task_description"] = task.description
                event_data["root_session_id"] = self.session.session_id
                yield event.model_copy(update={"data": event_data})

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
            yield AgentEvent(
                type="error",
                session_id=task_session.session_id,
                step=0,
                data={
                    "message": task.error,
                    "error_type": type(exc).__name__,
                    "task_id": task.id,
                    "task_description": task.description,
                    "root_session_id": self.session.session_id,
                },
            )
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
