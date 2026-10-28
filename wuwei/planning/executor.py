"""Plan-and-Execute 执行器

补完框架的规划闭环：Planner 产出 TaskList（DAG），PlanExecutor 按
拓扑序调度执行——无依赖的任务并行跑（复用 asyncio 信号量），每个任务
由独立会话的子 agent 完成，异常不自动重放整个任务，结果回填 Task。

用法：
    agent = Agent(llm=llm, load_builtins=["file", "http"])
    result = await agent.run_plan("调研三个竞品并写对比报告")
    for task in result.tasks:
        print(task.id, task.status, task.result)

也可显式使用执行器：
    planner = Planner(llm)
    tasks = await planner.plan_task(goal)
    executor = PlanExecutor(llm, tools=agent.tool_registry)
    result = await executor.execute(TaskList(tasks=tasks))
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque

from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.planning.task import PlanRunResult, Task, TaskList
from wuwei.tools import ToolRegistry

logger = logging.getLogger("wuwei.planning")

_DEFAULT_TASK_PROMPT = (
    "你是一个任务执行器。只完成分配给你的这一个任务，"
    "输出该任务的执行结果（事实、数据、结论），不要输出与任务无关的内容。"
)


class PlanExecutor:
    """DAG 任务执行器。

    调度语义：
    - 入度为 0 的任务立即可执行；任务在所有上游任务完成后启动
    - 上游的结果以「上游任务 N 的结果：...」形式拼进下游任务的上下文
    - 失败或挂起不会自动重放整个任务；下游递归标记 blocked
    - retries 保留为兼容参数；安全重试应由具体的无副作用操作实现
    - 并行度由 max_parallel 控制
    """

    def __init__(
        self,
        llm: LLMGateway,
        tools: ToolRegistry | None = None,
        *,
        task_system_prompt: str = _DEFAULT_TASK_PROMPT,
        max_steps_per_task: int = 6,
        max_parallel: int = 4,
        retries: int = 1,
        middleware=None,
        checkpointer=None,
    ):
        self.llm = llm
        self.tools = tools
        self.task_system_prompt = task_system_prompt
        self.max_steps_per_task = max_steps_per_task
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.max_parallel = max_parallel
        self.retries = retries
        self.middleware = middleware
        self.checkpointer = checkpointer
        self._tasks_usage: list[tuple[dict[str, int], int]] = []

    # ── 公共 API ─────────────────────────────────────────────────────

    async def execute(self, plan: TaskList | list[Task], *, context: str = "") -> PlanRunResult:
        """执行整个任务 DAG。context 是传给每个任务的全局背景说明。"""
        tasks = plan.tasks if isinstance(plan, TaskList) else list(plan)
        self._tasks_usage = [(t.usage, t.llm_calls) for t in tasks]
        if not tasks:
            return PlanRunResult(tasks=[])

        by_id = self._validate_tasks(tasks)
        dependents: dict[int, list[int]] = defaultdict(list)
        indegree: dict[int, int] = {t.id: 0 for t in tasks}
        for task in tasks:
            for downstream in task.next:
                if downstream in by_id:
                    dependents[task.id].append(downstream)
                    if task.status != "completed":
                        indegree[downstream] += 1

        start = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_parallel)
        pending_q: deque[int] = deque(t.id for t in tasks if indegree[t.id] == 0)
        running: set[asyncio.Task] = set()

        async def _runner(task_id: int) -> None:
            async with semaphore:
                task = by_id[task_id]
                task.status = "in_progress"
                upstream_ctx = self._upstream_context(task, by_id)
                output, error = await self._run_task(task, context + upstream_ctx)
                if error is None:
                    task.status = "completed"
                    task.result = output
                else:
                    if task.status == "in_progress":
                        task.status = "failed"
                    task.result = output
                    task.error = error
                    self._block_downstream(task, by_id, dependents)
                    return
                # 唤醒下游（放回调度队列）
                for downstream in dependents[task.id]:
                    indegree[downstream] -= 1
                    if indegree[downstream] == 0 and by_id[downstream].status == "pending":
                        pending_q.append(downstream)

        try:
            while pending_q or running:
                while pending_q:
                    tid = pending_q.popleft()
                    if by_id[tid].status != "pending":
                        continue
                    running.add(asyncio.create_task(_runner(tid)))
                if not running:
                    break
                done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    exc = d.exception()
                    if exc:
                        raise exc

        finally:
            for future in running:
                future.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
            for task in tasks:
                if task.status == "in_progress":
                    task.status = "cancelled"

        usage, llm_calls = self._aggregate_usage()
        latency_ms = int((time.monotonic() - start) * 1000)
        return PlanRunResult(
            tasks=tasks,
            status=(
                "completed"
                if all(t.status == "completed" for t in tasks)
                else "interrupted"
                if any(t.status == "interrupted" for t in tasks)
                else "incomplete"
            ),
            usage=usage,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            execution_usage=usage,
            execution_llm_calls=llm_calls,
            execution_latency_ms=latency_ms,
        )

    # ── 内部 ─────────────────────────────────────────────────────────

    @staticmethod
    def _upstream_context(task: Task, by_id: dict[int, Task]) -> str:
        """收集所有直接上游任务的结果作为本任务上下文。"""
        parts = []
        for t in by_id.values():
            if task.id in t.next and t.result:
                parts.append(f"[上游任务 {t.id}: {t.description}]\n{t.result}")
        if not parts:
            return ""
        return "\n\n以下是上游任务的执行结果，供参考：\n" + "\n\n".join(parts)

    @staticmethod
    def _validate_tasks(tasks):
        by_id = {t.id: t for t in tasks}
        if len(by_id) != len(tasks):
            raise ValueError("Task IDs must be unique")
        for task in tasks:
            if len(task.next) != len(set(task.next)):
                raise ValueError("Duplicate DAG edge")
            if any(target not in by_id for target in task.next):
                raise ValueError("Unknown downstream task ID")
        degrees = {t.id: 0 for t in tasks}
        for task in tasks:
            for target in task.next:
                degrees[target] += 1
        ready = deque(tid for tid, degree in degrees.items() if degree == 0)
        visited = 0
        while ready:
            tid = ready.popleft()
            visited += 1
            for target in by_id[tid].next:
                degrees[target] -= 1
                if degrees[target] == 0:
                    ready.append(target)
        if visited != len(tasks):
            raise ValueError("Task dependencies must form a DAG")
        return by_id

    async def resume(
        self,
        plan: TaskList | list[Task],
        task_id: int,
        *,
        approved: bool = True,
        reason: str = "",
        context: str = "",
    ) -> PlanRunResult:
        """Resume one saved approval and schedule its newly ready descendants."""
        tasks = plan.tasks if isinstance(plan, TaskList) else plan
        self._validate_tasks(tasks)
        matches = [task for task in tasks if task.id == task_id]
        if len(matches) != 1 or matches[0].status != "interrupted" or not matches[0].interrupt:
            raise ValueError("Task must identify one interrupted task")
        task = matches[0]
        output, error = await self._run_task(
            task, context, resume=task.interrupt, approved=approved, reason=reason
        )
        task.result, task.error = output, error
        if error is None:
            task.status = "completed"
            task.interrupt = None
            # Only unblock descendants of this task; failures on other branches stay blocked.
            descendants = set(task.next)
            queue = deque(task.next)
            by_id = {t.id: t for t in tasks}
            while queue:
                for target in by_id[queue.popleft()].next:
                    if target not in descendants:
                        descendants.add(target)
                        queue.append(target)
            for candidate in tasks:
                if candidate.id in descendants and candidate.status == "blocked":
                    upstream = [t for t in tasks if candidate.id in t.next]
                    if all(t.status in {"completed", "pending", "blocked"} for t in upstream):
                        candidate.status, candidate.error = "pending", None
        return await self.execute(tasks, context=context)

    async def _run_task(
        self, task: Task, context: str, *, resume=None, approved=True, reason=""
    ) -> tuple[str | None, str | None]:
        """用独立会话的子 agent 执行单个任务，返回 (result, error)。"""
        worker = Agent(
            llm=self.llm,
            tools=self.tools or ToolRegistry(),
            load_builtins=False,
            default_system_prompt=self.task_system_prompt,
            default_max_steps=self.max_steps_per_task,
            middleware=self._inherited_middleware(),
            checkpointer=self.checkpointer,
        )
        session = worker.create_session()
        prompt = f"任务 #{task.id}: {task.description}"
        if context.strip():
            prompt = f"{prompt}\n\n背景：\n{context}"
        try:
            if resume:
                result = await worker.resume(resume, approved=approved, reason=reason)
            else:
                result = await worker.run(prompt, session=session)
            self._tasks_usage.append((result.usage, result.llm_calls))
            task.usage, task.llm_calls = result.usage, result.llm_calls
            task.session_id = resume["session_id"] if resume else session.session_id
            if result.status != "completed":
                task.status = (
                    result.status
                    if result.status in {"interrupted", "incomplete", "unknown", "cancelled"}
                    else "failed"
                )
                task.interrupt = getattr(result, "interrupt", None)
                return result.content, f"Task stopped with status: {result.status}"
            return result.content, None
        except Exception as exc:
            logger.exception("[plan] 任务 %s 执行异常", task.id)
            # An exception may follow a side effect; never replay the whole task.
            self._tasks_usage.append((session.last_usage, session.last_llm_calls))
            task.usage, task.llm_calls = session.last_usage, session.last_llm_calls
            messages = session.context.get_messages()
            task.status = "unknown" if any(m.role == "tool" for m in messages) else "failed"
            return None, f"{type(exc).__name__}: {exc}"

    def _inherited_middleware(self):
        from wuwei.agent.sub_agent import inherit_middleware
        from wuwei.runtime.context import current_context

        parent = current_context.get()
        return inherit_middleware(
            parent.config.get("middleware") if parent else None, self.middleware
        )

    @staticmethod
    def _block_downstream(
        task: Task, by_id: dict[int, Task], dependents: dict[int, list[int]]
    ) -> None:
        """上游失败时递归阻断下游任务。"""
        for downstream in dependents[task.id]:
            dt = by_id[downstream]
            if dt.status == "pending":
                dt.status = "blocked"
                dt.error = f"上游任务 {task.id} 失败"
                PlanExecutor._block_downstream(dt, by_id, dependents)

    def _aggregate_usage(self) -> tuple[dict[str, int], int]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        llm_calls = 0
        for u, calls in self._tasks_usage:
            for k, value in u.items():
                usage[k] = usage.get(k, 0) + (value or 0)
            llm_calls += calls
        return usage, llm_calls
