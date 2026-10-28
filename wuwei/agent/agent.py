"""Agent 类 - 使用 Middleware + Plugin 系统"""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from wuwei.agent.base import BaseSessionAgent
from wuwei.agent.session import AgentSession
from wuwei.llm import AgentEvent, LLMGateway
from wuwei.plugin import Plugin, PluginContext, PluginManager
from wuwei.plugin.builtin import BUILTIN_GROUPS, load_builtin
from wuwei.runtime.agent_runner import AgentRunner
from wuwei.tools import Tool, ToolRegistry

ToolLike = Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]


def _extract_json(text: str):
    """从 LLM 输出中提取 JSON 对象（容忍 markdown 代码块与前后缀文本）。"""
    import json as _json

    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    # 括号匹配提取第一个完整 JSON 对象
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return _json.loads(text[start : i + 1])
                except _json.JSONDecodeError:
                    return None
    return None


class Agent(BaseSessionAgent):
    """普通单 agent 门面对象（使用 Middleware + Plugin）。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        default_system_prompt: str = "你是一个有用的助手",
        default_max_steps: int = 10,
        default_parallel_tool_calls: bool = False,
        middleware=None,
        skill_manager=None,
        mcp_manager=None,
        plugins: list[Plugin] | None = None,
        plugin_dir: str | Path | None = None,
        load_builtins: bool | str | list[str] = "minimal",
        checkpointer=None,
        event_bus=None,
    ) -> None:
        """初始化 Agent。

        Args:
            load_builtins: 内置插件加载策略——
                "minimal"（默认）: 只装 calc/time/json/text 等只读基础工具
                "all": 装载全部内置插件（含文件写入/git/npm/脚本执行等有副作用工具）
                list[str]: 按名称装载
                False: 不装载
            checkpointer: 检查点后端（MemoryCheckpointer / SQLiteCheckpointer），
                启用后每步落检查点，HITL interrupt 模式与 resume() 依赖它
            event_bus: EventBus 订阅总线，所有 AgentEvent 发布到总线
        """
        super().__init__(
            llm=llm,
            tools=tools,
            default_system_prompt=default_system_prompt,
            default_max_steps=default_max_steps,
            default_parallel_tool_calls=default_parallel_tool_calls,
            middleware=middleware,
        )

        # 保存供插件上下文使用
        self._skill_manager = skill_manager
        self._mcp_manager = mcp_manager

        # 初始化插件系统
        ctx = PluginContext(
            tool_registry=self.tool_registry,
            skill_manager=skill_manager,
            mcp_manager=mcp_manager,
            middleware_stack=self.middleware,
        )
        pm = PluginManager(ctx)

        # 按策略加载内置插件
        if load_builtins is True:
            load_builtin(pm, BUILTIN_GROUPS["all"])
        elif load_builtins is False:
            pass
        elif isinstance(load_builtins, str):
            group = BUILTIN_GROUPS.get(load_builtins)
            if group is None:
                raise ValueError(
                    f"未知内置插件组: {load_builtins}（可选: minimal, all，或传入插件名列表）"
                )
            load_builtin(pm, group)
        elif isinstance(load_builtins, list):
            load_builtin(pm, load_builtins)

        if plugins:
            for p in plugins:
                pm.register(p)

        if plugin_dir:
            pm.load_directory(plugin_dir)

        self._plugin_manager = pm
        self.checkpointer = checkpointer
        self.event_bus = event_bus
        # 惰性创建的默认会话：不传 session 时复用它，保证多轮对话历史连续
        self._default_session: AgentSession | None = None

    @property
    def plugin_manager(self) -> PluginManager:
        """返回插件管理器实例。"""
        return self._plugin_manager

    def create_runner(self, session: AgentSession) -> AgentRunner:
        """为普通 agent 会话创建执行器。"""
        return AgentRunner(
            llm=self.llm,
            tools=self.tool_registry.list_tools(),
            tool_executor=self.tool_executor,
            session=session,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
            event_bus=self.event_bus,
        )

    async def run(
        self,
        user_input: str,
        session: AgentSession | None = None,
        stream: bool = False,
    ) -> Any:
        """运行一次普通 agent。

        不传 session 时复用 agent 内部的默认会话——
        连续调用 run() 即为多轮对话；如需隔离上下文请显式传入新 session。
        """
        current_session = session
        if current_session is None:
            if self._default_session is None:
                self._default_session = self.create_session()
            current_session = self._default_session
        runner = self.create_runner(current_session)
        return await runner.run(user_input, stream=stream)

    def stream_events(
        self,
        user_input: str,
        session: AgentSession | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流运行一次普通 agent。"""
        current_session = session
        if current_session is None:
            if session_id is not None:
                current_session = self.create_or_get_session(session_id=session_id)
            else:
                if self._default_session is None:
                    self._default_session = self.create_session()
                current_session = self._default_session
        runner = self.create_runner(current_session)
        return runner.stream_events(user_input)

    async def resume(
        self,
        interrupt,
        *,
        approved: bool = True,
        reason: str = "",
        session: AgentSession | None = None,
    ):
        """从 HITL 挂起点恢复执行（需要 checkpointer）。

        - approved=True: 执行挂起的工具调用后继续
        - approved=False: 把拒绝结果回喂模型后继续（模型可调整策略）
        """
        from wuwei.runtime.interrupt import InterruptPayload

        payload = (
            InterruptPayload.from_dict(interrupt) if isinstance(interrupt, dict) else interrupt
        )
        current = session or self._sessions.get(payload.session_id)
        if current is None:
            if self.checkpointer is None:
                raise ValueError("resume requires checkpointer")
            state = await self.checkpointer.load(payload.checkpoint_id)
            saved = state.metadata.get("session")
            if not saved or saved.get("session_id") != payload.session_id:
                raise ValueError("Checkpoint does not contain a matching durable session")
            current = AgentSession.from_dict(saved)
            self._sessions[current.session_id] = current
        runner = self.create_runner(current)
        return await runner.resume(payload, approved=approved, reason=reason)

    async def run_structured(
        self,
        user_input: str,
        response_model: type,
        *,
        max_retries: int = 2,
        session: AgentSession | None = None,
    ):
        """Run with validated JSON output through the same tools, guards and budget.

        Returns (typed output, AgentRunResult). Output is None while interrupted.
        Schema validation retries belong to this run and its cumulative budget.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        current = session
        if current is None:
            if self._default_session is None:
                self._default_session = self.create_session()
            current = self._default_session
        runner = self.create_runner(current)
        runner.response_model = response_model
        runner.max_output_retries = max_retries
        result = await runner.run(user_input)
        output = (
            response_model.model_validate(result.output) if result.status == "completed" else None
        )
        result.output = output
        return output, result

    async def run_plan(
        self, goal: str, *, context: str = "", planner=None, session: AgentSession | None = None
    ):
        """Plan and execute in one governed scope, including planner tokens and history.

        Interrupted task payloads remain on PlanRunResult.tasks; use
        PlanExecutor.resume() with the same checkpointer to continue that DAG.
        """
        import time
        from uuid import uuid4
        from wuwei.planning import PlanExecutor, Planner, TaskList
        from wuwei.graph.state import State
        from wuwei.middleware.base import MiddlewareContext
        from wuwei.runtime.context import current_context

        current = session
        if current is None:
            if self._default_session is None:
                self._default_session = self.create_session()
            current = self._default_session
        async with current.lock:
            if current.metadata.get("_pending_interrupt") or current.metadata.get("_blocked_run"):
                raise ValueError("Session must be resumed/reconciled before planning")
            start = time.monotonic()
            metadata = {
                "run_id": uuid4().hex,
                "session_id": current.session_id,
                "usage_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "llm_calls": 0,
                "latency_ms": 0,
                "permission_state": {},
                "usage_records": [],
                "step_index": 0,
            }
            root = MiddlewareContext(
                state=State(messages=current.context.get_messages()),
                config={
                    "middleware": self.middleware,
                    "checkpointer": self.checkpointer,
                    "event_bus": self.event_bus,
                    "tool_executor": self.tool_executor,
                    "session": current,
                    "parent_context": current_context.get(),
                },
                metadata=metadata,
            )
            token = current_context.set(root)
            try:
                current.context.add_user_message(goal)
                if planner is None or isinstance(planner, Planner):
                    p = planner or Planner(self.llm)
                    planning_agent = Agent(
                        llm=p.llm,
                        load_builtins=False,
                        middleware=self.middleware,
                        checkpointer=self.checkpointer,
                        event_bus=self.event_bus,
                        default_system_prompt=current.system_prompt,
                    )
                    planning_session = planning_agent.create_session()
                    planning_session.context.replace_messages(
                        [m.model_copy(deep=True) for m in current.context.get_messages()[:-1]]
                    )
                    typed, planning_result = await planning_agent.run_structured(
                        p._build_plan_prompt(goal), TaskList, session=planning_session
                    )
                    if planning_result.status != "completed":
                        raise ValueError(f"Planning did not complete: {planning_result.status}")
                    tasks = typed.tasks
                    p.last_usage = planning_result.usage
                    p.last_llm_calls = planning_result.llm_calls
                    p.last_latency_ms = planning_result.latency_ms
                else:
                    # Trusted custom planners retain their extension entry point.
                    p = planner
                    before = metadata["llm_calls"]
                    tasks = await p.plan_task(goal)
                    if before == metadata["llm_calls"]:
                        for key in metadata["usage_total"]:
                            metadata["usage_total"][key] += (
                                getattr(p, "last_usage", {}) or {}
                            ).get(key, 0)
                        metadata["llm_calls"] += getattr(p, "last_llm_calls", 0)
                for task in tasks:
                    task.status = "pending"
                    task.result = task.error = task.interrupt = None
                execution = PlanExecutor(
                    self.llm,
                    tools=self.tool_registry,
                    middleware=self.middleware,
                    checkpointer=self.checkpointer,
                )
                result = await execution.execute(TaskList(tasks=tasks), context=context)
                result.planner_usage = dict(getattr(p, "last_usage", {}) or {})
                result.planner_llm_calls = getattr(p, "last_llm_calls", 0)
                result.planner_latency_ms = getattr(p, "last_latency_ms", 0)
                result.usage = dict(metadata["usage_total"])
                result.llm_calls = metadata["llm_calls"]
                result.latency_ms = int((time.monotonic() - start) * 1000)
                if metadata.get("unknown_tool"):
                    result.status = "unknown"
                current.context.add_ai_message(
                    "计划执行状态："
                    + result.status
                    + "\n"
                    + "\n".join(
                        f"#{t.id} {t.status}: {t.result or t.error or ''}" for t in result.tasks
                    )
                )
                return result
            finally:
                current.last_usage = dict(metadata["usage_total"])
                current.last_llm_calls = metadata["llm_calls"]
                current.last_latency_ms = metadata["latency_ms"]
                current_context.reset(token)

    async def fork(
        self, checkpoint_id: str, *, overrides: dict | None = None, allow_side_effects: bool = False
    ):
        """Time-Travel：从检查点分叉出新 run（可改工具参数/系统提示后重放）。

        见 wuwei.debug 模块。需要 checkpointer。
        """
        from wuwei.debug import agent_fork

        return await agent_fork(
            self, checkpoint_id, overrides=overrides, allow_side_effects=allow_side_effects
        )

    @classmethod
    def from_env(
        cls,
        *,
        tools: list[ToolLike] | None = None,
        system_prompt: str = "你是一个有用的助手",
        max_steps: int = 10,
        parallel_tool_calls: bool = False,
        middleware=None,
        skill_manager=None,
        mcp_manager=None,
        plugins: list[Plugin] | None = None,
        plugin_dir: str | Path | None = None,
        load_builtins: bool | str | list[str] = "minimal",
        checkpointer=None,
        event_bus=None,
        **llm_kwargs,
    ) -> "Agent":
        llm = LLMGateway.from_env(**llm_kwargs)
        registry = ToolRegistry()

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
            middleware=middleware,
            skill_manager=skill_manager,
            mcp_manager=mcp_manager,
            plugins=plugins,
            plugin_dir=plugin_dir,
            load_builtins=load_builtins,
            checkpointer=checkpointer,
            event_bus=event_bus,
        )
