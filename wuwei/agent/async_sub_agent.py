"""异步子代理系统

借鉴 DeepAgents 的 AsyncSubAgent 设计：
- AsyncSubAgent：可在后台异步执行的子代理
- 支持 start/check/cancel/list 操作
- 子代理继承父代理上下文（context inheritance）
- 父代理不阻塞，可并行委派多个子代理

通信模式：
    Parent Agent
        │
        ├─ start_async_task(researcher, "查天气") → task_id
        ├─ start_async_task(calculator, "算123*456") → task_id  ← 并行！
        │  （父代理继续执行其他任务...）
        ├─ check_async_task(task_id_1) → {"status": "completed", "result": "..."}
        ├─ check_async_task(task_id_2) → {"status": "completed", "result": "..."}
        └─ 综合结果 → 最终回答
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from wuwei.core.message import ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.tools import Tool


@dataclass
class AsyncSubAgent:
    """异步子代理配置

    与 SubAgent 的区别：不在调用时阻塞父代理，
    而是以后台任务方式执行，父代理可通过 check_async_task 查询进度。

    Attributes:
        name: 子代理名称
        description: 描述（LLM 决定何时委派）
        system_prompt: 子代理的系统提示
        tools: 可用工具列表
        model: 可选独立模型
        max_steps: 最大步数
        middleware: 额外中间件
        inherit_context: 是否继承父代理对话历史
    """

    name: str
    description: str
    system_prompt: str = "You are a helpful assistant."
    tools: list[Tool] | None = None
    model: Any | None = None
    max_steps: int = 10
    middleware: list[Middleware] | None = None
    inherit_context: bool = True
    """是否将父代理的对话历史注入子代理上下文"""


@dataclass
class AsyncTaskHandle:
    """异步任务句柄 — 追踪后台子代理执行状态"""

    task_id: str
    sub_agent_name: str
    status: str = "pending"  # pending | running | completed | failed | cancelled
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    elapsed_ms: int = 0
    steps: int = 0
    usage: dict = field(default_factory=dict)
    llm_calls: int = 0
    interrupt: dict | None = None
    reconciliation: dict | None = None
    owner_session: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class AsyncSubAgentMiddleware(Middleware):
    """异步子代理中间件

    暴露 4 个工具给父代理 LLM：
    - start_async_task: 启动后台子代理
    - check_async_task: 查询任务状态/结果
    - cancel_async_task: 取消运行中的任务
    - list_async_tasks: 列出所有任务

    使用 wrap_model_call 注入工具 schema，
    使用 before_tool 拦截工具调用。
    """

    is_tool_handler = True

    def __init__(
        self,
        sub_agents: list[AsyncSubAgent],
        parent_llm: Any,
    ):
        self.sub_agents: dict[str, AsyncSubAgent] = {}
        self.parent_llm = parent_llm
        self._tasks: dict[str, AsyncTaskHandle] = {}
        for sa in sub_agents:
            self.sub_agents[sa.name] = sa

    def get_task_tools(self) -> list[dict]:
        """获取异步子代理工具 schema"""
        sub_descs = "\n".join(f"- {sa.name}: {sa.description}" for sa in self.sub_agents.values())
        return [
            {
                "type": "function",
                "function": {
                    "name": "resume_async_task",
                    "description": "Resume an interrupted task with an approval decision.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "approved": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["task_id", "approved"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "start_async_task",
                    "description": (
                        f"Start a background sub-agent task. Available agents:\n{sub_descs}"
                        "\nReturns a task_id for later checking."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": (
                                    "Name of the sub-agent to use. One of: "
                                    f"{list(self.sub_agents.keys())}"
                                ),
                                "enum": list(self.sub_agents.keys()),
                            },
                            "task": {
                                "type": "string",
                                "description": "Task description for the sub-agent",
                            },
                        },
                        "required": ["agent_name", "task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_async_task",
                    "description": (
                        "Check background task status. Completed tasks include the result."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "The task_id returned by start_async_task",
                            },
                        },
                        "required": ["task_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_async_task",
                    "description": "Cancel a running background task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "The task_id to cancel",
                            },
                        },
                        "required": ["task_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_async_tasks",
                    "description": "List all background tasks and their statuses.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    async def wrap_model_call(
        self,
        ctx: MiddlewareContext,
        messages: list[Any],
        tools: list[Any],
        next_handler: Any,
    ) -> Any:
        """注入异步子代理工具 schema"""
        sub_schemas = self.get_task_tools()
        return await next_handler(messages, list(tools) + sub_schemas)

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: Any,
    ) -> Any:
        """拦截异步子代理工具调用"""
        tool_name = (
            tool_call.function.name
            if hasattr(tool_call, "function")
            else getattr(tool_call, "name", "")
        )

        if tool_name not in {
            "resume_async_task",
            "start_async_task",
            "check_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }:
            return tool_call  # 不是子代理工具，放行

        tc_id = tool_call.id if hasattr(tool_call, "id") else getattr(tool_call, "tool_call_id", "")
        args = tool_call.function.arguments if hasattr(tool_call, "function") else {}

        result_msg = await self._handle_async_tool(tool_name, args, ctx, tc_id)

        return result_msg

    @staticmethod
    def _owner_session(ctx: MiddlewareContext):
        return ctx.metadata.get("session_id") or getattr(ctx.state, "metadata", {}).get(
            "session_id"
        )

    async def _handle_async_tool(
        self,
        tool_name: str,
        args: dict,
        ctx: MiddlewareContext,
        tc_id: str,
    ) -> ToolMessage:
        """分发处理异步工具调用"""
        if tool_name == "start_async_task":
            result = await self._start_task(
                args.get("agent_name", ""),
                args.get("task", ""),
                ctx,
            )
        elif tool_name in {"check_async_task", "cancel_async_task", "resume_async_task"} and (
            args.get("task_id") in self._tasks
            and self._tasks[args["task_id"]].owner_session != self._owner_session(ctx)
        ):
            result = json.dumps({"ok": False, "error": "Task not found in this session"})
        elif tool_name == "resume_async_task":
            handle = self._tasks.get(args.get("task_id", ""))
            if handle is None or handle.status != "interrupted" or not handle.interrupt:
                result = json.dumps({"ok": False, "error": "Task is not interrupted"})
            else:
                ctx.metadata["continuation_decision"] = {
                    "approved": args.get("approved", True),
                    "reason": args.get("reason", ""),
                }
                result = await self._start_task(
                    handle.sub_agent_name,
                    "Resume task",
                    ctx,
                    resume=handle.interrupt,
                    task_id=handle.task_id,
                )
        elif tool_name == "check_async_task":
            result = await self._check_task(args.get("task_id", ""))
        elif tool_name == "cancel_async_task":
            result = await self._cancel_task(args.get("task_id", ""))
        elif tool_name == "list_async_tasks":
            result = self._list_tasks(self._owner_session(ctx))
        else:
            result = json.dumps({"ok": False, "error": f"Unknown tool: {tool_name}"})

        return ToolMessage(
            content=result,
            tool_call_id=tc_id,
            name=tool_name,
        )

    async def _start_task(
        self,
        agent_name: str,
        task: str,
        ctx: MiddlewareContext,
        *,
        resume=None,
        task_id=None,
    ) -> str:
        """启动异步子代理任务"""
        sub = self.sub_agents.get(agent_name)
        if not sub:
            return json.dumps({"ok": False, "error": f"Unknown agent: {agent_name}"})

        task_id = task_id or f"async_{uuid.uuid4().hex[:12]}"
        handle = AsyncTaskHandle(
            task_id=task_id,
            sub_agent_name=agent_name,
            status="running",
            started_at=time.time(),
            owner_session=self._owner_session(ctx),
        )

        if not task.strip():
            return json.dumps({"ok": False, "error": "Missing task"})

        # 提取父代理上下文
        parent_context = []
        if sub.inherit_context and hasattr(ctx.state, "messages"):
            parent_context = list(ctx.state.messages)

        # 创建后台任务
        async def _run_sub_agent():
            try:
                llm = sub.model or self.parent_llm
                result = await self._execute_sub_agent(
                    sub, task, llm, parent_context, ctx=ctx, resume=resume
                )
                payload = json.loads(result)
                handle.result = payload.get("result")
                handle.status = payload.get("status", "unknown")
                handle.usage = payload.get("usage", {})
                handle.llm_calls = payload.get("llm_calls", 0)
                handle.interrupt = payload.get("interrupt")
                handle.reconciliation = payload.get("reconciliation")
                handle.steps = payload.get("steps", 0)
                handle.completed_at = time.time()
                handle.elapsed_ms = int((handle.completed_at - handle.started_at) * 1000)
            except asyncio.CancelledError as exc:
                handle.usage = getattr(exc, "child_usage", {})
                handle.llm_calls = getattr(exc, "child_llm_calls", 0)
                handle.reconciliation = getattr(exc, "child_unknown", None)
                handle.status = "unknown" if handle.reconciliation else "cancelled"
                handle.error = (
                    "Tool outcome requires reconciliation"
                    if handle.reconciliation
                    else "Task cancelled"
                )
            except Exception as e:
                handle.reconciliation = getattr(e, "child_unknown", None)
                handle.status = "unknown" if handle.reconciliation else "failed"
                handle.error = str(e)
                handle.usage = getattr(e, "child_usage", {})
                handle.llm_calls = getattr(e, "child_llm_calls", 0)
                handle.completed_at = time.time()
            finally:
                handle.completed_at = time.time()
                handle.elapsed_ms = int((handle.completed_at - handle.started_at) * 1000)

        import contextvars

        from wuwei.runtime.context import current_context

        background_context = contextvars.copy_context()
        background_context.run(current_context.set, ctx)
        handle._task = asyncio.create_task(_run_sub_agent(), context=background_context)
        ctx.metadata.setdefault("background_tasks", []).append(handle._task)
        self._tasks[task_id] = handle

        return json.dumps(
            {
                "ok": True,
                "task_id": task_id,
                "agent_name": agent_name,
                "status": "running",
                "message": f"Task started. Use check_async_task('{task_id}') to get results.",
            },
            ensure_ascii=False,
        )

    async def _check_task(self, task_id: str) -> str:
        """查询异步任务状态"""
        handle = self._tasks.get(task_id)
        if not handle:
            return json.dumps({"ok": False, "error": f"Task not found: {task_id}"})

        return json.dumps(
            {
                "ok": handle.status not in {"failed", "error", "cancelled", "unknown"},
                "task_id": task_id,
                "status": handle.status,
                "result": handle.result,
                "error": handle.error,
                "usage": handle.usage,
                "llm_calls": handle.llm_calls,
                "interrupt": handle.interrupt,
                "reconciliation": handle.reconciliation,
                "steps": handle.steps,
                "elapsed_ms": (
                    int((time.time() - handle.started_at) * 1000)
                    if handle.status == "running"
                    else handle.elapsed_ms
                ),
            },
            ensure_ascii=False,
        )

    async def _cancel_task(self, task_id: str) -> str:
        """取消异步任务"""
        handle = self._tasks.get(task_id)
        if not handle:
            return json.dumps({"ok": False, "error": f"Task not found: {task_id}"})

        if handle.status != "running":
            return json.dumps(
                {"ok": False, "error": f"Task is not running (status: {handle.status})"}
            )

        if handle._task:
            handle._task.cancel()
            with suppress(asyncio.CancelledError):
                await handle._task

        if handle.status != "unknown":
            handle.status = "cancelled"
        return json.dumps(
            {
                "ok": True,
                "task_id": task_id,
                "status": handle.status,
                "reconciliation": handle.reconciliation,
            }
        )

    def _list_tasks(self, owner_session=None) -> str:
        """列出所有异步任务"""
        tasks = []
        for h in self._tasks.values():
            if owner_session is not None and h.owner_session != owner_session:
                continue
            tasks.append(
                {
                    "task_id": h.task_id,
                    "agent_name": h.sub_agent_name,
                    "status": h.status,
                    "elapsed_ms": h.elapsed_ms,
                }
            )
        return json.dumps({"ok": True, "tasks": tasks}, ensure_ascii=False)

    async def _execute_sub_agent(
        self,
        sub_agent: AsyncSubAgent,
        task: str,
        llm: Any,
        parent_context: list | None = None,
        ctx: MiddlewareContext | None = None,
        resume: dict | None = None,
    ) -> str:
        """执行子代理

        v3.2 起复用 SubAgentMiddleware 的标准执行循环（消除两份
        90% 重复的图构建代码），异步语义只体现在任务句柄层。
        """
        from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware

        task_text = task
        if parent_context:
            # 最近消息的硬截断预览（非摘要——避免二次 LLM 调用的延迟与成本）
            context_summary = self._summarize_context(parent_context)
            if context_summary:
                task_text = (
                    f"{task}\n\n[Parent agent context]\n{context_summary}\n[/Parent agent context]"
                )

        sync_sub = SubAgent(
            name=sub_agent.name,
            description=getattr(sub_agent, "description", ""),
            system_prompt=sub_agent.system_prompt,
            tools=list(sub_agent.tools) if sub_agent.tools else None,
            model=sub_agent.model,
            max_steps=sub_agent.max_steps,
            middleware=list(sub_agent.middleware) if sub_agent.middleware else None,
        )
        bridge = SubAgentMiddleware(sub_agents=[sync_sub], parent_llm=llm)
        if ctx is not None:
            from wuwei.agent.sub_agent import inherit_middleware

            ctx = MiddlewareContext(
                state=ctx.state,
                metadata=ctx.metadata,
                config={
                    **ctx.config,
                    "middleware": inherit_middleware(ctx.config.get("middleware"), exclude=self),
                },
            )
        return await bridge._run_sub_agent(sync_sub, task_text, llm, ctx=ctx, resume=resume)

    def _summarize_context(self, messages: list) -> str:
        """将父代理上下文压缩为最近消息的截断预览（非 LLM 摘要）。"""
        if not messages:
            return ""
        parts = []
        for msg in messages[-6:]:  # 最近 6 条
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "") or ""
            if content and role in ("user", "assistant"):
                parts.append(f"[{role}]: {content[:200]}")
        return "\n".join(parts)
