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
from dataclasses import dataclass, field
from typing import Any, Callable

from wuwei.core.message import AIMessage, HumanMessage, SystemMessage, ToolMessage
from wuwei.llm.types import Message
from wuwei.graph.graph import END, CompiledGraph, StateGraph
from wuwei.graph.state import State
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.tools import Tool, ToolRegistry, ToolExecutor


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
        sub_descs = "\n".join(
            f"- {sa.name}: {sa.description}"
            for sa in self.sub_agents.values()
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "start_async_task",
                    "description": f"Start a background sub-agent task. Available agents:\n{sub_descs}\nReturns a task_id for later checking.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": f"Name of the sub-agent to use. One of: {list(self.sub_agents.keys())}",
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
                    "description": "Check the status/result of a background task. If completed, returns the result.",
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
            if hasattr(tool_call, 'function')
            else getattr(tool_call, 'name', '')
        )

        if not tool_name.endswith("_async_task"):
            return tool_call  # 不是子代理工具，放行

        tc_id = (
            tool_call.id
            if hasattr(tool_call, 'id')
            else getattr(tool_call, 'tool_call_id', '')
        )
        args = (
            tool_call.function.arguments
            if hasattr(tool_call, 'function')
            else {}
        )

        result_msg = await self._handle_async_tool(tool_name, args, ctx, tc_id)

        # 将结果写入 state
        if hasattr(ctx.state, 'add_message'):
            ctx.state.add_message(result_msg)
        elif hasattr(ctx.state, 'messages'):
            ctx.state.messages.append(result_msg)

        return None  # 拦截，跳过默认工具执行

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
        elif tool_name == "check_async_task":
            result = await self._check_task(args.get("task_id", ""))
        elif tool_name == "cancel_async_task":
            result = await self._cancel_task(args.get("task_id", ""))
        elif tool_name == "list_async_tasks":
            result = self._list_tasks()
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
    ) -> str:
        """启动异步子代理任务"""
        sub = self.sub_agents.get(agent_name)
        if not sub:
            return json.dumps({"ok": False, "error": f"Unknown agent: {agent_name}"})

        task_id = f"async_{uuid.uuid4().hex[:12]}"
        handle = AsyncTaskHandle(
            task_id=task_id,
            sub_agent_name=agent_name,
            status="running",
            started_at=time.time(),
        )

        # 提取父代理上下文
        parent_context = []
        if sub.inherit_context and hasattr(ctx.state, 'messages'):
            parent_context = list(ctx.state.messages)

        # 创建后台任务
        async def _run_sub_agent():
            try:
                llm = sub.model or self.parent_llm
                result = await self._execute_sub_agent(sub, task, llm, parent_context)
                handle.result = result
                handle.status = "completed"
                handle.completed_at = time.time()
                handle.elapsed_ms = int((handle.completed_at - handle.started_at) * 1000)
            except asyncio.CancelledError:
                handle.status = "cancelled"
                handle.error = "Task cancelled"
            except Exception as e:
                handle.status = "failed"
                handle.error = str(e)
                handle.completed_at = time.time()

        handle._task = asyncio.create_task(_run_sub_agent())
        self._tasks[task_id] = handle

        return json.dumps({
            "ok": True,
            "task_id": task_id,
            "agent_name": agent_name,
            "status": "running",
            "message": f"Task started. Use check_async_task('{task_id}') to get results.",
        }, ensure_ascii=False)

    async def _check_task(self, task_id: str) -> str:
        """查询异步任务状态"""
        handle = self._tasks.get(task_id)
        if not handle:
            return json.dumps({"ok": False, "error": f"Task not found: {task_id}"})

        if handle.status == "running":
            return json.dumps({
                "ok": True,
                "task_id": task_id,
                "status": "running",
                "elapsed_ms": int((time.time() - handle.started_at) * 1000) if handle.started_at else 0,
                "message": "Task is still running. Check again later.",
            }, ensure_ascii=False)
        elif handle.status == "completed":
            return json.dumps({
                "ok": True,
                "task_id": task_id,
                "status": "completed",
                "result": handle.result,
                "elapsed_ms": handle.elapsed_ms,
                "steps": handle.steps,
            }, ensure_ascii=False)
        elif handle.status == "failed":
            return json.dumps({
                "ok": False,
                "task_id": task_id,
                "status": "failed",
                "error": handle.error,
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "ok": True,
                "task_id": task_id,
                "status": handle.status,
            }, ensure_ascii=False)

    async def _cancel_task(self, task_id: str) -> str:
        """取消异步任务"""
        handle = self._tasks.get(task_id)
        if not handle:
            return json.dumps({"ok": False, "error": f"Task not found: {task_id}"})

        if handle.status != "running":
            return json.dumps({"ok": False, "error": f"Task is not running (status: {handle.status})"})

        if handle._task:
            handle._task.cancel()
            try:
                await handle._task
            except asyncio.CancelledError:
                pass

        handle.status = "cancelled"
        return json.dumps({"ok": True, "task_id": task_id, "status": "cancelled"})

    def _list_tasks(self) -> str:
        """列出所有异步任务"""
        tasks = []
        for h in self._tasks.values():
            tasks.append({
                "task_id": h.task_id,
                "agent_name": h.sub_agent_name,
                "status": h.status,
                "elapsed_ms": h.elapsed_ms,
            })
        return json.dumps({"ok": True, "tasks": tasks}, ensure_ascii=False)

    async def _execute_sub_agent(
        self,
        sub_agent: AsyncSubAgent,
        task: str,
        llm: Any,
        parent_context: list | None = None,
    ) -> str:
        """执行子代理（与 SubAgentMiddleware 类似但支持上下文继承）"""
        from wuwei.middleware.stack import MiddlewareStack

        registry = ToolRegistry()
        if sub_agent.tools:
            for tool in sub_agent.tools:
                registry.register(tool)
        executor = ToolExecutor(registry)

        mw_stack = MiddlewareStack()
        if sub_agent.middleware:
            for mw in sub_agent.middleware:
                mw_stack.add(mw)

        all_tools = list(registry.list_tools())

        # 构建消息：系统提示 + 父代理上下文 + 用户任务
        messages = [SystemMessage(content=sub_agent.system_prompt)]
        if parent_context:
            # 注入父代理上下文摘要
            context_summary = self._summarize_context(parent_context)
            if context_summary:
                messages.append(SystemMessage(
                    content=f"[Parent agent context]\n{context_summary}\n[/Parent agent context]"
                ))
        messages.append(HumanMessage(content=task))

        start_time = time.monotonic()
        state = State(messages=messages, metadata={}, step=0)

        # 构建子代理图
        graph = StateGraph(State)
        graph.set_max_steps(sub_agent.max_steps)

        async def agent_node(s: State, config: dict | None = None) -> State:
            msgs = [m for m in s.messages]
            tools = list(all_tools)

            mw_ctx = MiddlewareContext(state=s, config=config or {}, step=s.step)
            mw_ctx = await mw_stack.execute_before_llm(mw_ctx)
            if hasattr(mw_ctx.state, 'messages'):
                msgs = mw_ctx.state.messages

            response = await llm.generate(msgs, tools=tools)

            ai_msg = AIMessage(
                content=response.message.content or "",
                tool_calls=response.message.tool_calls or [],
            )
            mw_ctx = await mw_stack.execute_after_llm(mw_ctx, ai_msg)
            s.add_message(ai_msg)
            s.metadata["_has_tool_calls"] = bool(response.message.tool_calls)
            s.metadata["_tool_calls"] = response.message.tool_calls or []
            s.metadata["_last_content"] = response.message.content or ""
            return s

        async def tool_node(s: State, config: dict | None = None) -> State:
            tool_calls = s.metadata.get("_tool_calls", [])
            mw_ctx = MiddlewareContext(state=s, config=config or {}, step=s.step)

            for tc in tool_calls:
                modified = await mw_stack.execute_before_tool(mw_ctx, tc)
                if modified is None:
                    continue
                try:
                    msg = await executor.execute_one(modified)
                except Exception as exc:
                    msg = Message(role="tool", content=f"Error: {exc}", tool_call_id=modified.id, name=modified.function.name)
                tool_msg = ToolMessage(content=msg.content or "", tool_call_id=msg.tool_call_id or modified.id, name=msg.name or modified.function.name)
                modified_msg = await mw_stack.execute_after_tool(mw_ctx, tool_msg)
                s.add_message(ToolMessage(role="tool", content=modified_msg.content or "", tool_call_id=modified_msg.tool_call_id, name=modified_msg.name))
            return s

        async def should_continue(s: State, config: dict | None = None) -> str:
            return "continue" if s.metadata.get("_has_tool_calls") else "end"

        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
        graph.add_edge("tools", "agent")

        final_state = await graph.compile().invoke(state)
        last_ai = final_state.get_last_ai_message()
        return last_ai.content if last_ai else ""

    def _summarize_context(self, messages: list) -> str:
        """将父代理上下文压缩为简短摘要"""
        if not messages:
            return ""
        parts = []
        for msg in messages[-6:]:  # 最近 6 条
            role = getattr(msg, 'role', '')
            content = getattr(msg, 'content', '') or ''
            if content and role in ('user', 'assistant'):
                parts.append(f"[{role}]: {content[:200]}")
        return "\n".join(parts)
