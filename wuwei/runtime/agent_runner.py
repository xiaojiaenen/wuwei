"""新的 AgentRunner - 使用 Middleware 替代 Hook

基于 StateGraph 的 Agent 运行时执行器。
内部构建标准 Agent 图（llm → tools → llm 循环），
委托给 CompiledGraph 执行。
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from wuwei.agent.session import AgentSession
from wuwei.core.message import AIMessage, ToolMessage as CoreToolMessage
from wuwei.graph.graph import END, CompiledGraph, StateGraph
from wuwei.graph.state import State
from wuwei.llm import (
    AgentEvent,
    AgentRunResult,
    LLMGateway,
    LLMResponse,
    LLMResponseChunk,
    Message,
    ToolCall,
)
from wuwei.middleware.base import MiddlewareContext
from wuwei.middleware.stack import MiddlewareStack
from wuwei.tools import Tool, ToolExecutor

if TYPE_CHECKING:
    from wuwei.planning import Task


class AgentRunner:
    """单个 agent 会话的运行时执行器。

    内部使用 StateGraph 构建标准 Agent 循环图：
    agent_node → (有 tool_calls? → tool_node → agent_node | → END)

    中间件生命周期在每个节点内完整调用。
    """

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool],
        tool_executor: ToolExecutor,
        session: AgentSession,
        middleware: MiddlewareStack | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = tool_executor
        self.session = session
        self.middleware = middleware or MiddlewareStack()

    # ── graph construction ──────────────────────────────────────────

    def _build_graph(self) -> CompiledGraph:
        """构建标准 Agent 循环图。

        Graph 结构:
            agent_node ──(有 tool_calls?)──▶ tool_node ──▶ agent_node
                            │
                            └──(无 tool_calls)──▶ END

        每个节点内部调用完整的中间件生命周期。
        """
        graph = StateGraph(State)
        graph.set_max_steps(self.session.max_steps)

        # Node: agent (LLM 调用)
        async def agent_node(state: State, config: dict | None = None) -> State:
            messages = [m for m in state.messages]
            tools = list(self.tools)

            # before_llm 中间件
            ctx = MiddlewareContext(state=state, config=config or {}, step=state.step)
            ctx = await self.middleware.execute_before_llm(ctx)
            messages = ctx.state.messages if hasattr(ctx.state, 'messages') else messages

            # LLM 调用（通过 wrap_model_call 洋葱模型）
            async def _direct_call(_messages, _tools):
                return await self.llm.generate(_messages, tools=_tools)

            response: LLMResponse = await self.middleware.execute_wrap_model_call(
                ctx, messages, tools, _direct_call
            )

            # 构建 AI 消息
            ai_msg = AIMessage(
                content=response.message.content or "",
                tool_calls=response.message.tool_calls or [],
            )

            # after_llm 中间件
            ctx = await self.middleware.execute_after_llm(ctx, ai_msg)

            state.add_message(ai_msg)
            state.metadata["_last_usage"] = {
                "prompt_tokens": response.usage.get("prompt_tokens", 0) if response.usage else 0,
                "completion_tokens": response.usage.get("completion_tokens", 0) if response.usage else 0,
                "total_tokens": response.usage.get("total_tokens", 0) if response.usage else 0,
            }
            state.metadata["_has_tool_calls"] = bool(response.message.tool_calls)
            state.metadata["_tool_calls"] = response.message.tool_calls or []
            return state

        # Node: tools (执行工具)
        async def tool_node(state: State, config: dict | None = None) -> State:
            tool_calls: list[ToolCall] = state.metadata.get("_tool_calls", [])
            ctx = MiddlewareContext(state=state, config=config or {}, step=state.step)

            for tc in tool_calls:
                # before_tool 中间件
                modified = await self.middleware.execute_before_tool(ctx, tc)
                if modified is None:
                    continue
                tc = modified

                try:
                    msg = await self.tool_executor.execute_one(tc)
                except Exception as exc:
                    msg = Message(
                        role="tool",
                        content=f"工具执行失败: {exc}",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )

                # after_tool 中间件
                tool_msg = CoreToolMessage(
                    content=msg.content or "",
                    tool_call_id=msg.tool_call_id or tc.id,
                    name=msg.name or tc.function.name,
                )
                modified_msg = await self.middleware.execute_after_tool(ctx, tool_msg)

                state.add_message(
                    CoreToolMessage(
                        role="tool",
                        content=modified_msg.content or "",
                        tool_call_id=modified_msg.tool_call_id,
                        name=modified_msg.name,
                    )
                )
            return state

        # Condition: 是否还有 tool calls
        async def should_continue(state: State, config: dict | None = None) -> str:
            if state.metadata.get("_has_tool_calls"):
                return "continue"
            return "end"

        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile()

    # ── public API ──────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        stream: bool = False,
        task: "Task | None" = None,
    ):
        """执行一次 agent 运行。"""
        if stream:
            return self._run_stream(user_input, task=task)
        return await self._run_non_stream(user_input, task=task)

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

    def _set_session_run_stats(
        self,
        *,
        usage: dict[str, int],
        latency_ms: int,
        llm_calls: int,
    ) -> None:
        self.session.last_usage = dict(usage)
        self.session.last_latency_ms = latency_ms
        self.session.last_llm_calls = llm_calls

    def _build_run_result(
        self,
        *,
        content: str,
        usage: dict[str, int],
        latency_ms: int,
        llm_calls: int,
    ) -> AgentRunResult:
        self._set_session_run_stats(
            usage=usage,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
        )
        return AgentRunResult(
            content=content,
            usage=dict(usage),
            latency_ms=latency_ms,
            llm_calls=llm_calls,
        )

    def _build_event(
        self,
        event_type: str,
        *,
        step: int,
        run_id: str | None = None,
        data: dict | None = None,
    ) -> AgentEvent:
        return AgentEvent(
            type=event_type,
            session_id=self.session.session_id,
            step=step,
            run_id=run_id,
            data=data or {},
        )

    async def _emit_event(
        self,
        event_type: str,
        *,
        step: int,
        run_id: str | None = None,
        data: dict | None = None,
    ) -> AgentEvent:
        event = self._build_event(event_type, step=step, run_id=run_id, data=data)
        # Middleware 不直接支持 emit_event，跳过
        return event

    async def stream_events(
        self,
        user_input: str,
        *,
        task: "Task | None" = None,
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流的形式执行一次 agent 运行。

        完整中间件生命周期：
        before_llm → llm.stream() → after_llm → before_tool → tool.execute() → after_tool
        """
        step_count = 0
        llm_calls = 0
        total_latency_ms = 0
        total_usage = self._empty_usage()
        run_id = uuid4().hex
        context = self.session.context
        context.add_user_message(user_input)
        yield self._build_event("run_start", step=0, run_id=run_id, data={"input": user_input})

        try:
            while step_count < self.session.max_steps:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                full_tool_calls = None
                messages = self._copy_messages()
                tools = list(self.tools)

                # --- before_llm 中间件 ---
                ctx = MiddlewareContext(
                    state=None,
                    config={},
                    step=step_count,
                )
                ctx.state = type('State', (), {'messages': messages, 'metadata': {}})()
                ctx = await self.middleware.execute_before_llm(ctx)
                messages = ctx.state.messages

                llm_start = time.monotonic()
                yield self._build_event(
                    "llm_start",
                    step=step_count,
                    run_id=run_id,
                    data={"tools": [tool.name for tool in tools]},
                )
                stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                    messages,
                    tools=tools,
                    stream=True,
                )
                llm_calls += 1

                async for chunk in stream:
                    if chunk.reasoning_content:
                        reasoning_parts.append(chunk.reasoning_content)
                        yield self._build_event(
                            "reasoning_delta",
                            step=step_count,
                            run_id=run_id,
                            data={"content": chunk.reasoning_content},
                        )

                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield self._build_event(
                            "text_delta",
                            step=step_count,
                            run_id=run_id,
                            data={"content": chunk.content},
                        )

                    self._merge_usage(total_usage, chunk.usage)

                    if chunk.tool_calls_complete:
                        full_tool_calls = chunk.tool_calls_complete

                total_latency_ms += int((time.monotonic() - llm_start) * 1000)
                yield self._build_event(
                    "llm_end",
                    step=step_count,
                    run_id=run_id,
                    data={
                        "latency_ms": total_latency_ms,
                        "usage": dict(total_usage),
                        "has_tool_calls": bool(full_tool_calls),
                    },
                )

                # 构建 AIMessage 并调用 after_llm 中间件
                from wuwei.core.message import AIMessage as AIMsg
                ai_msg = AIMsg(
                    content="".join(content_parts),
                    tool_calls=full_tool_calls or [],
                    reasoning_content="".join(reasoning_parts) or None,
                )
                ctx = await self.middleware.execute_after_llm(ctx, ai_msg)

                context.add_ai_message(
                    "".join(content_parts),
                    tool_calls=full_tool_calls,
                    reasoning_content="".join(reasoning_parts) or None,
                )

                if full_tool_calls:
                    # --- 工具执行（含 before_tool / after_tool 中间件） ---
                    for tool_call in full_tool_calls:
                        # before_tool 中间件
                        modified_call = await self.middleware.execute_before_tool(ctx, tool_call)
                        if modified_call is None:
                            continue
                        tool_call = modified_call

                        tool = self.tool_executor.registry.get(tool_call.function.name)
                        yield self._build_event(
                            "tool_start",
                            step=step_count,
                            run_id=run_id,
                            data={
                                "tool_name": tool_call.function.name,
                                "display_name": tool.display_name if tool else None,
                                "args": tool_call.function.arguments,
                                "tool_call_id": tool_call.id,
                            },
                        )

                        tool_message = await self._execute_one_tool_call(
                            tool_call,
                            step=step_count,
                            task=task,
                            run_id=run_id,
                        )

                        # after_tool 中间件
                        from wuwei.core.message import ToolMessage as TMsg
                        tool_msg = TMsg(
                            content=tool_message.content or "",
                            tool_call_id=tool_message.tool_call_id or "",
                            name=tool_message.name or "",
                        )
                        modified_msg = await self.middleware.execute_after_tool(ctx, tool_msg)
                        tool_message = Message(
                            role="tool",
                            content=modified_msg.content or "",
                            tool_call_id=modified_msg.tool_call_id,
                            name=modified_msg.name,
                        )
                        self.session.context.add_tool_message(
                            tool_message.content or "",
                            tool_message.tool_call_id,
                        )

                        yield self._build_event(
                            "tool_end",
                            step=step_count,
                            run_id=run_id,
                            data={
                                "tool_name": tool_call.function.name,
                                "tool_call_id": tool_call.id,
                                "output": tool_message.content,
                            },
                        )

                        error_message = self.tool_executor.extract_error_message(
                            tool_message.content
                        )
                        if error_message:
                            yield self._build_event(
                                "error",
                                step=step_count,
                                run_id=run_id,
                                data={
                                    "message": error_message,
                                    "tool_name": tool_call.function.name,
                                    "tool_call_id": tool_call.id,
                                },
                            )

                    step_count += 1
                    continue

                done_event = self._build_event(
                    "done",
                    step=step_count,
                    run_id=run_id,
                    data={
                        "usage": dict(total_usage),
                        "latency_ms": total_latency_ms,
                        "llm_calls": llm_calls,
                    },
                )
                yield done_event
                yield self._build_event(
                    "run_end", step=step_count, run_id=run_id, data=dict(done_event.data)
                )
                self._set_session_run_stats(
                    usage=total_usage,
                    latency_ms=total_latency_ms,
                    llm_calls=llm_calls,
                )
                return

            limit_message = "任务未完成，已达到最大步骤限制。"
            context.add_ai_message(limit_message)
            yield self._build_event(
                "text_delta",
                step=step_count,
                run_id=run_id,
                data={"content": limit_message},
            )
            done_event = self._build_event(
                "done",
                step=step_count,
                run_id=run_id,
                data={
                    "reason": "max_steps",
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
            )
            yield done_event
            yield self._build_event(
                "run_end", step=step_count, run_id=run_id, data=dict(done_event.data)
            )
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
        except Exception as exc:
            self._set_session_run_stats(
                usage=total_usage,
                latency_ms=total_latency_ms,
                llm_calls=llm_calls,
            )
            # --- on_error 中间件 ---
            ctx = MiddlewareContext(state=None, config={}, step=llm_calls)
            ctx.state = type('State', (), {'messages': [], 'metadata': {}})()
            await self.middleware.execute_on_error(ctx, exc)
            yield self._build_event(
                "error",
                step=step_count,
                run_id=run_id,
                data={
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "usage": dict(total_usage),
                    "latency_ms": total_latency_ms,
                    "llm_calls": llm_calls,
                },
            )

    def _copy_messages(self) -> list[Message]:
        return [message.model_copy(deep=True) for message in self.session.context.get_messages()]

    def _append_tool_messages(self, tool_messages: list[Message]) -> None:
        """把工具输出写回当前会话上下文。"""
        for tool_message in tool_messages:
            self.session.context.add_tool_message(
                tool_message.content or "", tool_message.tool_call_id
            )

    def _iter_tool_feedback_chunks(self, tool_messages: list[Message]):
        """把工具错误转换成流式 chunk，便于上层统一消费。"""
        for tool_message in tool_messages:
            error_message = self.tool_executor.extract_error_message(tool_message.content)
            if error_message:
                yield LLMResponseChunk(content=f"\n[工具执行错误] {error_message}\n")

    async def _execute_one_tool_call(
        self,
        tool_call: ToolCall,
        *,
        step: int,
        task: "Task | None" = None,
        run_id: str | None = None,
    ) -> Message:
        tool = self.tool_executor.registry.get(tool_call.function.name)
        await self._emit_event(
            "tool_start",
            step=step,
            run_id=run_id,
            data={
                "tool_name": tool_call.function.name,
                "display_name": tool.display_name if tool else None,
                "args": tool_call.function.arguments,
                "tool_call_id": tool_call.id,
                "side_effect": bool(tool and tool.execution.side_effect),
                "requires_approval": bool(tool and tool.execution.requires_approval),
            },
        )
        try:
            tool_message = await self.tool_executor.execute_one(tool_call)
        except Exception as exc:
            tool_message = Message(
                role="tool",
                content=f"工具执行失败: {exc}",
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
            )

        await self._emit_event(
            "tool_end",
            step=step,
            run_id=run_id,
            data={
                "tool_name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "output": tool_message.content,
            },
        )
        return tool_message

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        step: int,
        task: "Task | None" = None,
        run_id: str | None = None,
    ) -> list[Message]:
        """执行所有工具调用（不使用中间件）"""
        tool_messages = []
        for tool_call in tool_calls:
            tool_message = await self._execute_one_tool_call(
                tool_call,
                step=step,
                task=task,
                run_id=run_id,
            )
            tool_messages.append(tool_message)
        return tool_messages

    async def _execute_tool_calls_with_middleware(
        self,
        tool_calls: list[ToolCall],
        *,
        ctx: MiddlewareContext,
        step: int,
        task: "Task | None" = None,
        run_id: str | None = None,
    ) -> list[Message]:
        """执行所有工具调用（使用完整中间件生命周期）

        每个工具调用经过：
        before_tool → tool.execute() → after_tool
        """
        from wuwei.core.message import ToolMessage as TMsg

        tool_messages = []
        for tool_call in tool_calls:
            # --- before_tool 中间件 ---
            modified_call = await self.middleware.execute_before_tool(ctx, tool_call)
            if modified_call is None:
                # 中间件拦截了工具调用
                continue

            # 执行工具
            tool_message = await self._execute_one_tool_call(
                modified_call,
                step=step,
                task=task,
                run_id=run_id,
            )

            # --- after_tool 中间件 ---
            tool_msg = TMsg(
                content=tool_message.content or "",
                tool_call_id=tool_message.tool_call_id or "",
                name=tool_message.name or "",
            )
            modified_msg = await self.middleware.execute_after_tool(ctx, tool_msg)
            tool_messages.append(
                Message(
                    role="tool",
                    content=modified_msg.content or "",
                    tool_call_id=modified_msg.tool_call_id,
                    name=modified_msg.name,
                )
            )

        return tool_messages

    async def _run_non_stream(
        self,
        user_input: str,
        task: "Task | None" = None,
    ) -> AgentRunResult:
        """非流式执行 — 委托给 StateGraph。

        构建标准 agent 图，将 user_input 注入 State，
        由 CompiledGraph.invoke() 驱动完整执行循环。
        中间件生命周期在图的节点内部调用。
        """
        start_time = time.monotonic()
        total_usage = self._empty_usage()
        llm_calls = 0

        # 从 session context 构建初始 State
        self.session.context.add_user_message(user_input)
        state = State(
            messages=list(self._copy_messages()),
            metadata={},
            step=0,
        )

        try:
            graph = self._build_graph()
            state = await graph.invoke(state)

            # 统计 LLM 调用次数和 token
            llm_calls = state.step + 1
            usage = state.metadata.get("_last_usage", {})
            self._merge_usage(total_usage, usage)

            # 获取最终 AI 消息
            last_ai = state.get_last_ai_message()
            content = last_ai.content if last_ai else ""

            # 同步 state 回 session context
            self._sync_state_to_context(state)

            latency_ms = int((time.monotonic() - start_time) * 1000)
            return self._build_run_result(
                content=content,
                usage=total_usage,
                latency_ms=latency_ms,
                llm_calls=llm_calls,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            ctx = MiddlewareContext(state=None, config={}, step=llm_calls)
            ctx.state = type('State', (), {'messages': [], 'metadata': {}})()
            handled = await self.middleware.execute_on_error(ctx, exc)
            if handled is not None:
                raise
            return self._build_run_result(
                content=f"执行出错: {exc}",
                usage=total_usage,
                latency_ms=latency_ms,
                llm_calls=llm_calls,
            )

    def _sync_state_to_context(self, state: State) -> None:
        """将 State 中的消息同步回 session context。"""
        from wuwei.core.message import AIMessage as AIMsg, ToolMessage as TMsg
        existing_count = len(self.session.context.get_messages())
        for i, msg in enumerate(state.messages):
            if i < existing_count:
                continue
            self.session.context._messages.append(msg)
