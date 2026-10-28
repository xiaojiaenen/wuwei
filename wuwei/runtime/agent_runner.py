"""One event-driven execution engine for ordinary, streamed and delegated runs.

Run governance is JSON state; middleware instances hold configuration only.
Every tool result is committed before advancing. Uncertain side effects are not
replayed, and an approval consumes one exact call rather than skipping guards.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from wuwei.agent.session import AgentSession
from wuwei.core.message import AIMessage, ToolMessage
from wuwei.graph.state import State
from wuwei.llm import AgentEvent, AgentRunResult, LLMResponse, Message, ToolCall
from wuwei.middleware.base import MiddlewareContext
from wuwei.middleware.stack import MiddlewareStack
from wuwei.runtime.action_log import get_or_create_log
from wuwei.runtime.context import current_context
from wuwei.runtime.interrupt import InterruptPayload, InterruptSignal
from wuwei.tools.executor import bind_tool_execution


def _empty_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(total, usage):
    for key, value in (usage or {}).items():
        total[key] = total.get(key, 0) + (value or 0)


class AgentRunner:
    def __init__(
        self,
        llm,
        tools,
        tool_executor,
        session: AgentSession,
        middleware=None,
        checkpointer=None,
        event_bus=None,
    ):
        self.llm = llm
        self.tools = tools  # compatible constructor; effective tools come from registry per call
        self.tool_executor = tool_executor
        self.session = session
        self.middleware = middleware if middleware is not None else MiddlewareStack()
        self.checkpointer = checkpointer
        self.event_bus = event_bus
        self._metadata = {}
        self._parent = None
        self.response_model = None
        self.max_output_retries = 2
        self._resources = {}
        self._last_ctx = None

    async def run(self, user_input, stream=False, task=None):
        if stream:
            return self._iter_events(user_input, stream=True, task=task)
        return await self._collect(self._iter_events(user_input, stream=False, task=task))

    def stream_events(self, user_input, *, task=None):
        return self._iter_events(user_input, stream=True, task=task)

    async def _collect(self, events):
        result = None
        async for event in events:
            if event.type == "run_end":
                result = AgentRunResult(**event.data)
        if result is None:
            raise RuntimeError("Run ended without a committed terminal result")
        return result

    async def resume(self, interrupt, *, approved=True, reason="", state=None):
        payload = (
            InterruptPayload.from_dict(interrupt) if isinstance(interrupt, dict) else interrupt
        )
        if self.checkpointer is None:
            raise ValueError("resume requires checkpointer")
        loaded = state if state is not None else await self.checkpointer.load(payload.checkpoint_id)
        if loaded.metadata.get("session_id") != self.session.session_id:
            raise ValueError("Interrupt session does not match checkpoint")
        kind = loaded.metadata.get("kind")
        if kind not in ("interrupt", "pre_tool", "step", "tool_result", "fork"):
            raise ValueError("Checkpoint is not safely resumable; tool outcome may be unknown")
        pending = loaded.metadata.get("pending_tool_calls", [])
        if kind == "interrupt":
            target = loaded.metadata.get("approval_call") or (pending[0] if pending else None)
            if target is None:
                raise ValueError("Interrupt checkpoint has no pending approval")
            expected = ToolCall.model_validate(target)
            if (
                payload.tool_name != expected.function.name
                or payload.tool_args != expected.function.arguments
            ):
                raise ValueError("Approval arguments do not match the saved call")
            if payload.tool_call_id and payload.tool_call_id != expected.id:
                raise ValueError("Approval tool_call_id does not match checkpoint")
            if payload.run_id and payload.run_id != loaded.metadata.get("run_id"):
                raise ValueError("Approval run does not match checkpoint")
        saved_run = loaded.metadata.get("run_state") or {}
        cached = saved_run.get("batch_results", {})
        if any(
            saved_run.get("tool_states", {}).get(tc["id"]) in ("running", "unknown")
            and tc["id"] not in cached
            for tc in pending
        ):
            raise ValueError("Pending tool outcome unknown; reconcile instead of replay")
        if not await self.checkpointer.claim(payload.checkpoint_id):
            raise ValueError("Checkpoint already consumed or superseded; 不可重复恢复")
        return await self._collect(
            self._iter_events(
                None,
                stream=False,
                resume={
                    "state": loaded,
                    "payload": payload,
                    "approved": approved,
                    "reason": reason,
                },
            )
        )

    async def _iter_events(self, user_input, *, stream, task=None, resume=None):
        async with self.session.lock:
            if resume is None and self.session.metadata.get("_blocked_run"):
                raise ValueError(
                    "Session has an unknown tool outcome; reconcile externally before resetting the session"
                )
            if resume is None and self.session.metadata.get("_pending_interrupt"):
                raise ValueError(
                    "Session has an interrupted run; resume it before starting another run"
                )
            async for event in self._execute_events(
                user_input, stream=stream, task=task, resume=resume
            ):
                yield event

    def _ctx(self):
        meta = self._metadata
        ctx = MiddlewareContext(
            state=State(
                messages=self._copy_messages(),
                step=meta["step_index"],
                metadata={"session_id": self.session.session_id, "run_id": meta["run_id"]},
            ),
            config={
                "tool_executor": self.tool_executor,
                "checkpointer": self.checkpointer,
                "middleware": self.middleware,
                "event_bus": self.event_bus,
                "session": self.session,
                "parent_context": self._parent,
                "_runtime_resources": self._resources,
            },
            step=meta["step_index"],
            metadata=meta,
        )
        ctx.config["model_call"] = lambda messages, tools=None: self._middleware_model_call(
            ctx, messages, tools
        )
        self._last_ctx = ctx
        return ctx

    async def _middleware_model_call(self, ctx, messages, tools=None):
        from wuwei.middleware.budget import BudgetMiddleware

        budgets = [mw for mw in self.middleware.middlewares if isinstance(mw, BudgetMiddleware)]
        for mw in budgets:
            await mw.before_llm(ctx)
        started = time.monotonic()
        response = await self.llm.generate(messages, tools=tools, stream=False)
        self._charge_usage(
            response.usage,
            int((time.monotonic() - started) * 1000),
            response.model,
            getattr(response, "usage_available", True),
        )
        for mw in budgets:
            await mw.after_llm(ctx, response.message)
        return response

    async def _close_resources(self):
        ctx = self._last_ctx
        if ctx is not None:
            for middleware in reversed(self.middleware.middlewares):
                close = getattr(middleware, "aclose", None)
                if close is not None:
                    await close(ctx)

    async def _in_context(self, ctx, fn, *args):
        token = current_context.set(ctx)
        try:
            return await fn(*args)
        finally:
            current_context.reset(token)

    async def _save_checkpoint(self, *, kind, pending=None, approval_call=None, continuation=None):
        if self.checkpointer is None:
            return None
        meta = self._metadata
        metadata = {
            "schema_version": 1,
            "kind": kind,
            "session_id": self.session.session_id,
            "run_id": meta["run_id"],
            "step": meta["step_index"],
            "run_state": {k: v for k, v in meta.items() if k != "background_tasks"},
            "session": self.session.to_dict(),
            "pending_tool_calls": [tc.model_dump(mode="json") for tc in pending or []],
        }
        if approval_call is not None:
            metadata["approval_call"] = approval_call.model_dump(mode="json")
        if continuation is not None:
            metadata["continuation"] = continuation
        ancestor = self._parent
        if ancestor is not None:
            while ancestor.config.get("parent_context") is not None:
                ancestor = ancestor.config["parent_context"]
            metadata["ancestor_state"] = {
                k: v
                for k, v in ancestor.metadata.items()
                if k not in ("background_tasks", "ancestor_state")
            }
        state = State(messages=self._copy_messages(), metadata=metadata, step=meta["step_index"])
        return await self.checkpointer.save(state, metadata=metadata)

    def _charge_usage(self, usage, latency_ms, model, available=True):
        meta = self._metadata
        _merge_usage(meta["usage_total"], usage)
        meta["usage_available"] = meta.get("usage_available", True) and available
        meta["llm_calls"] += 1
        meta["latency_ms"] += latency_ms
        record = {"model": model, "usage": dict(usage or {}), "latency_ms": latency_ms}
        meta.setdefault("usage_records", []).append(record)
        parent = self._parent
        while parent is not None:
            _merge_usage(parent.metadata["usage_total"], usage)
            parent.metadata["usage_available"] = (
                parent.metadata.get("usage_available", True) and available
            )
            parent.metadata["llm_calls"] += 1
            parent.metadata["latency_ms"] += latency_ms
            parent.metadata.setdefault("usage_records", []).append(record)
            parent = parent.config.get("parent_context")

    async def _execute_events(self, user_input, *, stream, task=None, resume=None):
        self._parent = current_context.get()
        run_id = uuid4().hex
        self._metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "session_id": self.session.session_id,
            "step_index": 0,
            "usage_total": _empty_usage(),
            "llm_calls": 0,
            "latency_ms": 0,
            "permission_state": {},
            "approval_decisions": {},
            "tool_states": {},
            "usage_records": [],
            "status": "running",
        }
        pending = []
        if resume:
            saved = resume["state"]
            if self._parent is None and saved.metadata.get("ancestor_state"):
                self._parent = MiddlewareContext(
                    state=State(), config={}, metadata=saved.metadata["ancestor_state"]
                )
            self._metadata.update(saved.metadata.get("run_state") or {})
            self._metadata["run_id"] = saved.metadata.get("run_id") or run_id
            self.session.context.replace_messages([m.model_copy(deep=True) for m in saved.messages])
            pending = [
                ToolCall.model_validate(tc) for tc in saved.metadata.get("pending_tool_calls", [])
            ]
            if saved.metadata.get("continuation"):
                self._metadata["continuation_decision"] = {
                    "approved": resume["approved"],
                    "reason": resume["reason"],
                }
            elif saved.metadata.get("kind") == "interrupt":
                tc = ToolCall.model_validate(saved.metadata.get("approval_call") or pending[0])
                self._metadata["approval_decisions"][tc.id] = {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                    "approved": resume["approved"],
                    "reason": resume["reason"],
                }
            self.session.metadata.pop("_pending_interrupt", None)
        else:
            self.session.context.add_user_message(user_input)
        meta = self._metadata
        if self.response_model is not None:
            meta["structured_schema"] = self.response_model.model_json_schema()
            meta["output_max_retries"] = self.max_output_retries
        if self._parent is not None:
            meta["permission_state"] = self._parent.metadata.setdefault("permission_state", {})
        run_id = meta["run_id"]
        final_content = ""
        status = "completed"
        termination_reason = None
        yield self._event("run_start", data={"resumed": bool(resume), "input": user_input})
        try:
            while meta["step_index"] < self.session.max_steps:
                ctx = self._ctx()
                if not pending:
                    ctx = await self._in_context(ctx, self.middleware.execute_before_llm, ctx)
                    if meta.get("structured_schema"):
                        from wuwei.core.message import SystemMessage

                        ctx.state.messages.insert(
                            0,
                            SystemMessage(
                                content="只输出符合以下JSON Schema的JSON对象："
                                + json.dumps(meta["structured_schema"], ensure_ascii=False)
                            ),
                        )
                    effective_tools = self.tool_executor.registry.list_tools()
                    yield self._event(
                        "llm_start", data={"tools": [t.name for t in effective_tools]}
                    )
                    start = time.monotonic()
                    call_usage = _empty_usage()
                    content = []
                    reasoning = []
                    provider_meta = {}
                    finish = None
                    model = ""
                    usage_available = False

                    async def invoke(messages, tools):
                        return await self.llm.generate(messages, tools=tools, stream=stream)

                    result = await self._in_context(
                        ctx,
                        self.middleware.execute_wrap_model_call,
                        ctx,
                        ctx.state.messages,
                        effective_tools,
                        invoke,
                    )
                    if isinstance(result, LLMResponse):
                        usage_available = result.usage_available
                        content.append(result.message.content or "")
                        reasoning.append(result.message.reasoning_content or "")
                        pending = result.message.tool_calls or []
                        provider_meta.update(result.message.metadata)
                        _merge_usage(call_usage, result.usage)
                        finish = result.finish_reason
                        model = result.model
                    else:
                        try:
                            async for chunk in result:
                                usage_available = (
                                    usage_available
                                    or getattr(chunk, "usage_available", False)
                                    or any((chunk.usage or {}).values())
                                )
                                if chunk.content:
                                    content.append(chunk.content)
                                    yield self._event("text_delta", data={"content": chunk.content})
                                if chunk.reasoning_content:
                                    reasoning.append(chunk.reasoning_content)
                                    yield self._event(
                                        "reasoning_delta", data={"content": chunk.reasoning_content}
                                    )
                                provider_meta.update(getattr(chunk, "metadata", {}) or {})
                                if chunk.tool_calls_complete:
                                    pending = chunk.tool_calls_complete
                                if chunk.finish_reason:
                                    finish = chunk.finish_reason
                                for key, value in (chunk.usage or {}).items():
                                    call_usage[key] = max(call_usage.get(key, 0), value or 0)
                        finally:
                            close = getattr(result, "aclose", None)
                            if close:
                                await close()
                    latency = int((time.monotonic() - start) * 1000)
                    self._charge_usage(call_usage, latency, model, usage_available)
                    ai_msg = AIMessage(
                        content="".join(content),
                        reasoning_content="".join(reasoning) or None,
                        tool_calls=pending,
                        metadata=provider_meta,
                    )
                    # Store the full provider reasoning representation before later replay.
                    self.session.context.get_messages().append(ai_msg)
                    yield self._event(
                        "llm_end",
                        data={
                            "latency_ms": meta["latency_ms"],
                            "step_latency_ms": latency,
                            "usage": dict(meta["usage_total"]),
                            "has_tool_calls": bool(pending),
                        },
                    )
                    ctx = await self._in_context(
                        ctx, self.middleware.execute_after_llm, ctx, ai_msg
                    )
                    get_or_create_log(self.session).append(
                        run_id=run_id,
                        agent=self.session.session_id,
                        step=meta["step_index"],
                        kind="llm",
                        summary=ai_msg.content[:200],
                        data={"tool_calls": [tc.model_dump(mode="json") for tc in pending]},
                    )
                    final_content = ai_msg.content
                    if finish not in ("stop", "tool_calls"):
                        pending = []  # No partial tools execute on length/abort/unknown.
                        status = (
                            "cancelled"
                            if finish in ("cancelled", "aborted", "abort")
                            else "incomplete"
                        )
                        termination_reason = finish or "stream_incomplete"
                        meta["step_index"] += 1
                        break
                    if not pending:
                        if meta.get("structured_schema"):
                            from jsonschema import validate
                            from wuwei.agent.agent import _extract_json
                            from wuwei.core.errors import LLMError

                            try:
                                data = _extract_json(final_content)
                                if data is None:
                                    raise ValueError("Output is not valid JSON")
                                validate(data, meta["structured_schema"])
                                if self.response_model is not None:
                                    data = self.response_model.model_validate(data).model_dump(
                                        mode="json"
                                    )
                                meta["structured_output"] = data
                            except Exception as validation_error:
                                attempts = meta.get("output_validation_failures", 0) + 1
                                meta["output_validation_failures"] = attempts
                                if attempts > meta.get("output_max_retries", 2):
                                    raise LLMError(
                                        f"结构化输出校验失败: {validation_error}"
                                    ) from validation_error
                                self.session.context.add_user_message(
                                    "输出JSON校验失败，请纠正：" + str(validation_error)[:1000]
                                )
                                meta["step_index"] += 1
                                continue
                        meta["step_index"] += 1
                        break
                    await self._save_checkpoint(kind="pre_tool", pending=pending)
                try:
                    async for event in self._run_tool_phase(ctx=ctx, tool_calls=pending):
                        yield event
                except InterruptSignal as sig:
                    if self.checkpointer is None:
                        raise ValueError("interrupt requires checkpointer") from sig
                    continuation = getattr(sig, "continuation", None)
                    cid = await self._save_checkpoint(
                        kind="interrupt",
                        pending=sig.pending,
                        approval_call=sig.tool_call,
                        continuation=continuation,
                    )
                    payload = InterruptPayload(
                        checkpoint_id=cid,
                        session_id=self.session.session_id,
                        tool_name=sig.tool_call.function.name,
                        tool_args=dict(sig.tool_call.function.arguments),
                        reason=sig.reason,
                        run_id=run_id,
                        tool_call_id=sig.tool_call.id,
                        continuation=continuation,
                    )
                    self.session.metadata["_pending_interrupt"] = cid
                    yield self._event("approval_required", data=payload.to_dict())
                    async for event in self._finish(
                        final_content, "interrupted", interrupt=payload.to_dict()
                    ):
                        yield event
                    return
                if meta.pop("unknown_tool", False):
                    status = "unknown"
                    termination_reason = "tool_outcome_unknown"
                    final_content = "工具结果不确定，请核对实际结果后再继续，未自动重试。"
                    break
                pending = []
                meta["step_index"] += 1
                await self._save_checkpoint(kind="step")
            else:
                status = "incomplete"
                termination_reason = "max_steps"
                final_content = "任务未完成，已达到最大步骤限制。"
                self.session.context.add_ai_message(final_content)
                yield self._event("text_delta", data={"content": final_content})
            async for event in self._finish(
                final_content, status, termination_reason=termination_reason
            ):
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            self._block_uncertain_tools()
            await self._close_background_tasks()
            await self._close_resources()
            meta["status"] = "cancelled"
            self._set_session_run_stats()
            # Persist the last running tool marker; recovery must reconcile it, not replay it.
            await self._save_checkpoint(kind="cancelled")
            raise
        except Exception as exc:
            self._block_uncertain_tools()
            await self._close_background_tasks()
            await self._close_resources()
            meta["status"] = "error"
            self._set_session_run_stats()
            ctx = self._ctx()
            await self._in_context(ctx, self.middleware.execute_on_error, ctx, exc)
            await self._save_checkpoint(kind="error")
            yield self._event(
                "error",
                data={
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "usage": dict(meta["usage_total"]),
                    "llm_calls": meta["llm_calls"],
                },
            )
            raise

    async def _close_background_tasks(self):
        tasks = self._metadata.pop("background_tasks", [])
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _finish(self, content, status, termination_reason=None, interrupt=None):
        await self._close_background_tasks()
        await self._close_resources()
        if self._metadata.get("unknown_tool"):
            status = "unknown"
            termination_reason = "tool_outcome_unknown"
            content = "工具结果不确定，请核对实际结果后再继续，未自动重试。"
        self._metadata["status"] = status
        if status == "unknown":
            self.session.metadata["_blocked_run"] = self._metadata["run_id"]
        self._set_session_run_stats()
        if status != "interrupted":
            self.session.metadata.pop("_pending_interrupt", None)
            await self._save_checkpoint(kind="final")
        data = dict(
            content=content,
            usage=dict(self._metadata["usage_total"]),
            latency_ms=self._metadata["latency_ms"],
            llm_calls=self._metadata["llm_calls"],
            status=status,
            termination_reason=termination_reason,
            interrupt=interrupt,
            usage_available=self._metadata.get("usage_available", True),
            output=self._metadata.get("structured_output"),
        )
        yield self._event("done", data=dict(data))
        yield self._event("run_end", data=dict(data))

    async def _run_tool_phase(self, *, ctx, tool_calls):
        # Batches only cross tools with no before/after hooks. Guards that depend on a
        # previous result must run after that result, not before an entire batch.
        can_batch = (
            not self._metadata.get("batch_results")
            and self.session.parallel_tool_calls
            and not self.middleware._before_tool_mws
            and not self.middleware._after_tool_mws
            and len(tool_calls) > 1
        )
        if can_batch:
            for tc in tool_calls:
                yield self._tool_start_event(tc)
            for tc in tool_calls:
                self._metadata["tool_states"][tc.id] = "running"
            await self._save_checkpoint(kind="tool_running", pending=tool_calls)
            results = await self.tool_executor.execute(tool_calls, concurrent=True)
            self._metadata["batch_results"] = {
                tc.id: msg.model_dump(mode="json") for tc, msg in zip(tool_calls, results)
            }
            await self._save_checkpoint(kind="tool_result", pending=tool_calls)
            for index, (tc, msg) in enumerate(zip(tool_calls, results)):
                final = await self._commit_tool(ctx, tc, msg, tool_calls[index + 1 :])
                yield self._tool_end_event(tc, final)
            return
        for index, tc in enumerate(tool_calls):
            remaining = tool_calls[index:]
            cached = self._metadata.get("batch_results", {}).get(tc.id)
            if cached is not None:
                final = await self._commit_tool(
                    ctx, tc, Message.model_validate(cached), tool_calls[index + 1 :]
                )
                yield self._tool_end_event(tc, final)
                continue
            self._metadata["tool_states"][tc.id] = "preparing"
            await self._save_checkpoint(kind="tool_running", pending=remaining)
            try:
                result = await self._in_context(ctx, self.middleware.execute_before_tool, ctx, tc)
            except InterruptSignal as sig:
                remaining[0] = sig.tool_call
                tool_calls[index] = sig.tool_call
                self._record_final_call(sig.tool_call)
                sig.pending = remaining
                raise
            if result is None:
                result = ToolMessage(
                    content=json.dumps(
                        {
                            "ok": False,
                            "error": {"type": "tool_skipped", "message": "工具被中间件跳过"},
                        }
                    ),
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    status="error",
                )
            if getattr(result, "role", None) != "tool":
                tc = result
                self._record_final_call(tc)
            yield self._tool_start_event(tc)
            if getattr(result, "role", None) != "tool":
                self._metadata["tool_states"][tc.id] = "running"
                await self._save_checkpoint(kind="tool_running", pending=remaining)

                async def nested(nested_tc):
                    if self._metadata.get("unknown_tool"):
                        raise RuntimeError(
                            "An earlier tool outcome is unknown; further code tools are blocked"
                        )
                    # A code block cannot be replayed around an interrupt safely. Require
                    # a direct model tool call for approval instead of pretending resumability.
                    try:
                        nested_result = await self._in_context(
                            ctx, self.middleware.execute_before_tool, ctx, nested_tc
                        )
                    except InterruptSignal:
                        return ToolMessage(
                            content=json.dumps(
                                {
                                    "ok": False,
                                    "error": {
                                        "type": "direct_approval_required",
                                        "message": "请直接调用该工具以取得独立审批；代码块不可安全重放",
                                    },
                                }
                            ),
                            tool_call_id=nested_tc.id,
                            name=nested_tc.function.name,
                            status="error",
                        )
                    if nested_result is None:
                        return ToolMessage(
                            content='{"ok": false, "error": {"message": "工具被跳过"}}',
                            tool_call_id=nested_tc.id,
                            name=nested_tc.function.name,
                            status="error",
                        )
                    if getattr(nested_result, "role", None) != "tool":
                        nested_result = await self.tool_executor.execute_one(nested_result)
                    nested_result = self._to_core_tool_message(nested_result, nested_tc)
                    try:
                        unknown = json.loads(nested_result.content).get("status") == "unknown"
                    except (ValueError, AttributeError):
                        unknown = False
                    if unknown:
                        self._metadata["unknown_tool"] = True
                        raise RuntimeError("Nested tool outcome is unknown; reconcile before retry")
                    get_or_create_log(self.session).append(
                        run_id=self._metadata["run_id"],
                        agent=self.session.session_id,
                        step=self._metadata["step_index"],
                        kind="tool",
                        summary="nested " + nested_tc.function.name,
                        data={
                            "tool": nested_tc.function.name,
                            "arguments": nested_tc.function.arguments,
                            "nested": True,
                        },
                    )
                    return await self._in_context(
                        ctx, self.middleware.execute_after_tool, ctx, nested_result
                    )

                with bind_tool_execution(nested):
                    result = await self._in_context(ctx, self.tool_executor.execute_one, tc)
            final = await self._commit_tool(ctx, tc, result, tool_calls[index + 1 :])
            yield self._tool_end_event(tc, final)
            if self._metadata.get("unknown_tool"):
                return

    async def _commit_tool(self, ctx, tc, msg, remaining):
        final = self._to_core_tool_message(msg, tc)
        final = await self._in_context(ctx, self.middleware.execute_after_tool, ctx, final)
        final = self._to_core_tool_message(final, tc)
        self.session.context.get_messages().append(final)
        unknown = (
            final.metadata.get("status") == "unknown" or '"status": "unknown"' in final.content
        )
        self._metadata["tool_states"][tc.id] = "unknown" if unknown else final.status
        if unknown:
            self._metadata["unknown_tool"] = True
        self._metadata["approval_decisions"].pop(tc.id, None)
        self._metadata.get("batch_results", {}).pop(tc.id, None)
        get_or_create_log(self.session).append(
            run_id=self._metadata["run_id"],
            agent=self.session.session_id,
            step=self._metadata["step_index"],
            kind="tool",
            summary=f"{tc.function.name} -> {final.content[:200]}",
            data={
                "tool": tc.function.name,
                "arguments": tc.function.arguments,
                "status": self._metadata["tool_states"][tc.id],
            },
        )
        await self._save_checkpoint(kind="tool_result", pending=remaining)
        return final

    def _record_final_call(self, tc):
        for message in reversed(self.session.context.get_messages()):
            if message.role == "assistant" and message.tool_calls:
                message.tool_calls = [
                    tc if original.id == tc.id else original for original in message.tool_calls
                ]
                break

    def _block_uncertain_tools(self):
        cached = self._metadata.get("batch_results", {})
        for call_id, state in self._metadata.get("tool_states", {}).items():
            if state in ("running", "preparing") and call_id not in cached:
                self._metadata["tool_states"][call_id] = "unknown"
                self.session.metadata["_blocked_run"] = self._metadata.get("run_id")
                self._metadata["unknown_tool"] = True
                parent = self._parent
                while parent is not None:
                    parent.metadata["unknown_tool"] = True
                    parent_session = parent.config.get("session")
                    if parent_session is not None:
                        parent_session.metadata["_blocked_run"] = parent.metadata.get("run_id")
                    parent = parent.config.get("parent_context")

    @staticmethod
    def _to_core_tool_message(msg, tc):
        if isinstance(msg, ToolMessage):
            return msg
        content = getattr(msg, "content", "") or ""
        status = (
            "error"
            if ('"ok": false' in content.lower() or '"ok":false' in content.lower())
            else "success"
        )
        return ToolMessage(
            content=content,
            tool_call_id=getattr(msg, "tool_call_id", None) or tc.id,
            name=getattr(msg, "name", None) or tc.function.name,
            status=status,
            metadata=getattr(msg, "metadata", {}) or {},
        )

    def _tool_start_event(self, tc):
        tool = self.tool_executor.registry.get(tc.function.name)
        return self._event(
            "tool_start",
            data={
                "tool_name": tc.function.name,
                "tool_call_id": tc.id,
                "args": tc.function.arguments,
                "display_name": tool.display_name if tool else None,
                "side_effect": bool(tool and tool.execution.side_effect),
                "requires_approval": bool(tool and tool.execution.requires_approval),
            },
        )

    def _tool_end_event(self, tc, msg):
        return self._event(
            "tool_error" if msg.status == "error" else "tool_end",
            data={
                "tool_name": tc.function.name,
                "tool_call_id": tc.id,
                "output": msg.content,
                "status": msg.status,
            },
        )

    def _copy_messages(self):
        return [m.model_copy(deep=True) for m in self.session.context.get_messages()]

    def _set_session_run_stats(self):
        self.session.last_usage = dict(self._metadata["usage_total"])
        self.session.last_latency_ms = self._metadata["latency_ms"]
        self.session.last_llm_calls = self._metadata["llm_calls"]

    def _event(self, event_type, *, data=None, **kwargs):
        event = AgentEvent(
            type=event_type,
            session_id=self.session.session_id,
            step=self._metadata.get("step_index", 0),
            run_id=self._metadata.get("run_id"),
            data=data or {},
        )
        if self.event_bus is not None:
            self.event_bus.publish_nowait(event)
        return event
