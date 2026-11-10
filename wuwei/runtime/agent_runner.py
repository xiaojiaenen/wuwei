"""One event-driven execution engine for ordinary, streamed and delegated runs.

Run governance is JSON state; middleware instances hold configuration only.
Every tool result is committed before advancing. Uncertain side effects are not
replayed, and an approval consumes one exact call rather than skipping guards.
"""

import asyncio
import json
import logging
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
from wuwei.runtime.stream_bridge import bridge
from wuwei.tools.executor import bind_tool_execution

logger = logging.getLogger(__name__)


def _empty_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(total, usage):
    for key, value in (usage or {}).items():
        total[key] = total.get(key, 0) + (value or 0)


STRUCTURED_SCHEMA_MARKER = "只输出符合以下JSON Schema的JSON对象："


def _attach_structured_schema(messages, schema) -> None:
    """把结构化输出约束并入首条 system 消息。

    不能 ``insert(0, SystemMessage(...))``：会话第 0 条已经是 system prompt，
    再插一条就形成相邻两条 system，OpenAI 兼容端点（DeepSeek / vLLM / 各类
    网关）会直接 400 ``System message must be at the beginning``。

    合并而非新增还有两个好处：system 条数与顺序不变，prompt caching 的前缀
    保持稳定；注入的说明不会随轮次累加。重复调用是幂等的。
    """
    from wuwei.core.message import SystemMessage

    text = STRUCTURED_SCHEMA_MARKER + json.dumps(schema, ensure_ascii=False)
    if not messages:
        messages.append(SystemMessage(content=text))
        return
    head = messages[0]
    if head.role != "system":
        # 没有可合并的 system（调用方自建上下文）时才新增，且必须放在首位。
        messages.insert(0, SystemMessage(content=text))
        return
    if STRUCTURED_SCHEMA_MARKER in (head.content or ""):
        return
    head.content = f"{head.content}\n\n{text}" if head.content else text
    # 首条 system 若带多模态块，字符串追加到 content 后 provider 会把它当文本块发，
    # 这里无需重排 content_parts。


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
        *,
        consumer_idle_timeout: float | None = 30.0,
        event_queue_size: int = 64,
    ):
        self.llm = llm
        self.tools = tools  # compatible constructor; effective tools come from registry per call
        self.tool_executor = tool_executor
        self.session = session
        self.middleware = middleware if middleware is not None else MiddlewareStack()
        self.checkpointer = checkpointer
        self.event_bus = event_bus
        # A run owns the session lock; a consumer that stops reading must not keep
        # it alive, so events cross a bounded queue (see runtime/stream_bridge.py).
        self.consumer_idle_timeout = consumer_idle_timeout
        self.event_queue_size = event_queue_size
        self._metadata = {}
        self._parent = None
        self.response_model = None
        self.output_spec = None
        self.deps = None
        self.run_context = None
        self.max_output_retries = 2
        self._resources = {}
        self._last_ctx = None
        self._warned_serial_tools = False

    async def run(self, user_input, stream=False):
        if stream:
            return self._iter_events(user_input, stream=True)
        return await self._collect(self._iter_events(user_input, stream=False))

    def stream_events(self, user_input):
        return self._iter_events(user_input, stream=True)

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
        if loaded.metadata.get("schema_version", 1) not in (1, 2):
            raise ValueError("Unsupported checkpoint schema version")
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
        from wuwei.runtime.contracts import RunState

        saved_run = RunState.model_validate(loaded.metadata.get("run_state") or {}).model_dump()
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

    async def reconcile(self, checkpoint_id, *, tool_call_id, outcome, operator, evidence):
        """Record an authoritative external result without replaying the action.

        The atomic replacement is durable before continuation is allowed. Evidence
        is required even for an externally confirmed failure.
        """
        from wuwei.tools.outcome import ToolOutcome

        if not isinstance(outcome, ToolOutcome) or outcome.status == "unknown":
            raise ValueError("Reconciliation requires a known ToolOutcome")
        if not operator or not str(evidence).strip():
            raise ValueError("Reconciliation requires operator and evidence")
        if self.checkpointer is None:
            raise ValueError("Reconciliation requires checkpointer")
        async with self.session.lock:
            saved = await self.checkpointer.load(checkpoint_id)
            if saved.metadata.get("session_id") != self.session.session_id:
                raise ValueError("Reconciliation session does not match checkpoint")
            if saved.metadata.get("schema_version", 1) not in (1, 2):
                raise ValueError("Unsupported checkpoint schema version")
            from wuwei.runtime.contracts import RunState

            state = RunState.model_validate(saved.metadata.get("run_state") or {}).model_dump()
            if state.get("tool_states", {}).get(tool_call_id) not in (
                "unknown",
                "running",
                "preparing",
            ):
                raise ValueError("Tool is not awaiting reconciliation")
            calls = state.get("unresolved_calls", [])
            call = next(
                (ToolCall.model_validate(c) for c in calls if c["id"] == tool_call_id), None
            )
            if call is None:
                raise ValueError("Missing saved call evidence; cannot resume")
            message = outcome.to_message(call.id, call.function.name)
            saved.messages = [
                m
                for m in saved.messages
                if not (m.role == "tool" and getattr(m, "tool_call_id", None) == call.id)
            ]
            saved.messages.append(message)
            state["tool_states"][call.id] = message.status
            state.pop("unknown_tool", None)
            state.setdefault("reconciliations", []).append(
                {
                    "call_id": call.id,
                    "operator": operator,
                    "evidence": evidence,
                    "outcome": outcome.model_dump(mode="json"),
                    "time": time.time(),
                }
            )
            remaining = [c for c in calls if c["id"] != call.id]
            still_unknown = any(
                v in ("unknown", "running", "preparing") for v in state["tool_states"].values()
            )
            state["unresolved_calls"] = remaining
            state["status"] = "unknown" if still_unknown else "running"
            saved.metadata.update(
                kind="final" if still_unknown else "step",
                run_state=state,
                pending_tool_calls=remaining,
            )
            if not still_unknown:
                saved.metadata.get("session", {}).get("metadata", {}).pop("_blocked_run", None)
            replacement = await self.checkpointer.replace_current(
                checkpoint_id, saved, metadata=saved.metadata
            )
            if still_unknown:
                self.session.context.replace_messages(
                    [m.model_copy(deep=True) for m in saved.messages]
                )
                return {
                    "status": "unknown",
                    "checkpoint_id": replacement,
                    "reconciled_call_id": call.id,
                }
            self.session.metadata.pop("_blocked_run", None)
        payload = InterruptPayload(
            checkpoint_id=replacement,
            session_id=self.session.session_id,
            tool_name=call.function.name,
            tool_args=call.function.arguments,
            run_id=state["run_id"],
            tool_call_id=call.id,
        )
        return await self.resume(payload)

    async def _iter_events(self, user_input, *, stream, resume=None):
        source = self._locked_events(user_input, stream=stream, resume=resume)
        if self.consumer_idle_timeout is None:
            async for event in source:
                yield event
            return
        async for event in bridge(
            source,
            queue_size=self.event_queue_size,
            idle_timeout=self.consumer_idle_timeout,
            name=f"wuwei-run-{self.session.session_id}",
        ):
            yield event

    async def _locked_events(self, user_input, *, stream, resume=None):
        """Hold the session lock for the duration of the run, not of the caller.

        The inner generator is closed explicitly: when this generator is closed
        at its yield point, the ``async for`` below does not close it, and its
        cancellation path would otherwise run after the lock was already freed.
        """
        async with self.session.lock:
            if resume is None and self.session.metadata.get("_blocked_run"):
                raise ValueError(
                    "Session has an unknown tool outcome; reconcile externally before resetting the session"
                )
            if resume is None and self.session.metadata.get("_pending_interrupt"):
                raise ValueError(
                    "Session has an interrupted run; resume it before starting another run"
                )
            events = self._execute_events(user_input, stream=stream, resume=resume)
            try:
                async for event in events:
                    yield event
            finally:
                await events.aclose()

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
                "run_context": self.run_context,
                "run_state": self._metadata,
                "model": getattr(self.llm, "model", ""),
                "provider": getattr(self.llm, "provider", ""),
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

    @staticmethod
    def _persistent_state(meta):
        from wuwei.runtime.contracts import RunState

        return RunState.model_validate(
            {k: v for k, v in meta.items() if k != "background_tasks"}
        ).model_dump(mode="json")

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
            "run_state": self._persistent_state(meta),
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

    async def _execute_events(self, user_input, *, stream, resume=None):
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
            from wuwei.runtime.contracts import RunState

            self._metadata.update(
                RunState.model_validate(saved.metadata.get("run_state") or {}).model_dump()
            )
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
        from wuwei.runtime.contracts import RunContext

        inherited_deps = self.run_context.deps if self.run_context is not None else self.deps
        self.run_context = RunContext(
            inherited_deps,
            self._metadata["run_id"],
            self.session.session_id,
            self._parent.metadata.get("run_id") if self._parent else None,
        )
        meta = self._metadata
        if self._parent is not None:
            meta["parent_run_id"] = self._parent.metadata.get("run_id")
        if self.output_spec is None and meta.get("output_spec"):
            from wuwei.llm.output import OutputSpec

            self.output_spec = OutputSpec.model_validate(meta["output_spec"])
        if self.output_spec is not None:
            meta["output_spec"] = self.output_spec.model_dump(mode="json", by_alias=True)
            meta["structured_schema"] = self.output_spec.schema_
            meta["output_max_retries"] = self.max_output_retries
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
            for middleware in self.middleware.middlewares:
                hook = getattr(middleware, "before_run", None)
                if hook:
                    await hook(self._ctx())
            while meta["step_index"] < self.session.max_steps:
                ctx = self._ctx()
                if not pending:
                    ctx = await self._in_context(ctx, self.middleware.execute_before_llm, ctx)
                    if meta.get("structured_schema") and self.output_spec is None:
                        _attach_structured_schema(ctx.state.messages, meta["structured_schema"])
                    effective_tools = self.tool_executor.registry.list_tools()
                    yield self._event(
                        "llm_start", data={"tools": [t.name for t in effective_tools]}
                    )
                    start = time.monotonic()
                    call_usage = _empty_usage()
                    content = []
                    content_parts = []
                    reasoning = []
                    provider_meta = {}
                    finish = None
                    model = ""
                    usage_available = False

                    async def invoke(messages, tools):
                        kwargs = (
                            {"output_spec": self.output_spec}
                            if self.output_spec is not None
                            else {}
                        )
                        return await self.llm.generate(
                            messages, tools=tools, stream=stream, **kwargs
                        )

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
                        content_parts.extend(getattr(result.message, "content_parts", []) or [])
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
                                if getattr(chunk, "content_parts", None):
                                    content_parts = list(chunk.content_parts)
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
                    if self.output_spec is not None:
                        from wuwei.llm.output import STRUCTURED_TOOL_NAME

                        structured_calls = [
                            tc for tc in pending if tc.function.name == STRUCTURED_TOOL_NAME
                        ]
                        if structured_calls:
                            if len(structured_calls) != 1 or len(pending) != 1:
                                raise ValueError("Structured output must be the only tool call")
                            content = [
                                json.dumps(
                                    structured_calls[0].function.arguments, ensure_ascii=False
                                )
                            ]
                            pending = []
                            finish = "stop"
                    latency = int((time.monotonic() - start) * 1000)
                    self._charge_usage(call_usage, latency, model, usage_available)
                    ai_msg = AIMessage(
                        content="".join(content),
                        reasoning_content="".join(reasoning) or None,
                        tool_calls=pending,
                        metadata={
                            **provider_meta,
                            "usage_available": usage_available,
                            "model": model,
                            "provider": getattr(self.llm, "provider", ""),
                        },
                        usage=call_usage,
                        content_parts=content_parts,
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
                    if provider_meta.get("refusal"):
                        pending = []
                        status = "incomplete"
                        termination_reason = "refusal"
                        meta["step_index"] += 1
                        break
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
                final_content = (
                    f"任务未完成，已达到最大步骤限制（{self.session.max_steps} 步）。"
                    "如需继续，请提高 max_steps 或把任务拆小。"
                )
                self.session.context.add_ai_message(final_content)
                yield self._event("text_delta", data={"content": final_content})
            async for event in self._finish(
                final_content, status, termination_reason=termination_reason
            ):
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            self._block_uncertain_tools()
            await self._close_background_tasks()
            meta["status"] = "cancelled"
            for middleware in reversed(self.middleware.middlewares):
                hook = getattr(middleware, "after_run", None)
                if hook:
                    await hook(
                        self._ctx(), {"status": "cancelled", "termination_reason": "cancelled"}
                    )
            await self._close_resources()
            self._set_session_run_stats()
            # Persist the last running tool marker; recovery must reconcile it, not replay it.
            await self._save_checkpoint(kind="cancelled")
            raise
        except Exception as exc:
            self._block_uncertain_tools()
            await self._close_background_tasks()
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
            # A client that waits for a terminal event must not be left hanging just
            # because the run failed; the exception still reaches programmatic callers.
            data = self._result_payload(
                str(exc), "error", termination_reason=type(exc).__name__
            )
            for middleware in reversed(self.middleware.middlewares):
                hook = getattr(middleware, "after_run", None)
                if hook:
                    await hook(self._ctx(), data)
            await self._close_resources()
            yield self._event("done", data=dict(data))
            yield self._event("run_end", data=dict(data))
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
        if status != "interrupted":
            self.session.metadata.pop("_pending_interrupt", None)
        data = self._result_payload(content, status, termination_reason, interrupt)
        if self._metadata["status"] != "interrupted":
            await self._save_checkpoint(kind="final")
        for middleware in reversed(self.middleware.middlewares):
            hook = getattr(middleware, "after_run", None)
            if hook:
                await hook(self._ctx(), data)
        await self._close_resources()
        yield self._event("done", data=dict(data))
        yield self._event("run_end", data=dict(data))

    def _result_payload(self, content, status, termination_reason=None, interrupt=None):
        if self._metadata.get("unknown_tool"):
            status = "unknown"
            termination_reason = "tool_outcome_unknown"
            content = "工具结果不确定，请核对实际结果后再继续，未自动重试。"
        self._metadata["status"] = status
        if status == "unknown":
            self.session.metadata["_blocked_run"] = self._metadata["run_id"]
        self._set_session_run_stats()
        last_model = next(
            (m for m in reversed(self.session.context.get_messages()) if m.role == "assistant"),
            None,
        )
        final_parts = getattr(last_model, "content_parts", []) if last_model is not None else []
        final_metadata = dict(getattr(last_model, "metadata", {}) or {})
        return dict(
            content=content,
            content_parts=[part.model_dump(mode="json") for part in final_parts],
            metadata=final_metadata,
            usage=dict(self._metadata["usage_total"]),
            latency_ms=self._metadata["latency_ms"],
            llm_calls=self._metadata["llm_calls"],
            status=status,
            termination_reason=termination_reason,
            steps_used=self._metadata.get("step_index", 0),
            max_steps=self.session.max_steps,
            interrupt=interrupt,
            usage_available=self._metadata.get("usage_available", True),
            output=self._metadata.get("structured_output"),
        )

    async def _run_tool_phase(self, *, ctx, tool_calls):
        self._metadata["unresolved_calls"] = [tc.model_dump(mode="json") for tc in tool_calls]
        # Batches only cross tools with no before/after hooks. Guards that depend on a
        # previous result must run after that result, not before an entire batch.
        can_batch = (
            not self._metadata.get("batch_results")
            and self.session.parallel_tool_calls
            and not self.middleware._before_tool_mws
            and not self.middleware._after_tool_mws
            and len(tool_calls) > 1
        )
        if (
            not can_batch
            and self.session.parallel_tool_calls
            and len(tool_calls) > 1
            and (self.middleware._before_tool_mws or self.middleware._after_tool_mws)
            and not self._warned_serial_tools
        ):
            # Approval and sub-agent hooks decide per call and must see the previous
            # result, so they cannot run as a batch. Say so once instead of silently
            # ignoring the caller's parallel_tool_calls setting.
            self._warned_serial_tools = True
            logger.warning(
                "parallel_tool_calls ignored: %d before_tool / %d after_tool middleware(s) "
                "run per call, so tool calls in this session execute serially",
                len(self.middleware._before_tool_mws),
                len(self.middleware._after_tool_mws),
            )
        if can_batch:
            for tc in tool_calls:
                yield self._tool_start_event(tc)
            for tc in tool_calls:
                self._metadata["tool_states"][tc.id] = "running"
            self._note_dispatched(tc.id for tc in tool_calls)
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
            if self._guard_dispatches_tools():
                # A tool-handler middleware runs the tool inside before_tool, so the
                # call is in flight from here on; without one it is not dispatched yet.
                self._note_dispatched([tc.id])
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
                self._note_dispatched([tc.id])
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
                    unknown = (
                        nested_result.metadata.get("tool_outcome", {}).get("status") == "unknown"
                    )
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
        final.metadata.setdefault("tool_outcome", {"status": final.status, "executed": False})
        self.session.context.get_messages().append(final)
        unknown = final.metadata.get("tool_outcome", {}).get("status") == "unknown"
        self._metadata["tool_states"][tc.id] = "unknown" if unknown else final.status
        if unknown:
            self._metadata["unknown_tool"] = True
        else:
            self._metadata["unresolved_calls"] = [
                c for c in self._metadata.get("unresolved_calls", []) if c["id"] != tc.id
            ]
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

    def _guard_dispatches_tools(self) -> bool:
        return any(
            getattr(mw, "is_tool_handler", False)
            for mw in getattr(self.middleware, "_before_tool_mws", ())
        )

    def _note_dispatched(self, call_ids):
        dispatched = self._metadata.setdefault("dispatched_calls", [])
        for call_id in call_ids:
            if call_id not in dispatched:
                dispatched.append(call_id)

    def _block_uncertain_tools(self):
        """Close out every pending call so the history stays valid.

        A call that was dispatched may have run: mark it unknown and require
        reconciliation. A call that never reached dispatch cannot have run, so it
        is recorded as not executed — marking it unknown would block the session
        and demand evidence for a tool that was never called.
        """
        cached = self._metadata.get("batch_results", {})
        dispatched = set(self._metadata.get("dispatched_calls", []))
        unresolved = list(self._metadata.get("unresolved_calls", []))
        blocked = False
        for saved in unresolved:
            call_id = saved["id"]
            if call_id in cached:
                continue
            state = self._metadata.get("tool_states", {}).get(call_id)
            if state == "unknown":
                blocked = True
                continue
            if state in ("running", "preparing") and call_id in dispatched:
                self._metadata["tool_states"][call_id] = "unknown"
                self._metadata["unknown_tool"] = True
                blocked = True
                continue
            self._record_not_executed(saved)
        if blocked:
            self.session.metadata["_blocked_run"] = self._metadata.get("run_id")
            parent = self._parent
            while parent is not None:
                parent.metadata["unknown_tool"] = True
                parent_session = parent.config.get("session")
                if parent_session is not None:
                    parent_session.metadata["_blocked_run"] = parent.metadata.get("run_id")
                parent = parent.config.get("parent_context")

    def _record_not_executed(self, saved_call):
        call_id = saved_call["id"]
        name = saved_call.get("function", {}).get("name", "")
        message = ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": "not_executed",
                        "message": "运行在工具开始执行前中断，该工具未执行，可安全重试",
                    },
                },
                ensure_ascii=False,
            ),
            tool_call_id=call_id,
            name=name,
            status="error",
            metadata={"tool_outcome": {"status": "error", "executed": False}},
        )
        self.session.context.get_messages().append(message)
        self._metadata["tool_states"][call_id] = "error"
        self._metadata["unresolved_calls"] = [
            call for call in self._metadata.get("unresolved_calls", []) if call["id"] != call_id
        ]
        self._metadata["approval_decisions"].pop(call_id, None)
        self._metadata.get("batch_results", {}).pop(call_id, None)

    @staticmethod
    def _to_core_tool_message(msg, tc):
        if isinstance(msg, ToolMessage):
            return msg
        content = getattr(msg, "content", "") or ""
        outcome = (getattr(msg, "metadata", {}) or {}).get("tool_outcome", {})
        status = "error" if outcome.get("status") in ("error", "unknown") else "success"
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
