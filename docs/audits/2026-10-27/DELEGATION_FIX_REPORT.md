# Delegation and plan execution hardening

## Implemented

- Synchronous and background sub-agents now use `Agent` / `AgentRunner`, not a second LLM/tool graph. Parent middleware constraints are preserved and explicit child middleware adds constraints. Fresh stacks and child executors avoid rebinding shared middleware to another executor. Active dispatchers are omitted in child stacks to prevent immediate recursive self-dispatch.
- Both delegation middleware classes declare `is_tool_handler = True`; background dispatch returns `ToolMessage` through the common protocol rather than mutating messages and returning `None`.
- Nested child approval persists `delegations[call_id]` and raises a parent `InterruptSignal` with `continuation`. Resuming the parent reconstructs and resumes the child checkpoint; completed child tools are not replayed. Child usage is charged by Runner only, into all ancestor contexts.
- Background handles expose actual status, result, error, usage, call count, and interruption. Handles are session-scoped. `resume_async_task(task_id, approved, reason)` resumes the same handle from its child checkpoint. Background tasks register with Runner's transient task list and are cancelled/awaited when the parent run exits.
- PlanExecutor rejects duplicate IDs, unknown/duplicate edges, cycles, and nonpositive parallelism. Non-completed workers never become completed tasks; interrupted tasks preserve payload and session identity. No whole-task automatic retries follow arbitrary failures or tools. Exception costs already charged to the session are retained; exceptions with prior tool messages are conservatively `unknown`, initial model failures are `failed`.
- `PlanExecutor.resume(plan, task_id, approved=True, reason='')` resumes the saved task and schedules newly ready descendants without rerunning completed tasks. Per-task usage preserves complete plan totals across continuation. Cancellation cancels and awaits scheduled workers.
- MultiAgentGraph leader, synthesis, workers, and handoff use governed Agent runs. Team workers additionally inherit leader constraints. Invalid assignment blocks and non-completed members cannot be synthesized into apparent success. Delegation depth is bounded; graph step exhaustion raises an explicit incomplete error.

## Verification

No real service/network calls. Fake gateways and in-memory checkpoints exercised:

```sh
.venv/bin/python -m pytest tests/test_delegation_hardening.py tests/test_v32_features.py -q
# 34 passed
```

The new regression file covers inherited wrappers, DAG validation, exhausted steps, nested interrupted child resume, interrupt cost retention, async usage/session isolation, governed team calls, full plan continuation, and explicit background continuation. Nested parent/child continuation records exactly 60 tokens for four fake calls; plan continuation records 45 tokens for three calls, with the approval tool executed once.

Targeted Ruff checks for the changed delegation/planning modules and new tests pass.

## Independent review follow-up

All four concrete review paths were reproduced before correction:

1. Session ownership now uses one `_owner_session(ctx)` resolver, with no ambiguous boolean precedence. Regression contexts include the real Runner `State.metadata.session_id`; same-owner check, cancel, and resume succeed, other sessions are rejected.
2. Planner prompts require every newly proposed task to be pending. Parsed newly generated plans normalize status and clear execution-only state, including legacy `in_progress` roots; they execute instead of silently becoming incomplete. Persisted plans passed directly to PlanExecutor retain their execution states.
3. MultiAgent fan-out explicitly cancels and awaits unfinished sibling tasks after any exception or cancellation. The original exception exposes `delegation_outcomes`; cancelled siblings are `unknown`, not asserted side-effect-free. No synthesis follows a failed fan-out.
4. Child cancellation forwards a saved `_blocked_run` as `child_unknown`. Async handles expose `status=unknown` and `reconciliation={session_id, run_id}`; cancel no longer overwrites that state with cancelled. A thread-backed side-effect regression confirms the uncertainty remains visible while the thread can still finish. Parent final status propagation is owned by Runner.

Nested resume also avoids creating a fresh child session before restoring the checkpoint snapshot. The regression changes the configured child max_steps to zero between interruption and resume: the saved original session still resumes and executes the approved tool exactly once.

## Remaining boundaries

- Background handle inventory is deliberately in-process, not a durable background job service. Child approval checkpoints are durable when a durable checkpointer is supplied, but handle inventory is lost on process restart. Parent exit cancels running background work rather than allowing an ungoverned detached job.
- MultiAgentGraph retains its string-returning public API. Non-completed members raise `DelegationStoppedError`, whose `result` and `interrupt` preserve the member continuation; a team-level durable resume API is not provided.
- `retries` remains accepted for API compatibility, but no longer authorizes blind full-task replay. Safe retries belong to explicitly idempotent operations.
