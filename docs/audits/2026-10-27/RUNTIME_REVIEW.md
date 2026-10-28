# Independent runtime review

Reviewed the primary runtime candidate read-only (Runner/checkpoint/stack/HITL/budget/permissions/session/debug), against original audit requirements. Four actionable defects reproduced offline in `/tmp/runtime_review_probe.py` using existing `test_runtime_hardening` helpers.

## P1: parallel result checkpoint replays already-executed sibling effects

`AgentRunner._run_tool_phase` parallel branch first awaits execution of the **whole** batch, then `_commit_tool(..., tool_calls[index+1:])` persists later siblings as pending. A process death after the first `tool_result` save leaves a claimable checkpoint whose pending calls have already executed. Reproduction: two tools `a,b`, parallel enabled, checkpointer raises a simulated process-death BaseException immediately after durably saving first tool_result, then resume that checkpoint. Observed execution sequence: `['a', 'b', 'b']`.

**Repair:** preserve all uncommitted sibling outcomes as running/unknown until durably committed, or atomically save completed batch outcomes before exposing an individually resumable checkpoint. Never resume any pending call whose persisted tool_state is running/unknown.

## P1: later middleware rewrites bypass earlier approval/permission decisions

Stack runs each before_tool once; HITL checks the current revision, but subsequent middleware can return a different ToolCall, and Runner executes it directly. Reproduction: stack `[HitlMiddleware(mode='interrupt'), Rewrite()]`, initial call `safe` requiring approval, Rewrite changes its name to `danger` (also requires approval). Approve safe and resume: observed `['danger']`, final status completed. An earlier PermissionMiddleware has the same revision problem.

**Repair:** establish the final rewritten call before authorization, or invalidate and rerun applicable guards on every identity/name/argument revision. Record the finalized call in assistant history/checkpoint and bind the approval to that exact revision.

## P1: cancellation clears the execution lane while a sync side effect continues

Runner cancellation handler sets status cancelled and saves a checkpoint, but only `_finish(status='unknown')` sets `_blocked_run`. Sync tool handlers run in `asyncio.to_thread`; cancelling their await does not stop the thread. Reproduction: side-effect tool blocks on threading.Event, cancel agent.run after handler starts, immediately run another prompt on same session, then release thread. The new run completed, metadata had no blocked marker, and old tool then performed its effect.

**Repair:** when cancellation/error happens with a running side-effect call, persist unknown state and block session re-entry until explicit reconciliation. Preserve tool identity/arguments in the cancellation checkpoint. Treat before_tool handlers that execute child tools as an execution window too.

## P1: durable session deserialization deletes interruption/unknown guards

`AgentSession.from_dict` passes saved metadata into dataclass construction; `__post_init__` invokes reset, and reset pops `_pending_interrupt` and `_blocked_run`. Reproduction: serialize session with both flags and deserialize: observed restored metadata `{}`. Existing statistics restoration does not restore these flags. This allows a restored interrupted/unknown session to accept a fresh run even though an in-memory session correctly blocks it.

**Repair:** separate fresh initialization from explicit reset; deep-copy and restore saved metadata after construction, without mutating the caller's serialized dict. Add round-trip checks for both flags and ensure fresh run is rejected after reload.

## Other observations

One-shot checkpoint claims and head supersession reject duplicate resumes; ordinary fork uses a new session/run and ReplayGuard sorts before marked tool dispatchers, so the direct normal-tool fork case is improved. No finding here claims complete multi-process session exclusion or live-provider acceptance. Primary remains responsible for repair and cross-module regression validation.

## Scoped repair re-review — four findings closed

Re-read only the four repaired paths and ran their four targeted regressions: **4 passed**. No further actionable defect found in these fixes.

- Parallel batches now durably cache every result before publishing a per-result pending cursor. Resumption detects cached results, bypasses concurrent execution, and commits them without re-invoking tools. Uncached running/unknown pending calls are rejected before checkpoint claim, so the claim is not consumed merely by a rejected unsafe recovery.
- Preprocessors now precede marked HITL/Permission guards, which precede marked executing handlers; tool-call IDs cannot change. Finalized call name/arguments are stored in interrupt payload/history. A later resume re-applies preprocessors before matching the approval, so a newly changed revision cannot reuse the previous approval.
- Session deserialization restores saved metadata after fresh initialization and preserves both blocking flags without deleting the caller's saved flags.
- Cancellation/error marks uncached preparing/running tools unknown and persists the session block before permitting lane release. Cached completed batch results are intentionally excluded from unknown marking.

This verdict supersedes the four open findings above for the reviewed candidate; it is scoped to these repairs rather than a new repository-wide review.
