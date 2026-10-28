# Independent delegation/planning review

Read-only review of sub_agent/async_sub_agent/multi_agent and planning modules. Reused existing delegation test evidence for normal nested approval/resume, parent guard inheritance and successful PlanExecutor.resume cumulative usage (45 tokens / 3 calls, no duplicate charge). Four concrete gaps were independently reproduced with minimal offline probes; no broad suite was repeated.

## P1: real-run async task operations reject their own session

`AsyncSubAgentMiddleware._handle_async_tool` ownership branch combines
`owner_session != ctx.metadata.get('session_id') or ctx.state.metadata.get('session_id')`.
The last operand is a truthy session ID in a real Runner context, rather than the fallback session ID inside the ownership comparison. Consequently every check/cancel/resume with ordinary populated State metadata is rejected, including the owner. Existing tests use `State()` with no session metadata, masking the error.

Probe `/tmp/delegation_review_probe.py`: completed handle owned by `owner`, both ctx.metadata and ctx.state.metadata contain session_id=owner; check returns `{"ok": false, "error": "Task not found in this session"}`.

**Repair:** compute one owner ID from metadata-or-state, compare handle.owner_session against it, and add same-owner and foreign-owner tests with realistic state metadata.

## P1: planner's normal in_progress root is never executed

`Planner._build_plan_prompt` explicitly requires the initial task to have status `in_progress`. `PlanExecutor.execute` queues indegree-zero nodes but skips every status other than pending; its finally block converts untouched in_progress tasks to cancelled. No LLM call occurs, and descendants cannot become ready.

Probe: execute one root Task(status='in_progress') using FakeLLM([completed response]); observed plan status incomplete, task cancelled, **0 model calls**.

**Repair:** normalize newly generated plans to pending before first execution, while rejecting or explicitly recovering truly active persisted tasks. Align planner prompt/validated model and execution entry point; do not indiscriminately replay persisted in_progress nodes.

## P1: MultiAgentGraph failure leaves sibling workers running

`fan_out_node` uses `await asyncio.gather(*tasks)` without cancelling/draining sibling tasks on error. A member failure raises graph.run immediately while other members continue independently, including their tool effects. Runner's background_tasks registry does not own these gather-created member tasks.

Probe: leader assigns failing member and waiting member; failing member waits for sibling to start then raises. After graph.run raises, releasing waiting member still produces `['slow continued']`.

**Repair:** structured cancellation and draining of the whole fan-out on failure/cancellation; preserve interrupted/unknown results and reconcile uncertain side effects rather than abandoning workers. A TaskGroup or explicit task lifecycle is suitable.

## P1: async child cancellation hides uncertain side effects from parent

Parent Runner closes background tasks on terminal completion. Async child cancellation is swallowed into handle.status='cancelled', even if child Runner has just marked its running sync side effect unknown. The child's blocked session is ephemeral and its uncertain outcome never blocks the parent session. The parent returns completed and can immediately start another run while the cancelled thread later performs its effect.

Probe `/tmp/async_cancel_review.py`: background child enters a synchronous side_effect=True tool blocked on threading.Event; parent completes and cancels child; parent accepts a second run; release thread. Observed `root completed`, `next completed`, handle `cancelled`, effect count 1, parent metadata has no blocked marker.

**Repair:** propagate cancelled-child uncertainty to the root before emitting its terminal result. Preserve pending tool identity/outcome in the handle/durable continuation, mark root unknown/block it when a child side effect cannot be stopped, and distinguish successful cancellation from an ongoing unknown effect.

## Existing improvements retained

SubAgent runs through Runner and inherited middleware; nested interrupt continuation carries child payload rather than re-running child from scratch. PlanExecutor validates DAG identifiers/edges/cycles and no longer retries an entire failed task automatically. Normal successful resume uses each task's cumulative usage once when rebuilding aggregate usage. These improvements do not remove the four blockers above.
