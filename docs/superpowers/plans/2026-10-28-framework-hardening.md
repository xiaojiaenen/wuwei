# Framework hardening implementation plan

Goal: close all confirmed findings in the 2026-10-27 review and user-feedback/API review, preserving local work and existing callers through explicit compatibility paths.
Architecture: retain one Runner, share typed model/tool contracts and persist run-scoped governance. Keep protocol/provider adaptation at boundaries. No publishing, push, destructive checkout or real provider writes.
Spec: docs/audits/2026-10-27/REVIEW.md and USER_FEEDBACK_AND_API_REVIEW.md; external principles in FRAMEWORK_REFERENCES.md.

## Ownership and constraints
- Primary: Agent/Session/Runner, middleware/hitl/permissions/config, checkpoint/debug, integration and acceptance.
- Model owner: llm/**, core/types.py and message.py compatibility, provider tests; coordinate result-status additions.
- Boundary owner: serve/**, mcp/**, graph/graph.py and __init__.py, tools/**, ledger/codemode/streaming; no checkpoint edits.
- Follow-up owner: subagent/planning after Runner contract settled.
- Preserve other working changes. Work in current checkout on codex branch, because uncommitted/untracked 3.2 work is the reviewed candidate and would be absent in a new worktree. Do not commit existing mixed changes without user request.
- Tests first against each concrete regression, then scoped checks, finally integrated offline suite. Online model/service/device checks are separate acceptance.

## Tasks
- [x] 1. Model boundaries: request/client split, injection/close, finish reasons, strict tool JSON, retry/fallback/deadline, usage, legacy provider/type contracts.
- [x] 2. Tool/protocol boundaries: schema generation and validation, MCP transport/discovery, A2A versioned wire format, AG-UI sessions, graph stream reducers, ledger durable recall.
- [x] 3. Runtime: persistent run governance, multi-tool approval continuation, atomic checkpoint consumption, recoverable/unknown tool state, no approval bypass, honest termination and terminal ordering.
- [x] 4. Unified usage paths: same lifecycle for structured output/subagents/plans; fork isolation and safe replay; serialization and stable checkpoint pagination.
- [x] 5. Documentation and packaging: executable current examples, migration/compatibility paths, dependency floors, CI gates, closed-finding matrix.
- [x] 6. Integrated acceptance: existing offline tests + new failure-path regressions; independent scoped review of consequential edits.

## Review focus
- Duplicate/concurrent/restarted resume never repeats uncertain external writes.
- Approval of one call cannot approve sibling calls or skip permission checks.
- Child/plan/structured paths preserve parent governance and cumulative spend.
- Interrupted/aborted/incomplete streams cannot execute partial tool JSON or report completed.
- Existing adapter injection/import/session entry points remain usable or get explicit migration adapters.

## Progress
- 2026-10-28: preflight complete. Original working tree has extensive 3.2 changes; preserved. Native primary integration plus independent model/protocol owners under dispatching-parallel-agents skill. Original review probes already establish failing baseline.

- Model/boundary/runtime/delegation fixes implemented with failing-before/passing-after regressions. Independent runtime review four findings and model review three findings repaired; delegation review four findings repaired. Middleware integration cleanup and governed internal model calls added.
- Python3.11 matrix found typing.TypedDict/Pydantic incompatibility; corrected to typing_extensions.TypedDict rather than dropping Python3.11 support.
- Packaging and public documentation acceptance underway; fix status inventory at docs/audits/2026-10-27/FIX_STATUS.md. Push authorization received from user; push feature branch only, no PyPI publication or remote deployment.

- Final acceptance: Python3.11/3.12/3.13 each411 passed/9 skipped; officialA2A0.3.26; strict docs+wheel/sdist+isolated wheel smoke+fatal lint pass. No live model or external write acceptance claimed.
