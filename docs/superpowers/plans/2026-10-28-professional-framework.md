# Professional framework implementation

Approved scope: all findings and seven feature groups in docs/audits/2026-10-28/DESIGN_REVIEW.md. Preserve existing entry points, no real credentials/live paid calls, no automatic PyPI publication. Work on existing codex/framework-hardening-20261028 branch; BIGDATA_RESEARCH.md is unrelated.

## Direction and contracts
- Keep one AgentRunner; extend typed contracts, do not create parallel execution loops.
- Fix F1 trusted ToolOutcome vs business content; F2 completed-by-default eval truth; F3 Python/YAML effective settings equivalence first.
- Model profiles/output modes and typed content/artifacts must reach actual adapters and runner, not only data models.
- SessionStore owns durable conversation history and lifecycle. RunStore/JobBackend owns query/cancel/signal and persisted tasks, with safe single-winner claims. Long task orchestration uses an existing durable backend integration where available; never replay unknown external actions.
- Typed persistent RunState has explicit v1 compatibility and future-version rejection. Runtime deps/resources are transient.
- Evals statuses/tool outcomes/required judges/baselines and tracing consume actual committed runtime facts.

## Owners / write areas
1. model_capabilities: config/yaml_loader.py, llm/gateway.py, llm/adapters/*, core/message.py, new llm profile/output modules, new artifacts module, tests/docs for these. Coordinate runner hooks with runtime owner; root owns facade.
2. runtime_contracts: runtime/*, tools/*, agent/sub_agent.py, agent/async_sub_agent.py, agent/multi_agent.py; typed outcomes/state/context/deps, reconciliation, structured-mode integration and typed handoffs. Root owns agent/agent.py and agent/base.py/session.py.
3. quality_lifecycle: evals/*, sandbox/*, observability/*, tests/docs; trustworthy evaluations, baseline comparison, structured tracing and cancellation cleanup.
4. root: agent/agent.py/base.py/session.py, graph/checkpoint.py, new persistence/job modules, integration/export/docs/CI and final acceptance; durable sessions/tasks and shared contracts integration.

## Acceptance
- Offline regressions for three reproduced defects.
- Existing ordinary/stream/structured/plan/delegation/interrupt/unknown semantics retained.
- Real SDK MockTransport contracts for native schema and content; typed output unsupported explicit failure.
- Roundtrip artifact/session/checkpoint persistence; ownership and expiry; future snapshots rejected.
- Durable task restart/query/cancel/signal/repeated claim tests; no unknown side-effect replay.
- Evals noncompleted/default, executed vs requested, judge and empty assertions, baseline/threshold reporting.
- Sandbox cancellation actually cleans subprocess, not only timeout.
- Complete offline suite, strict docs, fatal lint, build and isolated wheel smoke. Report unavailable backend/live-service acceptance separately.

## Execution ledger
- Started 2026-10-28 on bb7b671. Architecture report and repro are current scope.
- 2026-10-29: F1/F2/F3 and all seven feature groups implemented and integrated. Model artifacts traverse actual SDK requests, Agent/session/events and typed delegation. Root completed native strict-schema and result projection after model/runtime subagent usage limits.
- Root persistence implementation adds fenced sessions, durable job truth, approval/cancel/reconciliation and optional official DBOS workflow/step integration. Independent persistence findings fixed with regressions.
- Acceptance: 500 passed/10 skipped locally, real DBOS+SQLite test, strict docs/build/Twine/isolated wheel pass. Feature-branch push triggers final Python3.11/3.12/3.13 CI; no automatic PyPI publication.
