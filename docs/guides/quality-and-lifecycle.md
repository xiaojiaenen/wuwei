# Reliable evaluation and process lifecycle

An evaluation passes only when the run reaches `completed` and every required
assertion passes. Set `expected_status` to test interruption or unknown outcomes.
Cases without assertions fail unless `allow_empty_assertions: true` explicitly
selects a status-only smoke test.

```yaml
name: conversation
input: Remember the number 42
turns:
  - What number did I give you?
response_contains: ["42"]
max_tokens: 500
max_latency_ms: 10000
```

Turns share one fresh session; content assertions inspect the final reply and
usage/latency limits cover the whole case. Noncompleted turns stop the conversation.
`max_cost` compares numeric `usage.cost` or `usage.estimated_cost`; unavailable cost
or token usage fails its configured limit rather than being treated as zero.

Tool assertions use `tool_trajectory: succeeded` by default. `requested` checks
assistant intentions; `executed` checks committed tool result evidence; `succeeded`
requires a successful committed result. The trusted `tool_outcome.executed` flag
distinguishes execution errors from denials before execution. Business output is
never parsed as execution evidence. Reports preserve all three trajectories.

A configured rubric requires a judge by default. Set `judge_required: false` to
permit an explicitly skipped evaluation when the judge is absent. Skips do not
contribute to the pass count or rate; JUnit records `<skipped>`, not a passing test.

Compare upgrades with a saved JSON report:

```sh
python -m wuwei.evals run cases.yaml --agent myapp:factory --json current.json --baseline previous.json
```

A formerly passing case that fails, becomes skipped, or disappears fails baseline
comparison regardless of the aggregate pass rate. Added/removed cases and per-case
latency/token changes and tool-trajectory changes are displayed in the comparison. Unknown usage remains unknown; duplicate case names are rejected rather than silently merged. Each factory-created agent is closed after its case; cleanup failures are visible as lifecycle assertions.

## Tracing without collecting conversation content

`TracingMiddleware` emits a root `agent_run` span and structured `wuwei.trace.v1`
records with run/session/parent identity, provider/model, status and usage facts.
LLM records preserve reported cache usage and retry counts when provided; missing
facts remain absent. A bounded `records` deque supports local diagnostics. Errors
record exception type, not potentially secret exception text.

Content capture is off by default. Opt in with
`TracingMiddleware(capture_content=True, content_limit=512)`. Captured terminal
content is truncated and common token/password/secret assignments and bearer
credentials are redacted. Redaction is a useful filter, not a substitute for an
application-specific sensitive-data policy. Tool arguments and full prompts are
not collected.

## Subprocess cancellation

Local Python, shell, and Docker CLI communication kill and reap the owned local
process group on timeout or cancellation on POSIX, then drain pipe readers and
reap the directly owned process before returning or propagating cancellation.
An orphaned descendant may briefly retain a zombie entry until the OS init process
reaps it; that entry cannot execute and `kill(pid, 0)` alone does not prove liveness. Other platforms terminate the direct owned
process. This is process cleanup, not a security isolation boundary.

Each Docker run owns a private cidfile. On timeout or cancellation, cleanup uses
`docker rm -f` only for a full container ID reported in that run's cidfile. Missing
identity, invalid identity, daemon failure, or failed removal remain explicitly
unconfirmed; cancellation propagates with a cleanup note. Real-daemon acceptance
remains not run. Offline tests exercise command construction and cleanup failures
with fake transports and owned synthetic subprocesses.
