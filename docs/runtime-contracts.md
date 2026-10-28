# Trusted tools and transient dependencies

Return ordinary text, dictionaries, or Pydantic models for business data. Fields such as `status: unknown` and `ok: false` in that data do not change execution state. Return `ToolOutcome(status="unknown", content=..., reconciliation=...)` when an external action's result is genuinely uncertain. A known failure uses `status="error"`.

`ToolMessage.metadata.tool_outcome` is the executor's control envelope. Middleware handling a tool directly must set a trusted outcome explicitly; committed messages retain execution facts for evaluation and tracing. The runtime does not parse business content for governance.

```python
from wuwei import Agent
from wuwei.runtime import RunContext

def lookup(key: str, ctx: RunContext[dict]):
    return ctx.deps["database"].get(key)

agent = Agent(llm=llm, tools=[lookup_tool], deps={"database": database})
```

Register `lookup` with `ToolRegistry.register_callable` to generate its schema. `ctx` is omitted from the model schema and cannot be supplied by model arguments. Dependencies are task-local runtime resources, not checkpoint data; callers must provide them again after restart. Child agents receive no dependencies unless `SubAgent(inherit_deps=True)` is explicitly configured. Parent budget and permission governance remains inherited independently.

`SubAgent` also accepts `input_model`, `output_model`, `context_keys`, and `artifact_keys`. Typed input is validated from the task JSON; typed output uses the existing governed structured-output runner. Only the named metadata keys are selected into `DelegationInput.context`. `artifact_keys` selects only explicitly named parent metadata references, validates them against the child gateway's ArtifactStore/owner, and supplies typed image/file parts. Unselected files and metadata do not cross the boundary. `DelegationOutput` carries validated typed output (when output_model is configured), child status, usage, interruption, reconciliation and artifact references without flattening control state into free text.

# Reconcile uncertain actions without replay

```python
from wuwei.tools import ToolOutcome
result = await agent.reconcile(
    checkpoint_id, tool_call_id="write1",
    outcome=ToolOutcome(content="receipt confirmed"),
    operator="operations-user", evidence="remote receipt 123",
)
```

The checkpoint must be the current unconsumed head and identify an uncertain saved call. A known outcome, operator and nonempty evidence are required. The checkpointer atomically consumes the old head and saves the result and audit before continuation. A successful action is never invoked again. With several uncertain actions, each reconciliation returns the replacement checkpoint ID until all actions have known results. Reconcile the remaining actions against that new head.

`RunState` snapshots use version 2. Version 1 is migrated on load, extension fields are retained, and unsupported versions fail explicitly. Neither this API nor local checkpoint storage promises external exactly-once execution: authoritative reconciliation evidence remains a business responsibility.
