# Middleware resources, internal calls and event delivery

## Delivered

- Tracing context managers and spans now live in transient `ctx.config['_runtime_resources']`, never checkpoint metadata. Stored span handles are used directly rather than guessing the globally current span. `TracingMiddleware.aclose(ctx)` is idempotent, closes remaining spans in reverse creation order, and `on_error` closes them too.
- Compression summary/cooldown/savings state and memory extraction counters are JSON-safe context/run state, not mutable middleware-instance state shared across users. Removed the shared summary cache and unnecessary content hashing. Compression now retains the protected conversation head in its returned message view.
- Compression and memory extraction call `await ctx.config['model_call'](messages, tools=None)` when supplied. BudgetExhausted propagates rather than being swallowed as optional extraction failure. Direct standalone use without a callback retains the original llm.generate route.
- EventBus tracks queued delivery tasks, serializes publish_nowait deliveries in arrival order, supports drain/aclose, rejects use after closure, and avoids creating/leaking unawaited coroutines in synchronous no-loop contexts. Subscriber exceptions remain isolated.

## Primary integration contract

1. Runner supplies one transient `_runtime_resources` dictionary shared by its contexts for the run, excluded from persisted state.
2. Before terminal/interrupt/cancel, Runner calls each cleanup-capable middleware's `await aclose(ctx)`; tracing on_error also handles exception paths.
3. Runner supplies async `model_call(messages, tools=None) -> LLMResponse`. It charges actual usage to run and ancestors and checks applicable budgets before/after the helper call, without recursively invoking extraction/compression. This callback is the governance boundary for internal model work.
4. Owners of EventBus may await drain before consuming delivery-dependent state, and await aclose when disposing the bus. A bus shared by multiple Agents must not be closed by one run.

## Evidence

Three initial tests failed before implementation: nonserializable tracing resource in metadata, compression ignored governed callback/shared-state isolation, and missing EventBus drain/order lifecycle.

Final scoped command:

```
.venv/bin/python -m pytest tests/test_middleware_lifecycle.py tests/test_middleware.py tests/test_v32_kernel.py tests/test_coverage_gaps.py -q
32 passed
```

Six new tests cover fake-tracer serialization/cleanup/error idempotency, independent compression contexts, governed extraction/counters and budget propagation, ordered event delivery/drain/close, and no-loop coroutine leak handling. No network or OTel exporter service used. Ruff passes for the four owned modules and new tests.

Whole-run integration of callback accounting and cleanup is owned by primary; these focused tests establish the middleware side of that interface.
