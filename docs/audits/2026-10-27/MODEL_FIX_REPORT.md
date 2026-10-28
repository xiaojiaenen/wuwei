# Model/provider boundary fixes

## Delivered

- One canonical `core.types.LLMResponse`, re-exported by `llm.types`. Legacy `content/tool_calls/reasoning_content` keyword construction is explicitly converted to `message`; conflicting old/new arguments raise. Existing modern `message/finish_reason/usage/model/latency_ms` construction remains valid. Canonical serialization contains the message once.
- `AgentRunResult` adds incomplete/cancelled/unknown, termination_reason, and structured output.
- Finish reasons are normalized with raw values retained. Known EOS/end_turn map to stop, truncation to length, abort to cancelled, unknown to unknown. Non-success responses never expose executable tool calls.
- OpenAI generation settings are retained. Client-only options are extracted from model request options. Responses maps max_tokens to max_output_tokens. Chat supports explicit `output_token_parameter='max_completion_tokens'`; compatible endpoints retain max_tokens by default. Reasoning-content replay is capability opt-in.
- Injectable clients/client factories, adapter factories/adapters, public register_provider, explicit borrowed/owned client lifecycle, aclose and async context management. SDK clients created here default to max_retries=0. Existing public `gateway.adapter` replacement remains supported.
- Legacy zhipu/dashscope/ollama names route through compatible endpoints with DeprecationWarning and their original environment namespaces.
- Whitelisted transient retries only, Retry-After numeric seconds/HTTP-date, bounded jittered backoff, full per-gateway deadlines covering stream establishment/consumption/backoff. Fallback retains its own retry/timeout policy. Streams retry/fallback only before publishing any event. Partial or interrupted tool calls are not replayed/executed; valid complete stop+tool arguments become completed calls. Malformed/empty JSON never becomes executable `{}`.
- TypedDict adapter event/tool delta contracts; no string-typed tool delta alternative. Adapter events remain mapping-compatible with existing adapters.
- Anthropic keeps every system instruction, groups adjacent parallel tool results, accepts base_url and client settings, reads real message_start.message.usage, counts cache-read/cache-creation input tokens, preserves cache counters, and recalculates merged total=input+output. Non-stream truncation/reasoning and signed thinking blocks are retained. OpenAI usage-only events include true zero usage.
- Usage has an explicit availability flag, separate from zero-valued counters. Responses reasoning items and Anthropic signed/redacted thinking are retained in message/chunk metadata for canonical replay. Runner must preserve this metadata in its context AI message (integration owned by primary).

## Evidence

Tests were added before implementation: five initial failing regression tests reproduced ignored temperature/budget, Responses mapping, strict finish reason, Anthropic base_url, and duplicate response type.

Final focused run:

```
.venv/bin/python -m pytest tests/test_model_hardening.py tests/test_core.py tests/test_framework.py tests/test_coverage_gaps.py -q
54 passed
```

The new model tests use official SDK typed ChatCompletionChunk / RawMessageStartEvent and httpx.MockTransport, plus deterministic fake streams for deadlines, non-replay, cancellation, split usage, malformed calls, fallback policy, lifecycle and signed reasoning replay. No provider credentials or online calls were used. Ruff and compileall were also applied to owned model files.

Earlier broader run reached 81 passed with a CodeMode failure in test_v32_features::test_run_code_calls_tools_and_captures_print during concurrent implementation; later kernel tests saw approval-policy failures while runtime ownership was active. These are not treated as final acceptance for those modules; primary owns their integration checks.

## Remaining acceptance / compatibility limits

- Live provider service behavior and every legacy provider endpoint remain untested offline.
- SDK dependency declaration must stop admitting OpenAI versions without Responses. Installed/tested versions are openai 3.23.0 and anthropic 1.11.0. Responses first appeared in 1.66.0, but that historical introduction alone does not establish a tested minimum for all event features here. Prefer declaring the tested floors (openai>=3.23.0, anthropic>=1.11.0) unless a lower-version compatibility matrix is executed. Dependency changes belong to primary, not this model-only owner.
- Independent review found and corrected an explicit Responses budget override, opt-in reasoning replay for canonical BaseMessage, and fallback deadline reset. The primary deadline now bounds the entire call including fallback; the fallback retains its retry policy and may impose a shorter timeout. Added explicit regression tests for non-stream and stream deadline sharing.
- Whole-agent usage availability propagation depends on Runner aggregation; model responses/chunks expose the distinction.
