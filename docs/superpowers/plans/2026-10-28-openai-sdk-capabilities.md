# OpenAI SDK capabilities

Approved scope: Responses prompt-cache configuration, detailed cache/reasoning usage, and modern model parameter compatibility. Preserve older OpenAI-compatible endpoints; no live/paid calls, automatic release, or unrelated APIs.

1. Add offline request/usage contract tests against injected clients, including streaming and aggregation.
2. Responses defaults to omitting temperature; explicit numbers still pass through. Chat retains legacy defaults; explicit None omits optional sampling parameters. Preserve max_completion_tokens configuration.
3. Expose request-level prompt_cache_options/key explicitly on Responses; retain provider argument passthrough. Cache reads/writes and reasoning are optional flat usage fields, not extra total tokens.
4. Carry detail fields through stream, runner, planning and delegated aggregation. Cost estimation remains existing input/output estimate; no invented discount or prices.
5. Document working configuration, known scope and limits. Run full offline suite, fatal lint and strict docs; commit and push without publishing a new version.
