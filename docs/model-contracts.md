# Explicit model output and durable media

Use `LLMGateway.generate(messages, output_spec=OutputSpec(schema=..., mode="auto"))` for a schema-backed call. `native` sends strict JSON Schema to official OpenAI Chat/Responses endpoints, or Anthropic `output_config.format`. `prompted` adds a JSON instruction and requires local validation. `auto` chooses native for an endpoint profile advertising support, otherwise prompted. `tool` forces the reserved `__wuwei_structured_output` function and cannot coexist with executable tools. The unified runner consumes this function as output, never as a business tool.

A custom `base_url` defaults to a conservative `ModelProfile`; explicitly pass `model_profile` to declare verified native schema, image/file input, tool-calling and streaming support. `LLMGateway.register_provider(..., model_profile=...)` associates capabilities with a registered provider. Explicit unsupported modes fail before sending a request. Endpoint capability does not guarantee that every model hosted there supports every schema: providers may reject unsupported schemas/models.

YAML model settings use the same effective gateway defaults as Python. An omitted temperature remains unset (Chat retains 0.2; Responses omits it), while explicit `temperature: null` suppresses it. `provider_options` carries endpoint-specific parameters; duplicate declarations are rejected. Cache key/options and reasoning settings are accepted explicitly. Unknown top-level YAML keys remain errors. Anthropic sampling parameters depend on the installed SDK; the default omits temperature, and an explicit unsupported value raises rather than being silently dropped.

`BaseMessage.content` remains a string. `content_parts` adds `TextPart`, `ImagePart`, `FilePart` and `CitationPart`; these serialize through Pydantic normally. Native adapter responses expose typed text/citations and retain complete SDK blocks in `message.metadata["provider_output"]`. Refusals are normalized in `metadata["refusal"]`. Streaming metadata retains provider events, deltas and final output blocks, including unfamiliar output items rather than silently dropping them. Provider payloads are observational data, not execution authority.

```python
store = ArtifactStore(".wuwei/artifacts")
ref = store.put(pdf_bytes, owner="tenant-a", media_type="application/pdf", ttl=3600)
message = HumanMessage("Summarize this", content_parts=[FilePart(artifact=ref, filename="report.pdf")])
gateway = LLMGateway({
    "provider": "responses", "api_key": api_key,
    "artifact_store": store, "artifact_owner": "tenant-a",
})
```

The local store persists bytes and owner/type/size/expiry metadata. Reads validate the supplied reference against stored metadata, ownership, expiry and size. IDs cannot contain paths. Files use private modes; symlink artifacts are refused. The store is a trusted local directory, not an untrusted shared writable filesystem. References—not base64 blobs—remain in message history/checkpoints. Gateway resolution checks the authorized owner and constructs request-local media data only; it does not mutate the persisted message. The default blob limit is 32 MiB. Unsupported provider file-reference forms fail explicitly; provider-specific media and total request limits still apply.

Offline acceptance covers real installed SDK requests through MockTransport for all three providers. It does not establish live model quality or provider account/model eligibility.

OpenAI strict schemas require an object root: nested objects are closed, all declared properties become explicit required output, and defaults are applied locally rather than sent on the wire. Nullable fields retain their null schema. Open maps (`dict[str, ...]` / `extra="allow"`) cannot be represented faithfully and are rejected before the call; select prompted mode for those outputs. Caller schemas are never mutated.

Use `ArtifactStore.delete(ref, owner=...)` to delete a known owned artifact, or `purge_expired(owner=...)` for explicit owner-scoped expiry maintenance. Reads never silently delete files or artifacts owned by other tenants.
