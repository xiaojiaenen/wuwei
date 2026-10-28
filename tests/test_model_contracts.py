import pytest

from wuwei.config.yaml_loader import LLMConfig
from wuwei.core.message import HumanMessage
from wuwei.llm.gateway import LLMGateway


def test_yaml_unset_and_null_temperature():
    from wuwei.config.yaml_loader import effective_llm_config

    for provider in ("openai", "responses"):
        for values in ({}, {"temperature": None}, {"temperature": 0.7}):
            config = {"provider": provider, "api_key": "offline", **values}
            a = LLMGateway(config)
            b = LLMGateway(effective_llm_config(LLMConfig(**config)))
            assert a.adapter.build_request([HumanMessage("x")]) == b.adapter.build_request(
                [HumanMessage("x")]
            )


def test_native_requests():
    from wuwei.llm.output import OutputSpec

    spec = OutputSpec(
        schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        mode="native",
    )
    for provider in ("openai", "responses", "anthropic"):
        g = LLMGateway({"provider": provider, "api_key": "offline"})
        request = g.build_request([HumanMessage("x")], output_spec=spec)
        assert (
            "response_format" in request
            if provider == "openai"
            else "text" in request
            if provider == "responses"
            else "output_config" in request
        )


def test_artifact_durable_and_owner(tmp_path):
    from wuwei.artifacts import ArtifactStore

    s = ArtifactStore(tmp_path)
    ref = s.put(b"hello", owner="alice", media_type="text/plain")
    assert ArtifactStore(tmp_path).read(ref, owner="alice") == b"hello"
    with pytest.raises(PermissionError):
        s.read(ref, owner="bob")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "responses", "anthropic"])
async def test_real_sdk_native_transport(provider):
    import json

    import httpx

    from wuwei.llm.output import OutputSpec

    if provider == "anthropic":
        import httpx2 as httpx
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        if provider == "openai":
            body = {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{}", "refusal": None},
                    }
                ],
            }
        elif provider == "responses":
            body = {
                "id": "r1",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "id": "m1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": "no"}],
                    }
                ],
            }
        else:
            body = {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "{}", "citations": []}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    g = LLMGateway({"provider": provider, "api_key": "offline", "http_client": client})
    try:
        response = await g.generate(
            [HumanMessage("x")],
            output_spec=OutputSpec(
                schema={"type": "object", "properties": {}, "additionalProperties": False}
            ),
        )
        assert response.message.metadata["provider_output"]
        assert captured
        wire = captured[0]
        if provider == "openai":
            assert wire["response_format"]["json_schema"]["strict"] is True
        elif provider == "responses":
            assert wire["text"]["format"]["strict"] is True
            assert response.message.metadata["refusal"] == "no"
        else:
            assert wire["output_config"]["format"]["type"] == "json_schema"
    finally:
        await g.aclose()


def test_profile_rejects_native_custom_endpoint():
    from wuwei.llm.output import OutputSpec

    g = LLMGateway({"api_key": "offline", "base_url": "https://invalid.example/v1"})
    with pytest.raises(ValueError, match="native"):
        g.build_request([HumanMessage("x")], output_spec=OutputSpec(schema={}, mode="native"))


def test_typed_parts_roundtrip_and_adapter():
    from wuwei.core.message import BaseMessage, FilePart, ImagePart, TextPart

    m = HumanMessage(
        "", content_parts=[TextPart(text="look"), ImagePart(url="https://example.test/image.png")]
    )
    restored = BaseMessage.model_validate_json(m.model_dump_json())
    assert restored.content_parts == m.content_parts
    for provider in ("openai", "responses", "anthropic"):
        g = LLMGateway({"provider": provider, "api_key": "offline"})
        r = g.build_request([restored])
        assert "image" in str(r)
    with pytest.raises(ValueError):
        HumanMessage("", content_parts=[FilePart()]).provider_content("openai")


def test_artifact_expiry_and_forgery(tmp_path):
    from wuwei.artifacts import ArtifactStore

    s = ArtifactStore(tmp_path, max_size=5)
    with pytest.raises(ValueError):
        s.put(b"123456", owner="a", media_type="text/plain")
    ref = s.put(b"1", owner="a", media_type="text/plain")
    with pytest.raises(ValueError):
        s.read(ref.model_copy(update={"size": 2}), owner="a")
    with pytest.raises(ValueError):
        s.read({"id": "../bad", "owner": "a", "size": 0, "media_type": "text/plain"}, owner="a")


def test_artifact_inputs_resolved_only_with_owner(tmp_path):
    from wuwei.artifacts import ArtifactStore
    from wuwei.core.message import ImagePart

    store = ArtifactStore(tmp_path)
    ref = store.put(b"png", owner="a", media_type="image/png")
    message = HumanMessage("", content_parts=[ImagePart(artifact=ref)])
    for provider in ("openai", "responses", "anthropic"):
        gateway = LLMGateway(
            {
                "provider": provider,
                "api_key": "offline",
                "artifact_store": store,
                "artifact_owner": "a",
            }
        )
        request = gateway.build_request([message])
        assert "cG5n" in str(request)
        assert message.content_parts[0].artifact == ref
    gateway = LLMGateway({"api_key": "offline", "artifact_store": store, "artifact_owner": "b"})
    with pytest.raises(PermissionError):
        gateway.build_request([message])


def test_citations_and_stream_refusal_preserved():
    from types import SimpleNamespace as N

    from wuwei.llm.adapters.responses import ResponsesAdapter

    adapter = ResponsesAdapter(api_key="offline")
    output = N(
        type="message",
        content=[
            N(
                type="output_text",
                text="answer",
                annotations=[
                    {"type": "url_citation", "url": "https://example.test", "title": "source"}
                ],
            ),
            N(type="refusal", refusal="no"),
        ],
    )
    result = adapter.parse_response(
        N(output=[output], status="completed", model="gpt-4o", usage=None)
    )
    assert result.message.content_parts[1].type == "citation"
    assert result.message.metadata["refusal"] == "no"
    event = adapter.parse_stream_chunk(N(type="response.refusal.delta", delta="no"))
    assert event["provider_metadata"]["refusal_delta"] == "no"


def test_provider_options_and_strict_tool_modes():
    from wuwei.config.yaml_loader import effective_llm_config
    from wuwei.llm.output import OutputSpec

    with pytest.raises(ValueError, match="Reserved"):
        effective_llm_config(LLMConfig(provider_options={"api_key": "secret"}))
    for provider in ("openai", "responses", "anthropic"):
        g = LLMGateway({"provider": provider, "api_key": "offline"})
        req = g.build_request(
            [HumanMessage("x")],
            output_spec=OutputSpec(schema={"type": "object", "properties": {}}, mode="tool"),
        )
        function = req["tools"][0].get("function", req["tools"][0])
        assert function["strict"] is True
        assert req["tool_choice"]


@pytest.mark.asyncio
async def test_stream_keeps_refusal_events_and_usage():
    class Adapter:
        model = "test-model"

        def build_request(self, *args, **kwargs):
            return kwargs

        async def call(self, request):
            async def events():
                yield {
                    "provider_metadata": {
                        "refusal_delta": "no ",
                        "provider_event": {"type": "citation", "url": "a"},
                    }
                }
                yield {
                    "provider_metadata": {
                        "refusal_delta": "thanks",
                        "provider_event": {"type": "image", "id": "i"},
                    }
                }
                yield {
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                }

            return events()

        def parse_stream_chunk(self, chunk):
            return chunk

    gateway = LLMGateway({"api_key": "offline", "adapter": Adapter()})
    chunks = [chunk async for chunk in await gateway.generate([], stream=True)]
    assert chunks[-1].metadata["refusal"] == "no thanks"
    assert len(chunks[-1].metadata["provider_events"]) == 2
    assert chunks[-1].metadata["retry_count"] == 0
    assert chunks[-1].usage["total_tokens"] == 3


def test_artifact_expiry_and_symlink(tmp_path, monkeypatch):
    import wuwei.artifacts as artifacts

    store = artifacts.ArtifactStore(tmp_path)
    monkeypatch.setattr(artifacts.time, "time", lambda: 10)
    ref = store.put(b"x", owner="a", media_type="text/plain", ttl=1)
    monkeypatch.setattr(artifacts.time, "time", lambda: 12)
    with pytest.raises(ValueError, match="expired"):
        store.read(ref, owner="a")
    monkeypatch.setattr(artifacts.time, "time", lambda: 10)
    blob = tmp_path / (ref.id + ".blob")
    blob.unlink()
    blob.symlink_to(tmp_path / (ref.id + ".json"))
    with pytest.raises(ValueError, match="symlink"):
        store.read(ref, owner="a")


def test_native_pydantic_defaults_and_nested_schema_are_strict():
    from pydantic import BaseModel
    from wuwei.llm.output import OutputSpec

    class Item(BaseModel):
        label: str = "default"

    class Report(BaseModel):
        item: Item
        note: str | None = None

    original = Report.model_json_schema()
    spec = OutputSpec(schema=original, mode="native")
    wire = LLMGateway({"provider": "responses", "api_key": "unused"}).build_request(
        [HumanMessage("report")], output_spec=spec
    )["text"]["format"]["schema"]
    assert wire["additionalProperties"] is False
    assert set(wire["required"]) == {"item", "note"}
    assert wire["$defs"]["Item"]["additionalProperties"] is False
    assert wire["$defs"]["Item"]["required"] == ["label"]
    assert "default" not in wire["$defs"]["Item"]["properties"]["label"]
    assert original == Report.model_json_schema()  # No mutation of caller schema.


def test_native_open_dictionary_fails_before_request():
    from pydantic import BaseModel
    from wuwei.llm.output import OutputSpec

    class Report(BaseModel):
        arbitrary: dict[str, int]

    with pytest.raises(ValueError, match="additionalProperties"):
        LLMGateway({"provider": "responses", "api_key": "unused"}).build_request(
            [HumanMessage("report")], output_spec=OutputSpec(schema=Report.model_json_schema())
        )


def test_planner_native_schema_excludes_execution_state():
    from wuwei.planning.task import PlanSpec
    from wuwei.llm.output import strict_json_schema

    schema = strict_json_schema(PlanSpec.model_json_schema())
    task = schema["$defs"]["PlannedTask"]
    assert "interrupt" not in task["properties"]
    assert "usage" not in task["properties"]


def test_artifact_owner_delete_and_expired_cleanup(tmp_path):
    from wuwei.artifacts import ArtifactStore
    import time

    store = ArtifactStore(tmp_path)
    live = store.put(b"live", owner="a", media_type="text/plain")
    expired = store.put(b"expired", owner="a", media_type="text/plain", ttl=0.01)
    other = store.put(b"private", owner="b", media_type="text/plain", ttl=0.01)
    with pytest.raises(PermissionError):
        store.delete(live, owner="b")
    time.sleep(0.02)
    assert store.purge_expired(owner="a") == 1
    assert store.read(live, owner="a") == b"live"
    assert (tmp_path / f"{other.id}.blob").exists()
    store.delete(live, owner="a")
    assert not (tmp_path / f"{live.id}.blob").exists()


def test_registered_response_protocol_honors_advertised_native_profile(monkeypatch):
    from types import SimpleNamespace
    from wuwei.llm.adapters.responses import ResponsesAdapter
    from wuwei.llm.profile import ModelProfile, ModelCapabilities
    from wuwei.llm.output import OutputSpec
    from wuwei.llm.gateway import _PROVIDER_SPECS

    monkeypatch.setitem(
        _PROVIDER_SPECS,
        "custom-response-test",
        {
            "adapter": ResponsesAdapter,
            "default_model": "test",
            "env_prefix": "TEST",
            "requires_api_key": False,
            "model_profile": ModelProfile(
                name="verified-response", capabilities=ModelCapabilities(native_schema=True)
            ),
        },
    )
    gateway = LLMGateway({"provider": "custom-response-test", "client": SimpleNamespace()})
    request = gateway.build_request(
        [HumanMessage("x")],
        output_spec=OutputSpec(schema={"type": "object", "properties": {}}, mode="native"),
    )
    assert request["text"]["format"]["type"] == "json_schema"
