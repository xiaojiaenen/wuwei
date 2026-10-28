from types import SimpleNamespace as Ns

import pytest

from wuwei.llm.adapters.anthropic import AnthropicAdapter
from wuwei.llm.adapters.openai import OpenAIAdapter
from wuwei.llm.gateway import LLMGateway
from wuwei.llm.types import LLMResponse, Message


def test_request_config_and_client_split():
    a = OpenAIAdapter("x", temperature=0.73, max_tokens=17, max_retries=0)
    r = a.build_request([])
    assert r["temperature"] == 0.73 and r["max_tokens"] == 17
    assert "max_retries" not in r and a.client.max_retries == 0


def test_responses_budget():
    r = LLMGateway(
        {"provider": "responses", "api_key": "x", "max_tokens": 17}
    ).adapter.build_request([])
    assert r["max_output_tokens"] == 17 and "max_tokens" not in r


def test_unknown_reason_and_invalid_json():
    a = OpenAIAdapter("x")
    raw = Ns(
        choices=[Ns(finish_reason="eos", message=Ns(content="ok", tool_calls=[]))],
        usage=None,
        model="x",
    )
    r = a.parse_response(raw)
    assert r.finish_reason == "stop" and r.raw_finish_reason == "eos" and not r.usage_available
    raw.choices[0].message.tool_calls = [Ns(id="x", function=Ns(name="f", arguments="{bad"))]
    with pytest.raises(ValueError):
        a.parse_response(raw)


def test_system_and_usage():
    a = AnthropicAdapter("x", base_url="https://example.test")
    assert "one" in str(
        a.build_request(
            [Message(role="system", content="one"), Message(role="system", content="two")]
        )["system"]
    )
    e = a.parse_stream_chunk(
        Ns(
            type="message_start",
            message=Ns(usage=Ns(input_tokens=100, output_tokens=0, cache_read_input_tokens=5)),
        )
    )
    assert e["usage"]["prompt_tokens"] == 105


def test_core_response_compatibility():
    from wuwei.core.types import LLMResponse as Legacy

    assert Legacy is LLMResponse
    assert Legacy(content="hello").message.content == "hello"


class Adapter:
    def __init__(self, events=None, error=None):
        self.calls = 0
        self.events = events or []
        self.error = error

    def build_request(self, *a, **kw):
        return kw

    async def call(self, r):
        self.calls += 1
        if self.error:
            raise self.error

        async def stream():
            for event in self.events:
                yield event

        return stream()

    def parse_stream_chunk(self, e):
        return e


@pytest.mark.asyncio
async def test_complete_tools_stop_and_aborted():
    g = LLMGateway({"api_key": "x"})
    g.adapter = Adapter(
        [
            {"tool_calls_delta": [{"index": 0, "id": "c", "name": "f", "arguments": '{"x":1}'}]},
            {"finish_reason": "stop"},
        ]
    )
    chunks = [c async for c in await g.generate([], stream=True)]
    assert chunks[-1].tool_calls_complete[0].function.arguments == {"x": 1}
    g.adapter.events[-1] = {"finish_reason": "abort"}
    chunks = [c async for c in await g.generate([], stream=True)]
    assert chunks[-1].finish_reason == "cancelled" and chunks[-1].tool_calls_complete is None
    g.adapter.events[-1] = {"finish_reason": "stop"}
    g.adapter.events[0]["tool_calls_delta"][0]["arguments"] = "{bad"
    with pytest.raises(ValueError):
        _ = [c async for c in await g.generate([], stream=True)]
    assert g.adapter.calls == 3


@pytest.mark.asyncio
async def test_no_retry_local_error_or_observed_disconnect():
    g = LLMGateway({"api_key": "x", "retry": {"max_attempts": 3, "base_delay": 0}})
    a = Adapter(error=TypeError("bad argument"))
    g.adapter = a
    with pytest.raises(TypeError):
        _ = [c async for c in await g.generate([], stream=True)]
    assert a.calls == 1

    class Disconnect(Adapter):
        async def call(self, r):
            self.calls += 1

            async def stream():
                yield {"tool_calls_delta": [{"index": 0, "id": "c"}]}
                raise TimeoutError()

            return stream()

    a = Disconnect()
    g.adapter = a
    with pytest.raises(TimeoutError):
        _ = [c async for c in await g.generate([], stream=True)]
    assert a.calls == 1


@pytest.mark.asyncio
async def test_deadline_and_fallback_policy():
    import asyncio

    class Hanging(Adapter):
        async def call(self, r):
            await asyncio.sleep(1)

    g = LLMGateway({"api_key": "x", "timeout": 0.01})
    g.adapter = Hanging()
    with pytest.raises(TimeoutError):
        _ = [c async for c in await g.generate([], stream=True)]
    fb = LLMGateway({"api_key": "x", "retry": {"max_attempts": 2, "base_delay": 0}})
    fb.adapter = Adapter(error=TimeoutError())
    g.adapter = Adapter(error=TimeoutError())
    g.retry_policy = {"max_attempts": 1}
    g.set_fallback(fb)
    with pytest.raises(TimeoutError):
        _ = [c async for c in await g.generate([], stream=True)]
    assert fb.adapter.calls == 2


@pytest.mark.asyncio
async def test_sdk_mocktransport_and_borrowed_lifecycle():
    import json

    import httpx
    from openai import AsyncOpenAI

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "x",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    client = AsyncOpenAI(
        api_key="x", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    g = LLMGateway({"api_key": "x", "client": client, "temperature": 0.73, "max_tokens": 17})
    response = await g.generate([Message(role="user", content="hi")])
    assert response.content == "ok" and response.usage_available
    assert seen[0]["temperature"] == 0.73 and seen[0]["max_tokens"] == 17
    await g.aclose()
    assert not client.is_closed()
    await client.close()


def test_typed_sdk_events_and_unknown_finish():
    from openai.types.chat import ChatCompletionChunk

    a = OpenAIAdapter("x")
    event = ChatCompletionChunk(
        id="x",
        created=1,
        model="x",
        object="chat.completion.chunk",
        choices=[],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    assert a.parse_stream_chunk(event)["usage"]["total_tokens"] == 0
    from anthropic.types import RawMessageStartEvent

    e = RawMessageStartEvent(
        type="message_start",
        message={
            "id": "x",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "x",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    )
    assert AnthropicAdapter("x").parse_stream_chunk(e)["usage"]["prompt_tokens"] == 100
    assert (
        LLMResponse(message=Message(role="assistant"), finish_reason="mystery").finish_reason
        == "unknown"
    )


@pytest.mark.asyncio
async def test_owned_lifecycle_registry_alias_and_retry_whitelist():
    class Client:
        closed = False

        async def close(self):
            self.closed = True

    c = Client()
    g = LLMGateway({"api_key": "x", "client": c, "owns_client": True})
    await g.aclose()
    assert c.closed
    with pytest.warns(DeprecationWarning):
        legacy = LLMGateway({"provider": "ollama"})
    assert str(legacy.adapter.client.base_url) == "http://localhost:11434/v1/"
    await legacy.aclose()
    LLMGateway.register_provider("offline_test", lambda **kw: Adapter(), default_model="test")
    assert isinstance(LLMGateway({"provider": "offline_test"}).adapter, Adapter)
    g = LLMGateway({"api_key": "x", "retry": {"max_attempts": 3, "base_delay": 0}})
    a = Adapter(error=TypeError("bad"))
    g.adapter = a
    a.parse_response = lambda r: r
    with pytest.raises(TypeError):
        await g.generate([])
    assert a.calls == 1


@pytest.mark.asyncio
async def test_anthropic_split_usage_and_empty_arguments_rejected():
    g = LLMGateway({"api_key": "x"})
    g.adapter = Adapter(
        [
            {"usage": {"prompt_tokens": 105, "completion_tokens": 0, "total_tokens": 105}},
            {
                "usage": {"prompt_tokens": 0, "completion_tokens": 20, "total_tokens": 20},
                "finish_reason": "end_turn",
            },
        ]
    )
    chunks = [c async for c in await g.generate([], stream=True)]
    assert chunks[-1].usage["total_tokens"] == 125
    g.adapter = Adapter(
        [{"tool_calls_delta": [{"index": 0, "id": "c", "name": "f"}]}, {"finish_reason": "stop"}]
    )
    with pytest.raises(ValueError):
        _ = [c async for c in await g.generate([], stream=True)]


def test_anthropic_parallel_results_and_truncation():
    from wuwei.core.message import ToolMessage

    a = AnthropicAdapter("x")
    req = a.build_request(
        [
            ToolMessage(content="1", tool_call_id="a", name="f"),
            ToolMessage(content="2", tool_call_id="b", name="g"),
        ]
    )
    assert len(req["messages"]) == 1 and len(req["messages"][0]["content"]) == 2
    r = a.parse_response(
        Ns(
            content=[Ns(type="text", text="partial")],
            stop_reason="max_tokens",
            usage=None,
            model="x",
        )
    )
    assert r.finish_reason == "length" and r.raw_finish_reason == "max_tokens"


@pytest.mark.asyncio
async def test_signed_reasoning_is_retained_and_replayed():
    g = LLMGateway({"api_key": "x"})
    g.adapter = Adapter(
        [
            {
                "provider_metadata": {
                    "anthropic_reasoning_start": {
                        "index": 0,
                        "block": {"type": "thinking", "thinking": "", "signature": ""},
                    }
                }
            },
            {"reasoning_content": "thought"},
            {
                "provider_metadata": {
                    "anthropic_signature_delta": {"index": 0, "signature": "signed"}
                }
            },
            {"finish_reason": "end_turn"},
        ]
    )
    chunks = [c async for c in await g.generate([], stream=True)]
    blocks = chunks[-1].metadata["anthropic_reasoning"]
    assert blocks == [{"type": "thinking", "thinking": "thought", "signature": "signed"}]
    msg = Message(role="assistant", content="answer", metadata={"anthropic_reasoning": blocks})
    a = AnthropicAdapter("x")
    assert a.build_request([msg])["messages"][0]["content"][0] == blocks[0]


def test_explicit_responses_budget_and_reasoning_capability():
    g = LLMGateway({"provider": "responses", "api_key": "x", "max_output_tokens": 99})
    assert g.adapter.build_request([])["max_output_tokens"] == 99
    a = OpenAIAdapter("x", supports_reasoning_content=True)
    message = Message(role="assistant", content="answer", reasoning_content="reason")
    assert a.build_request([message])["messages"][0]["reasoning_content"] == "reason"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_fallback_shares_overall_deadline(streaming):
    import asyncio
    import time

    class Slow(Adapter):
        async def call(self, request):
            await asyncio.sleep(0.06)
            if request.get("stream"):

                async def events():
                    yield {"finish_reason": "stop"}

                return events()
            return LLMResponse(message=Message(role="assistant"), finish_reason="stop")

        def parse_response(self, raw):
            return raw

    g = LLMGateway({"api_key": "x", "timeout": 0.02, "retry": {"max_attempts": 1}})
    fallback = LLMGateway({"api_key": "x", "timeout": 0.2})
    g.adapter = Adapter(error=TimeoutError())
    fallback.adapter = Slow()
    g.set_fallback(fallback)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        if streaming:
            _ = [c async for c in await g.generate([], stream=True)]
        else:
            await g.generate([])
    assert time.monotonic() - started < 0.055
