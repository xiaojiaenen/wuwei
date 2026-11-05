"""P0 回归：流式 Accept 头与批次取消语义。

#18：OpenAI 兼容网关按 Accept 门控响应格式。若流式请求仍带
Accept: application/json，上游返回裸 JSON，SSE 解码器得到 0 个事件，
最终报一句与真实原因无关的 "stream ended without a terminal event"。

R-8：asyncio.gather(return_exceptions=True) 会把 CancelledError 当作结果
返回；若用 isinstance(result, Exception) 判定，它既不是 Exception 也不被
重新抛出，就被当成工具结果塞进列表，后续 model_dump 直接崩。
"""

import asyncio

import pytest

from wuwei.core.message import HumanMessage
from wuwei.llm.adapters.base import BaseAdapter
from wuwei.llm.adapters.openai import OpenAIAdapter
from wuwei.llm.adapters.responses import ResponsesAdapter
from wuwei.llm.gateway import LLMGateway
from wuwei.llm.types import FunctionCall, ToolCall
from wuwei.tools.executor import ToolExecutor
from wuwei.tools.registry import ToolRegistry


def _adapter(cls, **kwargs):
    return cls(api_key="test-key", **kwargs)


def test_stream_request_forces_sse_accept():
    adapter = _adapter(OpenAIAdapter)
    request = adapter.build_request([HumanMessage(content="hi")], stream=True)
    assert request["extra_headers"]["Accept"] == "text/event-stream"


def test_non_stream_request_keeps_sdk_default_accept():
    adapter = _adapter(OpenAIAdapter)
    request = adapter.build_request([HumanMessage(content="hi")], stream=False)
    assert "extra_headers" not in request


def test_caller_cannot_downgrade_stream_accept():
    adapter = _adapter(OpenAIAdapter)
    request = adapter.build_request(
        [HumanMessage(content="hi")],
        stream=True,
        extra_headers={"Accept": "application/json", "X-Trace": "abc"},
    )
    assert request["extra_headers"]["Accept"] == "text/event-stream"
    assert request["extra_headers"]["X-Trace"] == "abc"


def test_responses_stream_request_forces_sse_accept():
    adapter = _adapter(ResponsesAdapter)
    request = adapter.build_request([HumanMessage(content="hi")], stream=True)
    assert request["extra_headers"]["Accept"] == "text/event-stream"


def test_responses_non_stream_request_keeps_sdk_default_accept():
    adapter = _adapter(ResponsesAdapter)
    request = adapter.build_request([HumanMessage(content="hi")], stream=False)
    assert "extra_headers" not in request


class _EmptyStreamAdapter(BaseAdapter):
    """模拟忽略 Accept、返回非 SSE 正文的兼容网关：SDK 产出 0 个事件。"""

    provider_name = "openai"
    model = "empty"

    def build_request(self, messages, tools=None, stream=False, **kwargs):
        return {"stream": True, "messages": []}

    async def call(self, request):
        async def stream():
            if False:  # pragma: no cover - 保持空异步生成器
                yield None

        return stream()

    def parse_response(self, raw):
        raise NotImplementedError

    def parse_stream_chunk(self, chunk):
        return None


@pytest.mark.asyncio
async def test_zero_event_stream_reports_accept_as_cause():
    gateway = LLMGateway(
        {
            "provider": "openai",
            "api_key": "test-key",
            "adapter": _EmptyStreamAdapter(),
            "retry": {"max_attempts": 1},
        }
    )
    with pytest.raises(ValueError, match="produced no events"):
        stream = await gateway.generate([HumanMessage(content="hi")], stream=True)
        async for _ in stream:
            pass


def _call(name, arguments=None):
    return ToolCall(function=FunctionCall(name=name, arguments=arguments or {}))


def _registry(*handlers):
    registry = ToolRegistry()
    for handler in handlers:
        registry.register_callable(handler)
    return registry


async def _returns_ok() -> str:
    """正常工具。"""
    return "ok"


async def _raises_runtime():
    """普通异常。"""
    raise RuntimeError("boom")


async def _raises_cancel():
    """取消。"""
    raise asyncio.CancelledError()


class _Fatal(BaseException):
    """非 Exception 的 BaseException，模拟任务级致命错误。"""


async def _raises_fatal():
    """BaseException 而非 Exception。"""
    raise _Fatal("fatal")


@pytest.mark.asyncio
async def test_batch_cancellation_propagates():
    executor = ToolExecutor(_registry(_returns_ok, _raises_cancel))
    with pytest.raises(asyncio.CancelledError):
        await executor.execute([_call("_raises_cancel"), _call("_returns_ok")], concurrent=True)


@pytest.mark.asyncio
async def test_batch_base_exception_propagates():
    executor = ToolExecutor(_registry(_returns_ok, _raises_fatal))
    with pytest.raises(_Fatal):
        await executor.execute([_call("_raises_fatal"), _call("_returns_ok")], concurrent=True)


@pytest.mark.asyncio
async def test_batch_exception_still_becomes_tool_message():
    executor = ToolExecutor(_registry(_returns_ok, _raises_runtime))
    messages = await executor.execute([_call("_raises_runtime"), _call("_returns_ok")], True)
    assert len(messages) == 2
    assert all(isinstance(m, object) and hasattr(m, "role") for m in messages)
    assert messages[0].role == "tool"
    assert messages[1].content == "ok"
