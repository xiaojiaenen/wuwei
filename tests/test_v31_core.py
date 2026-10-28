"""v3.1 核心重构测试

覆盖重写后的关键路径（旧测试没有的部分）：
1. AgentRunner 单一循环：非流式/流式/usage 累计/默认 session 复用
2. 真并行工具执行
3. HITL 拒绝回喂（不再炸穿 run）
4. 子代理工具在流式模式下可用（wrap_model_call 注入）
5. 三个适配器的协议契约（parse_response / parse_stream_chunk）
6. DecisionClient（Jev）判定解析与 fail-closed 行为
"""

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from wuwei.agent import Agent
from wuwei.agent.session import AgentSession
from wuwei.core.message import BaseMessage as Message, ToolCall, FunctionCall
from wuwei.llm import LLMGateway, LLMResponse
from wuwei.llm.adapters.openai import OpenAIAdapter
from wuwei.llm.adapters.anthropic import AnthropicAdapter
from wuwei.llm.adapters.responses import ResponsesAdapter
from wuwei.llm.types import LLMResponseChunk
from wuwei.middleware import Middleware, MiddlewareStack
from wuwei.middleware.hitl import HitlMiddleware
from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware
from wuwei.decision import DecisionClient, DecisionUnavailable, LowConfidence, Noul, Choice
from wuwei.tools import Tool, ToolParameters, ToolRegistry


# ── 测试工具 ──────────────────────────────────────────────────────────


def make_tool(name: str, output, sleep: float = 0.0, concurrency_safe: bool = True):
    async def _run(**kwargs):
        if sleep:
            await __import__("asyncio").sleep(sleep)
        return output

    return Tool(
        name=name,
        description=f"test tool {name}",
        parameters=ToolParameters(),
        handler=_run,
    )


class FakeAdapter:
    """脚本化假适配器：按序回放响应。"""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def build_request(self, messages, tools=None, stream=False, **kwargs):
        self.calls.append({
            "messages": messages,
            "tools": tools or [],
            "stream": stream,
        })
        return {"messages": messages, "tools": tools, "stream": stream}

    async def call(self, request):
        return request

    def parse_response(self, raw):
        return self.responses.pop(0)

    def parse_stream_chunk(self, chunk):
        return None


def make_llm_response(content="", tool_calls=None, usage=None):
    return LLMResponse(
        message=Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls or [],
        ),
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="fake-model",
        latency_ms=0,
    )


class FakeGateway:
    """runner 所需的最小网关接口：generate(messages, tools, stream)。"""

    def __init__(self, responses: list[LLMResponse]):
        self.adapter = FakeAdapter(responses)

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        self.adapter.calls.append({
            "messages": messages,
            "tools": tools or [],
            "stream": stream,
        })
        resp = self.adapter.responses.pop(0)
        if not stream:
            return resp

        async def _chunk_gen():
            yield LLMResponseChunk(
                content=resp.message.content,
                finish_reason=resp.finish_reason,
                tool_calls_complete=resp.message.tool_calls or None,
                usage=resp.usage,
            )

        return _chunk_gen()


def make_agent(responses: list[LLMResponse], middleware: MiddlewareStack | None = None, **kwargs) -> Agent:
    return Agent(llm=FakeGateway(responses), load_builtins=False, middleware=middleware, **kwargs)


# ── Runner 单一循环 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_accumulates_usage_across_steps():
    """多步任务的 usage 必须逐次累加（旧版只统计最后一次调用）。"""
    tool_call = ToolCall(
        id="call_1", function=FunctionCall(name="echo", arguments={"text": "hi"})
    )
    responses = [
        make_llm_response(tool_calls=[tool_call], usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
        make_llm_response(content="done", usage={"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}),
    ]
    agent = make_agent(responses)
    agent.tool_registry.register(make_tool("echo", "hi"))

    result = await agent.run("hello")
    assert result.content == "done"
    assert result.usage["total_tokens"] == 150 + 220
    assert result.llm_calls == 2


@pytest.mark.asyncio
async def test_run_without_session_reuses_default_session():
    """不传 session 连续 run() 应复用默认会话（多轮历史连续）。"""
    responses = [make_llm_response(content="第一轮"), make_llm_response(content="第二轮")]
    agent = make_agent(responses)

    await agent.run("你好")
    await agent.run("继续")
    messages = agent._default_session.context.get_messages()
    contents = [m.content for m in messages]
    assert "你好" in contents and "继续" in contents
    assert contents.count("第一轮") == 1 and contents.count("第二轮") == 1


@pytest.mark.asyncio
async def test_stream_run_returns_events():
    """run(stream=True) 必须返回事件流（旧版调用不存在的方法直接 AttributeError）。"""
    agent = make_agent([make_llm_response(content="流式回答")])
    events = []
    async for event in await agent.run("hi", stream=True):
        events.append(event)

    types = [e.type for e in events]
    assert "run_start" in types
    assert "text_delta" in types
    assert types[-1] == "run_end"


@pytest.mark.asyncio
async def test_tool_result_fed_back_to_llm():
    """工具结果必须写回会话历史并在下一轮 LLM 调用中可见。"""
    tool_call = ToolCall(id="c1", function=FunctionCall(name="db", arguments={}))
    responses = [
        make_llm_response(tool_calls=[tool_call]),
        make_llm_response(content="final"),
    ]
    gateway = FakeGateway(responses)
    adapter = gateway.adapter
    agent = Agent(llm=gateway, load_builtins=False)
    agent.tool_registry.register(make_tool("db", {"rows": 3}))

    await agent.run("query")
    second_call_messages = adapter.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert '"rows": 3' in tool_messages[0].content or "rows" in tool_messages[0].content


@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """parallel_tool_calls=True 时并发安全的工具应真正并发执行。"""
    import asyncio

    timestamps = []

    async def slow_a(**kwargs):
        await asyncio.sleep(0.15)
        timestamps.append("a_end")
        return "A"

    async def slow_b(**kwargs):
        await asyncio.sleep(0.15)
        timestamps.append("b_end")
        return "B"

    tool_a = Tool(name="slow_a", description="a", parameters=ToolParameters(), handler=slow_a)
    tool_b = Tool(name="slow_b", description="b", parameters=ToolParameters(), handler=slow_b)

    responses = [
        make_llm_response(tool_calls=[
            ToolCall(id="c1", function=FunctionCall(name="slow_a", arguments={})),
            ToolCall(id="c2", function=FunctionCall(name="slow_b", arguments={})),
        ]),
        make_llm_response(content="both done"),
    ]
    agent = make_agent(responses)
    agent.tool_registry.register(tool_a)
    agent.tool_registry.register(tool_b)

    session = agent.create_session(parallel_tool_calls=True)
    start = time.monotonic()
    await agent.run("go", session=session)
    elapsed = time.monotonic() - start

    # 串行需要 ~0.3s，并发应在 ~0.2s 内
    assert elapsed < 0.28, f"工具疑似串行执行，耗时 {elapsed:.2f}s, order={timestamps}"


@pytest.mark.asyncio
async def test_hitl_rejection_fed_back_not_raised():
    """HITL 拒绝：run 正常完成，模型收到 approval_rejected 的工具结果。"""
    async def deny_all(tool_call):
        return False

    hitl = HitlMiddleware(approval_provider=deny_all)
    tool_call = ToolCall(id="c1", function=FunctionCall(name="rm_rf", arguments={}))
    responses = [
        make_llm_response(tool_calls=[tool_call]),
        make_llm_response(content="好的，我不执行该操作"),
    ]
    agent = make_agent(responses, middleware=None)
    agent.tool_registry.register(make_tool("rm_rf", "deleted"))
    agent.middleware.add(hitl)

    result = await agent.run("删除一切")
    assert result.content == "好的，我不执行该操作"

    tool_msgs = [
        m for m in agent._default_session.context.get_messages() if m.role == "tool"
    ]
    assert any("approval_rejected" in m.content for m in tool_msgs)


@pytest.mark.asyncio
async def test_sub_agent_tools_available_in_streaming():
    """流式模式下子代理工具必须通过 wrap_model_call 注入（旧版流式路径失效）。"""
    sub = SubAgent(
        name="worker",
        description="do work",
        system_prompt="worker",
    )
    sub_mw = SubAgentMiddleware(sub_agents=[sub], parent_llm=None)
    agent = make_agent([make_llm_response(content="hi")])
    agent.middleware.add(sub_mw)

    events = []
    async for ev in agent.stream_events("hello"):
        events.append(ev)

    llm_start = next(e for e in events if e.type == "llm_start")
    adapter_call = agent.llm.adapter.calls[0]
    tool_names = [
        t["function"]["name"] if isinstance(t, dict) and "function" in t else getattr(t, "name", "")
        for t in adapter_call["tools"]
    ]
    assert "task_worker" in tool_names


# ── 适配器契约测试 ────────────────────────────────────────────────────


class TestOpenAIAdapterContract:
    def test_parse_response(self):
        adapter = OpenAIAdapter(api_key="k", model="gpt-4o")
        raw = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="hello",
                    tool_calls=[SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(name="search", arguments='{"q": "x"}'),
                    )],
                ),
                finish_reason="tool_calls",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4o",
        )
        resp = adapter.parse_response(raw)
        assert resp.message.content == "hello"
        assert resp.message.tool_calls[0].function.name == "search"
        assert resp.message.tool_calls[0].function.arguments == {"q": "x"}
        assert resp.usage["total_tokens"] == 15
        assert resp.finish_reason == "tool_calls"

    def test_parse_usage_none_does_not_crash(self):
        """usage 缺失不应崩溃（部分兼容端点不返回 usage）。"""
        adapter = OpenAIAdapter(api_key="k", model="gpt-4o")
        raw = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=None), finish_reason="stop")],
            usage=None,
            model="m",
        )
        resp = adapter.parse_response(raw)
        assert resp.usage["total_tokens"] == 0


class TestAnthropicAdapterContract:
    def test_tool_arguments_delta_not_leaked_into_content(self):
        """input_json_delta 必须走 tool_calls_delta 通道，不得污染文本流。"""
        adapter = AnthropicAdapter(api_key="k")
        chunk = SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"q": "x'),
        )
        data = adapter.parse_stream_chunk(chunk)
        assert data["content"] == ""
        assert data["tool_calls_delta"][0]["arguments"] == '{"q": "x'
        assert data["tool_calls_delta"][0]["index"] == 1

    def test_build_request_sets_stream(self):
        adapter = AnthropicAdapter(api_key="k")
        request = adapter.build_request(
            [Message(role="user", content="hi")], stream=True
        )
        assert request["stream"] is True

    def test_system_prompt_caching(self):
        adapter = AnthropicAdapter(api_key="k")
        request = adapter.build_request(
            [Message(role="system", content="sys"), Message(role="user", content="hi")],
        )
        assert request["system"][0]["cache_control"] == {"type": "ephemeral"}


class TestResponsesAdapterContract:
    def test_parse_response(self):
        adapter = ResponsesAdapter(api_key="k")
        raw = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="answer")],
                ),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_9",
                    name="get_weather",
                    arguments='{"city": "北京"}',
                ),
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8, total_tokens=28),
            model="gpt-4o",
        )
        resp = adapter.parse_response(raw)
        assert resp.message.content == "answer"
        assert resp.finish_reason == "tool_calls"
        assert resp.message.tool_calls[0].id == "call_9"
        assert resp.message.tool_calls[0].function.arguments == {"city": "北京"}


# ── DecisionClient（Jev） ────────────────────────────────────────────


def _jev_response(answers: dict) -> httpx.Response:
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2}})


class TestDecisionClient:
    @pytest.mark.asyncio
    async def test_judge_parses_all_types(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _jev_response({
                "is_urgent": {"type": "noul", "noul": 0.95},
                "department": {
                    "type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.88, "technical": 0.12},
                    "confidence": 0.81,
                },
            })

        client = DecisionClient(api_key="k", transport=httpx.MockTransport(handler))
        answers = await client.judge(
            "payout failing",
            {"is_urgent": Noul("urgent?"), "department": Choice("who?", {"billing": None, "technical": None})},
        )
        assert answers["is_urgent"].value == 0.95
        assert answers["department"].value == "billing"
        assert answers["department"].confidence == 0.81

    @pytest.mark.asyncio
    async def test_approve_fail_closed_on_service_error(self):
        """决策服务不可达时必须升级人工（fail-closed）。"""
        def handler(request):
            raise httpx.ConnectError("boom")

        client = DecisionClient(api_key="k", max_retries=0, transport=httpx.MockTransport(handler))
        verdict = await client.approve("delete database", threshold=0.9)
        assert verdict == "escalate"

    @pytest.mark.asyncio
    async def test_route_low_confidence_raises(self):
        def handler(request):
            return _jev_response({
                "route": {"type": "choice", "choice": "a", "probabilities": {"a": 0.5}, "confidence": 0.4},
            })

        client = DecisionClient(api_key="k", transport=httpx.MockTransport(handler))
        with pytest.raises(LowConfidence):
            await client.route("state", ["a", "b"], min_confidence=0.7)
