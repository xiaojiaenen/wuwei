"""2026-11 审计修复验证脚本（纯离线，fake LLM，不联网）。

历史基线（3.3.3，修复前）与修复后断言：
1. 检查点写入放大：40 步 201 次保存 / 13.32 MB → 现在 ≤130 次 / ≤6 MB。
2. Ledger 归档的是被 executor 截断后的内容 → 现在归档未截断原文、占位符
   报告真实长度、丢失 0 字符。
3. prompted 结构化输出把 system 追加到尾部 → 现在与 runner 共用
   inject_prompted_schema 并入首条 system；兼容端点 auto 优先选择 tool 模式。

运行：uv run python docs/audits/2026-11-11/reproduce_efficiency_gaps.py
任一断言失败即以非零码退出，可作为改造前后的门禁。
"""

from __future__ import annotations

import asyncio
import json
import sys

from wuwei import Agent, MiddlewareStack, ToolRegistry
from wuwei.core.message import HumanMessage, SystemMessage
from wuwei.graph.checkpoint import MemoryCheckpointer
from wuwei.llm.gateway import LLMGateway
from wuwei.llm.output import OutputSpec, STRUCTURED_SCHEMA_MARKER
from wuwei.llm.types import FunctionCall, LLMResponse, Message, ToolCall
from wuwei.middleware.ledger import ContextLedger, LedgerMiddleware, ledger_tools


class FakeLLM:
    """按步数返回工具调用的假模型，只为驱动 runner。"""

    model = "fake"
    provider = "fake"

    def __init__(self, steps: int, tool_name: str = "echo", arguments=None) -> None:
        self.steps = steps
        self.tool_name = tool_name
        self.arguments = {"text": "hi"} if arguments is None else arguments
        self.calls = 0

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        self.calls += 1
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self.calls <= self.steps:
            return LLMResponse(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{self.calls}",
                            type="function",
                            function=FunctionCall(name=self.tool_name, arguments=self.arguments),
                        )
                    ],
                ),
                finish_reason="tool_calls",
                usage=usage,
                model="fake",
            )
        return LLMResponse(
            message=Message(role="assistant", content="done"),
            finish_reason="stop",
            usage=usage,
            model="fake",
        )


class CountingCheckpointer(MemoryCheckpointer):
    def __init__(self) -> None:
        super().__init__()
        self.saves = 0
        self.bytes_written = 0
        self.last_size = 0
        self.kinds: dict[str, int] = {}

    async def save(self, state, checkpoint_id=None, metadata=None):
        self.saves += 1
        blob = json.dumps({"messages": state.messages, "metadata": metadata}, default=str)
        size = len(blob.encode())
        self.bytes_written += size
        self.last_size = size
        kind = (metadata or {}).get("kind", "?")
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        return await super().save(state, checkpoint_id, metadata)


async def probe_checkpoint_amplification() -> None:
    print("=" * 78)
    print("1) 检查点写入（修复后门禁：40 步 ≤130 次保存 / ≤6 MB）")
    print("=" * 78)
    for steps in (5, 10, 20, 40):
        registry = ToolRegistry()

        @registry.tool()
        def echo(text: str) -> str:
            """回显。"""
            return text * 200

        cp = CountingCheckpointer()
        agent = Agent(llm=FakeLLM(steps), tools=registry, checkpointer=cp, load_builtins=False)
        session = agent.create_session()
        session.max_steps = steps + 3
        await agent.run("go", session=session)
        print(
            f"  steps={steps:3d} saves={cp.saves:4d} written={cp.bytes_written / 1024:9.1f}KB "
            f"last={cp.last_size / 1024:7.1f}KB kinds={cp.kinds}"
        )
        if steps == 40:
            assert cp.saves <= 130, f"保存次数回归：{cp.saves} > 130（修复前 201）"
            assert cp.bytes_written <= 6 * 1024 * 1024, (
                f"写出量回归：{cp.bytes_written / 1024 / 1024:.2f} MB > 6 MB（修复前 13.32 MB）"
            )
            assert cp.kinds.get("tool_running", 0) <= 45, (
                f"tool_running 重复写入回归：{cp.kinds.get('tool_running')}（修复前 80）"
            )
            cid = (await cp.list_checkpoints(limit=1))[0]["id"]
            saved = await cp.load(cid)
            meta = saved.metadata
            assert meta.get("session_id"), "v2 加载必须还原 metadata（含 session）"
            assert saved.messages, "v2 加载必须从会话副本重灌 messages"
            sess = json.dumps(meta.get("session", {}), default=str)
            log = json.dumps(
                meta.get("session", {}).get("metadata", {}).get("action_log", {}), default=str
            )
            msgs = json.dumps(saved.messages, default=str)
            print(
                f"          末次检查点构成: 总计 {cp.last_size / 1024:.1f}KB | "
                f"session {len(sess.encode()) / 1024:.1f}KB（其中 action_log "
                f"{len(log.encode()) / 1024:.1f}KB）| 重灌 messages {len(msgs.encode()) / 1024:.1f}KB"
            )
            print(f"          run_state 键: {list(meta.get('run_state', {}))}")


async def probe_ledger_truncation() -> None:
    print()
    print("=" * 78)
    print("2) Ledger 全量保真（修复后：归档未截断原文，丢失 0 字符）")
    print("=" * 78)
    registry = ToolRegistry()

    @registry.tool()
    def big() -> str:
        """返回 150k 字符。"""
        return "A" * 150_000

    ledger = ContextLedger()
    stack = MiddlewareStack(LedgerMiddleware(ledger, archive_threshold=2000))
    agent = Agent(
        llm=FakeLLM(1, tool_name="big", arguments={}),
        tools=registry,
        middleware=stack,
        load_builtins=False,
    )
    for tool in ledger_tools(ledger):
        registry.register(tool)
    await agent.run("go")

    tool_msgs = [m for m in agent._default_session.context.get_messages() if m.role == "tool"]
    entries = ledger.list_entries()
    archived = sum(len(e.content) for e in entries)
    print(f"  工具原始返回: 150000 字符")
    print(f"  窗口内工具消息: {len(tool_msgs[0].content)} 字符 -> {tool_msgs[0].content[:40]}...")
    print(f"  账本归档长度: {[len(e.content) for e in entries]} 字符")
    print(f"  结论: 丢失 {150_000 - archived} 字符")
    assert archived == 150_000, f"归档不完整：{archived} != 150000（修复前 100027）"
    assert "原文 150000 字符" in tool_msgs[0].content, "占位符必须报告真实原文长度"


def probe_prompted_system() -> None:
    print()
    print("=" * 78)
    print("3) 结构化输出注入（修复后：并入首条 system；auto 优先 tool 模式）")
    print("=" * 78)
    messages = [SystemMessage("sys"), HumanMessage("hi")]
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}

    explicit = LLMGateway({"provider": "openai", "api_key": "x", "model": "gpt-4o"})
    req = explicit.build_request(
        messages, stream=False, output_spec=OutputSpec(schema=schema, mode="prompted")
    )
    roles = [m["role"] for m in req["messages"]]
    print("  mode=prompted 消息角色:", roles)
    assert roles == ["system", "user"], f"prompted 注入必须并入首条 system（实际 {roles}）"
    assert req["messages"][0]["content"].count(STRUCTURED_SCHEMA_MARKER) == 1

    compat = LLMGateway(
        {"provider": "openai", "api_key": "x", "model": "qwen", "base_url": "http://x/v1"}
    )
    spec = OutputSpec(schema=schema, mode="auto")
    resolved = spec.resolve(compat.model_profile, has_tools=False)
    req2 = compat.build_request(messages, stream=False, output_spec=spec)
    tools = req2.get("tools") or []
    print("  兼容端点 auto 解析为:", resolved)
    print("  实际 tools:", [t["function"]["name"] for t in tools])
    assert resolved == "tool", f"兼容端点 auto 应优先 tool 模式（实际 {resolved}）"
    assert tools and tools[0]["function"]["name"] == "__wuwei_structured_output"
    roles2 = [m["role"] for m in req2["messages"]]
    assert roles2 == ["system", "user"], f"tool 模式不应注入任何 schema system（实际 {roles2}）"


if __name__ == "__main__":
    asyncio.run(probe_checkpoint_amplification())
    asyncio.run(probe_ledger_truncation())
    probe_prompted_system()
    print()
    print("全部断言通过 ✅")
    sys.exit(0)
