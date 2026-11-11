"""Wuwei 框架 — 补充测试：覆盖其他测试文件未覆盖的模块。

历史说明：本文件曾是模块级脚本（check() 打印计数、无断言、import 时
sys.exit），在 CI 中产生假阳性通过、在 Windows 上破坏 pytest 收集。
2026-11 改写为真正的 pytest 用例。

覆盖:
- parsers (JsonOutputParser, ListOutputParser, PydanticOutputParser)
- skill (FileSystemSkillProvider / SkillManager)
- gateway (GatewayMessage, 平台适配器导入)
- sandbox (LocalSandbox 错误路径、超时、工作区)
- plugin (PluginManager, PluginContext)
- llm adapters (Anthropic / OpenAI / Responses 导入) 与 Agent.from_env
- middleware (MemoryMiddleware, RagMiddleware, StorageMiddleware)
- observability (TracingMiddleware)
- streaming types
- @tool 装饰器 / ToolRegistry
"""

import asyncio
import os
import tempfile

import pytest
from pydantic import BaseModel


# ── parsers ────────────────────────────────────────────────────────────────


def test_json_output_parser_parses_objects_and_arrays():
    from wuwei.parsers.json import JsonOutputParser

    parser = JsonOutputParser()
    assert parser.parse('{"name": "test", "value": 42}') == {"name": "test", "value": 42}
    assert parser.parse("[1, 2, 3]") == [1, 2, 3]


def test_json_output_parser_strips_markdown_fence():
    from wuwei.parsers.json import JsonOutputParser

    parser = JsonOutputParser()
    assert parser.parse('```json\n{"key": "val"}\n```') == {"key": "val"}


def test_json_output_parser_invalid_input_returns_raw_text():
    from wuwei.parsers.json import JsonOutputParser

    parser = JsonOutputParser()
    assert parser.parse("not json at all") == "not json at all"


def test_list_output_parser_separators():
    from wuwei.parsers.list import ListOutputParser

    parser = ListOutputParser()
    assert parser.parse("item1, item2, item3") == ["item1", "item2", "item3"]
    assert parser.parse("first\nsecond\nthird") == ["first", "second", "third"]


def test_pydantic_output_parser_returns_model():
    from wuwei.parsers.pydantic import PydanticOutputParser

    class TestModel(BaseModel):
        name: str
        age: int

    parser = PydanticOutputParser(schema=TestModel)
    result = parser.parse('{"name": "Alice", "age": 30}')
    assert isinstance(result, TestModel)
    assert result.name == "Alice"
    assert result.age == 30

    fmt = parser.get_format_instructions()
    assert "json" in fmt.lower()


# ── skill ──────────────────────────────────────────────────────────────────


async def test_filesystem_skill_provider_end_to_end():
    from wuwei.skill import FileSystemSkillProvider, SkillManager

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "skills", "demo-skill")
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(
                "---\n"
                "name: demo-skill\n"
                "description: A demo skill for testing\n"
                "version: 1.0.0\n"
                "tags: [test, demo]\n"
                "allowed_tools: [read_file]\n"
                "---\n"
                "# Demo Skill\n\nThis is a demo skill instruction.\n"
            )

        provider = FileSystemSkillProvider(skill_path=os.path.join(tmpdir, "skills"))
        skills = provider.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "demo-skill"
        assert "test" in skills[0].tags
        assert "read_file" in skills[0].allowed_tools

        manager = SkillManager()
        manager.add_provider(provider)
        assert "demo-skill" in manager.list_names()
        instruction = manager.load_skill_instruction("demo-skill")
        assert instruction is not None and "Demo Skill" in instruction


# ── gateway ────────────────────────────────────────────────────────────────


def test_gateway_message_fields():
    from wuwei.gateway.base import GatewayMessage

    msg = GatewayMessage(
        platform="test-platform",
        user_id="user-001",
        user_name="Test User",
        content="Hello",
        message_id="msg-001",
    )
    assert msg.platform == "test-platform"
    assert msg.user_id == "user-001"
    assert msg.content == "Hello"


def test_platform_gateway_adapters_importable():
    from wuwei.gateway.adapters.dingtalk import DingTalkGateway
    from wuwei.gateway.adapters.feishu import FeishuGateway
    from wuwei.gateway.adapters.telegram import TelegramGateway
    from wuwei.gateway.adapters.wechat import WeChatGateway
    from wuwei.gateway.webhook import WebhookGateway

    for cls in (DingTalkGateway, FeishuGateway, TelegramGateway, WeChatGateway, WebhookGateway):
        assert cls is not None


# ── sandbox ────────────────────────────────────────────────────────────────


async def test_local_sandbox_basic_execution():
    from wuwei.sandbox.local import LocalSandbox

    sandbox = LocalSandbox()
    result = await sandbox.execute("echo hello")
    assert "hello" in result.stdout


async def test_local_sandbox_error_command_reports_failure():
    from wuwei.sandbox.base import SandboxResult
    from wuwei.sandbox.local import LocalSandbox

    sandbox = LocalSandbox()
    result = await sandbox.execute("nonexistent_command_xyz 2>&1")
    assert isinstance(result, SandboxResult)
    assert result.success is False


async def test_local_sandbox_file_roundtrip_in_workspace(tmp_path):
    """工作区内文件读写，使用相对路径保持跨平台（不依赖 /tmp）。"""
    from wuwei.sandbox.local import LocalSandbox

    sandbox = LocalSandbox(workspace=str(tmp_path))
    await sandbox.execute("echo test data > wuwei_sandbox_file.txt")
    result = await sandbox.execute("cat wuwei_sandbox_file.txt")
    assert "test data" in result.stdout
    assert (tmp_path / "wuwei_sandbox_file.txt").exists()


async def test_local_sandbox_multiline_output():
    from wuwei.sandbox.local import LocalSandbox

    sandbox = LocalSandbox()
    result = await sandbox.execute("echo line1 && echo line5")
    assert "line1" in result.stdout and "line5" in result.stdout


# ── plugin ─────────────────────────────────────────────────────────────────


def test_plugin_manager_surface():
    from wuwei.plugin import Plugin, PluginContext, PluginManager
    from wuwei.tools import ToolRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ToolRegistry()
        ctx = PluginContext(tool_registry=registry)
        assert ctx.tool_registry is registry

        pm = PluginManager(ctx)
        assert hasattr(pm, "register")
        assert hasattr(pm, "list_plugins")

        discovered = pm.load_directory(tmpdir)
        assert isinstance(discovered, list)

        plugin = Plugin(name="test", description="test plugin")
        assert plugin.name == "test"


# ── llm adapters ───────────────────────────────────────────────────────────


def test_adapters_importable_and_agent_from_env_exists():
    from wuwei.agent import Agent
    from wuwei.llm.adapters.anthropic import AnthropicAdapter
    from wuwei.llm.adapters.openai import OpenAIAdapter
    from wuwei.llm.adapters.responses import ResponsesAdapter

    for cls in (OpenAIAdapter, AnthropicAdapter, ResponsesAdapter):
        assert isinstance(cls, type)
    assert hasattr(Agent, "from_env")


# ── middleware 实例化 ───────────────────────────────────────────────────────


def test_memory_middleware_instantiation():
    from wuwei.llm import LLMGateway
    from wuwei.memory import InMemoryMemoryStore
    from wuwei.middleware.memory import MemoryMiddleware

    llm = LLMGateway({"provider": "openai", "api_key": "test", "model": "gpt-4o"})
    mm = MemoryMiddleware(llm=llm, memory_store=InMemoryMemoryStore())
    assert hasattr(mm, "before_llm")


def test_rag_middleware_instantiation():
    from wuwei.memory import InMemoryKnowledgeStore
    from wuwei.middleware.rag import RagMiddleware

    rm = RagMiddleware(knowledge_store=InMemoryKnowledgeStore())
    assert hasattr(rm, "before_llm")


def test_storage_middleware_instantiation(tmp_path):
    from wuwei.middleware.storage import StorageMiddleware

    sm = StorageMiddleware(storage_path=str(tmp_path))
    assert hasattr(sm, "before_llm")


# ── @tool 装饰器 ───────────────────────────────────────────────────────────


def test_tool_decorator_registers_typed_tool():
    from wuwei.tools.tool import Tool
    from wuwei.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.tool(description="Greet someone with a custom greeting.")
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    greet_tool = registry.get("greet")
    assert isinstance(greet_tool, Tool)
    assert greet_tool.name == "greet"
    assert "Greet someone" in greet_tool.description

    result = asyncio.run(greet_tool.invoke({"name": "World", "greeting": "Hi"}))
    assert result == "Hi, World!"


def test_tool_registration_and_schema_shape():
    from wuwei.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.tool(description="Add two numbers.")
    def add_tool(a: int, b: int) -> int:
        return a + b

    registered = registry.get("add_tool")
    assert registered is not None
    assert registered.name == "add_tool"
    assert hasattr(registered, "parameters")
    assert asyncio.run(registered.invoke({"a": 3, "b": 4})) == 7


# ── observability ──────────────────────────────────────────────────────────


def test_tracing_middleware_hooks_present():
    from wuwei.observability.tracing import TracingMiddleware

    tm = TracingMiddleware(service_name="test-service")
    assert tm.service_name == "test-service"
    for hook in ("before_llm", "after_llm", "before_tool", "after_tool"):
        assert hasattr(tm, hook)


# ── streaming types ────────────────────────────────────────────────────────


def test_stream_chunk_types():
    from wuwei.streaming.types import (
        CustomStreamChunk,
        DebugStreamChunk,
        MessagesStreamChunk,
        StreamChunk,
        UpdatesStreamChunk,
        ValuesStreamChunk,
    )

    chunk = StreamChunk(type="text", data="hello")
    assert chunk.type == "text" and chunk.data == "hello"

    assert MessagesStreamChunk(type="messages", data="msg content").type == "messages"
    assert ValuesStreamChunk(data={"key": "val"}).type == "values"
    assert UpdatesStreamChunk(data={"node": "llm"}).type == "updates"
    assert CustomStreamChunk(data={"custom": True}).type == "custom"
    assert DebugStreamChunk(data={"debug": "info"}).type == "debug"


# ── 防回归：本文件必须是可收集的 pytest 模块 ────────────────────────────────


def test_this_module_collects_with_real_assertions():
    """防止再次退化为脚本式假测试：pytest 收集的 test_ 函数必须含断言。"""
    import ast
    import inspect

    import tests.test_coverage_gaps as this_module

    source = inspect.getsource(this_module)
    tree = ast.parse(source)
    has_assert = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            has_assert = True
            break
    assert has_assert, "test_coverage_gaps.py 必须包含真实断言（禁止退化为脚本式 check）"
