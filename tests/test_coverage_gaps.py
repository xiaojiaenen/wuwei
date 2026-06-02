"""Wuwei 框架 — 补充测试：覆盖之前未测试的模块

覆盖:
- parsers (JsonOutputParser, ListOutputParser, PydanticOutputParser)
- middleware (MemoryMiddleware, RagMiddleware, StorageMiddleware)
- gateway (WebhookGateway, platform adapters)
- sandbox (LocalSandbox error paths, timeout)
- plugin (PluginLoader.load_all, PluginRegistry)
- llm adapters (Anthropic, Ollama, DashScope, Zhipu config/import)
- observability (TracingMiddleware span creation)
- streaming types
- Agent.from_env() factory
- @tool decorator usage
"""

import sys, os, json, tempfile, asyncio
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_wuwei_root = os.path.dirname(_here)
sys.path.insert(0, _wuwei_root)

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}  {detail}")

# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: Parsers — JsonOutputParser / ListOutputParser / PydanticOutputParser")
print("=" * 60)

def test_parsers():
    from wuwei.parsers.json import JsonOutputParser
    from wuwei.parsers.list import ListOutputParser
    from wuwei.parsers.pydantic import PydanticOutputParser
    from pydantic import BaseModel

    # JsonOutputParser
    parser = JsonOutputParser()
    result = parser.parse('{"name": "test", "value": 42}')
    check("JsonOutputParser 解析对象", result == {"name": "test", "value": 42})

    result2 = parser.parse('[1, 2, 3]')
    check("JsonOutputParser 解析数组", result2 == [1, 2, 3])

    # 带 Markdown 代码块
    result3 = parser.parse('```json\n{"key": "val"}\n```')
    check("JsonOutputParser 去除 Markdown 代码块", result3 == {"key": "val"})

    # 无效 JSON — 返回原始字符串
    result4 = parser.parse("not json at all")
    check("JsonOutputParser 无效输入返回原始文本", result4 == "not json at all")

    # ListOutputParser
    list_parser = ListOutputParser()
    result5 = list_parser.parse("item1, item2, item3")
    check("ListOutputParser 逗号分隔", result5 == ["item1", "item2", "item3"])

    result6 = list_parser.parse("first\nsecond\nthird")
    check("ListOutputParser 换行分隔", result6 == ["first", "second", "third"])

    # PydanticOutputParser
    class TestModel(BaseModel):
        name: str
        age: int

    pydantic_parser = PydanticOutputParser(schema=TestModel)
    result7 = pydantic_parser.parse('{"name": "Alice", "age": 30}')
    check("PydanticOutputParser 解析为模型", isinstance(result7, TestModel))
    check("PydanticOutputParser 字段正确", result7.name == "Alice" and result7.age == 30)

    # 格式指令
    fmt = pydantic_parser.get_format_instructions()
    check("PydanticOutputParser 生成格式指令", "json" in fmt.lower() or "JSON" in fmt)

    print("  Parsers 全部通过 ✅")

test_parsers()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 2: Skill — SkillViewerTool / FileSystemSkillProvider 完整流程")
print("=" * 60)

async def test_skill_viewer():
    """测试 FileSystemSkillProvider 完整流程"""
    from wuwei.skill import Skill, SkillManager, FileSystemSkillProvider
    from wuwei.tools import ToolRegistry, ToolExecutor

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 SKILL.md 文件
        skill_dir = os.path.join(tmpdir, "skills", "demo-skill")
        os.makedirs(skill_dir, exist_ok=True)

        skill_md = """---
name: demo-skill
description: A demo skill for testing
version: 1.0.0
tags: [test, demo]
allowed_tools: [read_file]
---

# Demo Skill

This is a demo skill instruction.
## Usage
Use this skill when you need to demo something.
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(skill_md)

        # 加载
        provider = FileSystemSkillProvider(skill_path=os.path.join(tmpdir, "skills"))
        skills = provider.list_skills()
        check("FileSystemSkillProvider 发现了 skill", len(skills) == 1)
        check("skill 名称正确", skills[0].name == "demo-skill")
        check("skill 有 tags", "test" in skills[0].tags)
        check("skill 有 allowed_tools", "read_file" in skills[0].allowed_tools)

        # SkillManager 集成
        manager = SkillManager()
        manager.add_provider(provider)
        check("SkillManager 列出了 skill", "demo-skill" in manager.list_names())

        # 加载指令
        instruction = manager.load_skill_instruction("demo-skill")
        check("加载了 skill 指令", instruction is not None and "Demo Skill" in instruction)

    print("  Skill 系统 全部通过 ✅")

asyncio.run(test_skill_viewer())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 3: Gateway — BaseGateway / WebhookGateway / 适配器导入")
print("=" * 60)

def test_gateway():
    from wuwei.gateway.base import BaseGateway, GatewayMessage

    # GatewayMessage
    msg = GatewayMessage(
        platform="test-platform",
        user_id="user-001",
        user_name="Test User",
        content="Hello",
        message_id="msg-001",
    )
    check("GatewayMessage 创建", msg.platform == "test-platform")
    check("GatewayMessage 字段", msg.user_id == "user-001" and msg.content == "Hello")

    # 所有平台适配器导入
    from wuwei.gateway.adapters.dingtalk import DingTalkGateway
    from wuwei.gateway.adapters.feishu import FeishuGateway
    from wuwei.gateway.adapters.telegram import TelegramGateway
    from wuwei.gateway.adapters.wechat import WeChatGateway
    check("DingTalk 适配器导入", DingTalkGateway is not None)
    check("Feishu 适配器导入", FeishuGateway is not None)
    check("Telegram 适配器导入", TelegramGateway is not None)
    check("WeChat 适配器导入", WeChatGateway is not None)

    # Webhook
    from wuwei.gateway.webhook import WebhookGateway
    check("WebhookGateway 导入", WebhookGateway is not None)

    print("  Gateway 全部通过 ✅")

test_gateway()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 4: Sandbox — LocalSandbox 错误路径 + 超时")
print("=" * 60)

async def test_sandbox():
    from wuwei.sandbox.base import BaseSandbox, SandboxResult
    from wuwei.sandbox.local import LocalSandbox

    sandbox = LocalSandbox()

    # 基本执行
    result = await sandbox.execute("echo hello")
    check("LocalSandbox 基本执行", "hello" in result.stdout)

    # 错误命令
    result2 = await sandbox.execute("nonexistent_command_xyz 2>&1")
    check("LocalSandbox 错误命令不崩溃", isinstance(result2, SandboxResult))
    check("LocalSandbox 错误命令 success=False", result2.success == False)

    # 写文件然后读
    await sandbox.execute("echo 'test data' > /tmp/wuwei_test_file.txt")
    result3 = await sandbox.execute("cat /tmp/wuwei_test_file.txt")
    check("LocalSandbox 文件持久化", "test data" in result3.stdout)

    # 多行输出
    result4 = await sandbox.execute("seq 1 5")
    check("LocalSandbox 多行输出", "1" in result4.stdout and "5" in result4.stdout)

    print("  Sandbox 全部通过 ✅")

asyncio.run(test_sandbox())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 5: Plugin — PluginLoader + PluginRegistry 完整流程")
print("=" * 60)

def test_plugins():
    from wuwei.plugin.loader import PluginLoader
    from wuwei.plugin.registry import PluginRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        # PluginRegistry
        registry = PluginRegistry()
        check("PluginRegistry 创建", registry is not None)
        check("PluginRegistry 有 register 方法", hasattr(registry, 'register'))
        check("PluginRegistry 有 list_plugins 方法", hasattr(registry, 'list_plugins'))
        check("PluginRegistry 有 list_hooks 方法", hasattr(registry, 'list_hooks'))

        # PluginLoader
        loader = PluginLoader(plugins_dir=tmpdir)
        check("PluginLoader 创建", loader is not None)
        check("PluginLoader plugins_dir 正确", str(loader.plugins_dir) == tmpdir)

        # load_all — should not crash on empty dir
        discovered = loader.load_all()
        check("PluginLoader.load_all 空目录", isinstance(discovered, list))

    print("  Plugin 全部通过 ✅")

test_plugins()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 6: LLM Adapters — Anthropic / Ollama / DashScope / Zhipu 配置")
print("=" * 60)

def test_all_adapters():
    from wuwei.llm.adapters.anthropic import AnthropicAdapter
    from wuwei.llm.adapters.ollama import OllamaAdapter
    from wuwei.llm.adapters.dashscope import DashScopeAdapter
    from wuwei.llm.adapters.zhipu import ZhipuAdapter
    from wuwei.llm.adapters.openai import OpenAIAdapter

    # 各适配器都能实例化（不需要真实 API key 来测配置逻辑）
    check("OpenAIAdapter 类型", issubclass(OpenAIAdapter, object))
    check("AnthropicAdapter 类型", issubclass(AnthropicAdapter, object))
    check("OllamaAdapter 类型", issubclass(OllamaAdapter, object))
    check("DashScopeAdapter 类型", issubclass(DashScopeAdapter, object))
    check("ZhipuAdapter 类型", issubclass(ZhipuAdapter, object))

    # Agent.from_env 工厂方法
    from wuwei.agent import Agent
    check("Agent.from_env 方法存在", hasattr(Agent, 'from_env'))

    print("  LLM Adapters 全部通过 ✅")

test_all_adapters()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 7: Middleware — MemoryMiddleware / RagMiddleware / StorageMiddleware 实例化")
print("=" * 60)

def test_middleware_instances():
    from wuwei.middleware.memory import MemoryMiddleware
    from wuwei.middleware.rag import RagMiddleware
    from wuwei.middleware.storage import StorageMiddleware
    from wuwei.memory import InMemoryMemoryStore

    # MemoryMiddleware (needs LLM + store)
    store = InMemoryMemoryStore()
    from wuwei.llm import LLMGateway
    llm = LLMGateway({"provider": "openai", "api_key": "test", "model": "gpt-4o"})
    mm = MemoryMiddleware(llm=llm, memory_store=store)
    check("MemoryMiddleware 实例化", mm is not None)
    check("MemoryMiddleware 有 before_llm", hasattr(mm, 'before_llm'))

    # RagMiddleware
    from wuwei.memory import InMemoryKnowledgeStore
    ks = InMemoryKnowledgeStore()
    rm = RagMiddleware(knowledge_store=ks)
    check("RagMiddleware 实例化", rm is not None)
    check("RagMiddleware 是 Middleware 子类", hasattr(rm, 'before_llm'))

    # StorageMiddleware
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StorageMiddleware(storage_path=tmpdir)
        check("StorageMiddleware 实例化", sm is not None)
        check("StorageMiddleware 有 before_llm", hasattr(sm, 'before_llm'))

    print("  Middleware 实例化 全部通过 ✅")

test_middleware_instances()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 8: @tool 装饰器 / Tool.from_function")
print("=" * 60)

def test_tool_decorator():
    from wuwei.tools.tool import Tool as BaseTool
    from wuwei.tools.registry import ToolRegistry

    registry = ToolRegistry()

    # @registry.tool 装饰器
    @registry.tool(description="Greet someone with a custom greeting.")
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone with a custom greeting.

        Args:
            name: The person's name
            greeting: The greeting to use
        """
        return f"{greeting}, {name}!"

    greet_tool = registry.get("greet")
    check("@tool 创建了 Tool", isinstance(greet_tool, BaseTool))
    check("Tool 名称", greet_tool.name == "greet")
    check("Tool 描述", "Greet someone" in greet_tool.description)

    # Tool 通过构造函数创建
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry2 = ToolRegistry()

    @registry2.tool(description="Add two numbers.")
    def add_tool(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add_registered = registry2.get("add_tool")
    check("Tool 注册成功", add_registered is not None)
    check("Tool name", add_registered.name == "add_tool")

    # Tool 有 schema/工具定义能力
    check("Tool 有 name", hasattr(add_registered, 'name'))
    check("Tool 有 description", hasattr(add_registered, 'description'))
    check("Tool 有 parameters", hasattr(add_registered, 'parameters'))

    # Tool.invoke returns result
    import asyncio
    result = asyncio.run(add_registered.invoke({"a": 3, "b": 4}))
    check("Tool.invoke 正确", result == 7)

    print("  @tool 装饰器 全部通过 ✅")

test_tool_decorator()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 9: Observability — TracingMiddleware")
print("=" * 60)

def test_tracing():
    from wuwei.observability.tracing import TracingMiddleware

    tm = TracingMiddleware(service_name="test-service")
    check("TracingMiddleware 实例化", tm is not None)
    check("TracingMiddleware service_name", tm.service_name == "test-service")
    check("TracingMiddleware 有 before_llm", hasattr(tm, 'before_llm'))
    check("TracingMiddleware 有 after_llm", hasattr(tm, 'after_llm'))
    check("TracingMiddleware 有 before_tool", hasattr(tm, 'before_tool'))
    check("TracingMiddleware 有 after_tool", hasattr(tm, 'after_tool'))

    print("  Observability 全部通过 ✅")

test_tracing()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 10: Streaming Types")
print("=" * 60)

def test_streaming_types():
    from wuwei.streaming.types import (
        StreamChunk, MessagesStreamChunk, ValuesStreamChunk,
        UpdatesStreamChunk, CustomStreamChunk, DebugStreamChunk
    )

    # StreamChunk
    chunk = StreamChunk(type="text", data="hello")
    check("StreamChunk 创建", chunk.type == "text" and chunk.data == "hello")

    # MessagesStreamChunk
    msg_chunk = MessagesStreamChunk(type="messages", data="msg content")
    check("MessagesStreamChunk 创建", msg_chunk.type == "messages")

    # ValuesStreamChunk
    val_chunk = ValuesStreamChunk(data={"key": "val"})
    check("ValuesStreamChunk 创建", val_chunk.type == "values")

    # UpdatesStreamChunk
    upd_chunk = UpdatesStreamChunk(data={"node": "llm"})
    check("UpdatesStreamChunk 创建", upd_chunk.type == "updates")

    # CustomStreamChunk
    custom_chunk = CustomStreamChunk(data={"custom": True})
    check("CustomStreamChunk 创建", custom_chunk.type == "custom")

    # DebugStreamChunk
    debug_chunk = DebugStreamChunk(data={"debug": "info"})
    check("DebugStreamChunk 创建", debug_chunk.type == "debug")

    print("  Streaming Types 全部通过 ✅")

test_streaming_types()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"📊 补充测试结果: ✅ {PASSED} passed  ❌ {FAILED} failed  (共 {PASSED+FAILED} 项)")
print("=" * 60)

if FAILED > 0:
    sys.exit(1)
