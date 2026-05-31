"""框架完整性测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestCoreModule:
    """核心模块测试"""

    def test_imports(self):
        """测试所有核心模块可以导入"""
        from wuwei.core import (
            Runnable,
            RunnableSequence,
            RunnableConfig,
            BaseMessage,
            AIMessage,
            ToolMessage,
            SystemMessage,
            HumanMessage,
            ToolCall,
            FunctionCall,
            WuweiError,
            ToolError,
            LLMError,
        )

    def test_message_conversion(self):
        """测试消息格式转换"""
        from wuwei.core import HumanMessage, AIMessage, ToolCall, FunctionCall

        # HumanMessage
        msg = HumanMessage(content="hello")
        assert msg.to_openai() == {"role": "user", "content": "hello"}

        # AIMessage with tool calls
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function=FunctionCall(name="search", arguments={"q": "test"}),
                )
            ],
        )
        openai_format = ai_msg.to_openai()
        assert "tool_calls" in openai_format
        assert len(openai_format["tool_calls"]) == 1


class TestToolModule:
    """工具模块测试"""

    def test_tool_from_function(self):
        """测试从函数创建工具"""
        from wuwei.tools.base import Tool, tool

        @tool
        def my_func(x: int, y: str = "default") -> str:
            """测试函数"""
            return f"{x} {y}"

        assert my_func.name == "my_func"
        schema = my_func.to_openai_schema()
        assert "function" in schema
        assert "parameters" in schema["function"]

    def test_tool_registry(self):
        """测试工具注册表"""
        from wuwei.tools.base import Tool
        from wuwei.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tool = Tool(
            name="test",
            description="test tool",
            handler=lambda: None,
        )
        registry.register(tool)
        assert registry.get("test") is not None


class TestGraphModule:
    """状态图模块测试"""

    @pytest.mark.asyncio
    async def test_state_graph(self):
        """测试状态图"""
        from wuwei.graph import State, StateGraph
        from wuwei.graph.graph import END

        async def node(state, config):
            state.messages.append(MagicMock(content="test"))
            return state

        graph = StateGraph(State)
        graph.add_node("test", node)
        graph.add_edge("test", END)
        graph.set_entry_point("test")

        app = graph.compile()
        state = await app.invoke()
        assert len(state.messages) == 1


class TestMiddlewareModule:
    """中间件模块测试"""

    @pytest.mark.asyncio
    async def test_middleware_stack(self):
        """测试中间件栈"""
        from wuwei.middleware import Middleware, MiddlewareStack
        from wuwei.graph.state import State
        from wuwei.middleware.base import MiddlewareContext

        class TestMiddleware(Middleware):
            def __init__(self):
                self.called = False

            async def before_llm(self, ctx):
                self.called = True
                return ctx

        stack = MiddlewareStack()
        mw = TestMiddleware()
        stack.add(mw)

        ctx = MiddlewareContext(state=State())
        await stack.execute_before_llm(ctx)
        assert mw.called


class TestMCPModule:
    """MCP 模块测试"""

    def test_mcp_config(self):
        """测试 MCP 配置"""
        from wuwei.mcp import MCPServerConfig, MCPConfig

        config = MCPConfig()
        config.add_server(
            MCPServerConfig(name="test", command="test")
        )
        assert "test" in config.mcp_servers

    def test_mcp_tool_adapter(self):
        """测试 MCP 工具适配器"""
        from wuwei.mcp.tools import MCPToolAdapter
        from unittest.mock import AsyncMock

        client = AsyncMock()
        client.list_tools.return_value = [
            {"name": "tool1", "description": "test"}
        ]

        adapter = MCPToolAdapter(client, "server")
        tools = adapter._create_handler("tool1")
        assert callable(tools)


class TestSkillModule:
    """技能模块测试"""

    def test_skill_manager(self):
        """测试技能管理器"""
        from wuwei.skill import SkillManager, Skill

        manager = SkillManager()
        skill = Skill(
            name="test",
            description="test skill",
            instruction="test instruction",
        )

        class MockProvider:
            def list_skills(self):
                return [skill]

        manager.add_provider(MockProvider())
        assert len(manager.list_skills()) == 1
        assert manager.list_names() == ["test"]


class TestGatewayModule:
    """网关模块测试"""

    def test_gateway_imports(self):
        """测试网关模块导入"""
        from wuwei.gateway import BaseGateway, GatewayMessage, WebhookGateway
        from wuwei.gateway.adapters import (
            WeChatGateway,
            DingTalkGateway,
            FeishuGateway,
            TelegramGateway,
        )


class TestSandboxModule:
    """沙箱模块测试"""

    @pytest.mark.asyncio
    async def test_local_sandbox(self):
        """测试本地沙箱"""
        from wuwei.sandbox import LocalSandbox

        sandbox = LocalSandbox()
        result = await sandbox.execute("echo hello")
        assert result.success
        assert "hello" in result.stdout


class TestObservabilityModule:
    """可观测性模块测试"""

    def test_tracing_middleware(self):
        """测试追踪中间件"""
        from wuwei.observability import TracingMiddleware

        middleware = TracingMiddleware(service_name="test")
        assert middleware.service_name == "test"


class TestConfigModule:
    """配置模块测试"""

    def test_agent_config(self):
        """测试 Agent 配置"""
        from wuwei.config import AgentConfig

        config = AgentConfig(
            name="test-agent",
            system_prompt="test prompt",
        )
        assert config.name == "test-agent"


class TestMultiAgent:
    """多 Agent 协作测试"""

    def test_swarm_imports(self):
        """测试 MultiAgent 导入"""
        from wuwei.agent import MultiAgentGraph, TeamMember, HandoffMiddleware
        from wuwei.agent.async_sub_agent import AsyncSubAgent, AsyncSubAgentMiddleware
        assert MultiAgentGraph is not None
        assert AsyncSubAgent is not None

    def test_team_member(self):
        """测试 TeamMember 数据模型"""
        from wuwei.agent.multi_agent import TeamMember
        member = TeamMember(name="test", agent=None, role="tester")
        assert member.name == "test"
        assert member.role == "tester"


class TestBuiltinTools:
    """内置工具测试"""

    def test_json_tools(self):
        """测试 JSON 工具"""
        from wuwei.tools.builtin.json_tools import JSON_TOOLS
        assert len(JSON_TOOLS) == 2

    def test_http_tools(self):
        """测试 HTTP 工具"""
        from wuwei.tools.builtin.http_tools import HTTP_TOOLS
        assert len(HTTP_TOOLS) == 2

    def test_text_tools(self):
        """测试文本工具"""
        from wuwei.tools.builtin.text_tools import TEXT_TOOLS
        assert len(TEXT_TOOLS) == 6


class TestEndToEnd:
    """端到端测试"""

    def test_full_import(self):
        """测试完整导入"""
        from wuwei import (
            Agent,
            LLMGateway,
            Tool,
            ToolRegistry,
            AgentSession,
        )
        from wuwei.core import Runnable, BaseMessage
        from wuwei.graph import StateGraph, State
        from wuwei.middleware import MiddlewareStack
        from wuwei.mcp import MCPSessionManager
        from wuwei.skill import SkillManager
        from wuwei.gateway import WebhookGateway
        from wuwei.sandbox import LocalSandbox
        from wuwei.observability import TracingMiddleware
