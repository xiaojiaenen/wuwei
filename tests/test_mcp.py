"""MCP 模块测试

不依赖真实外部 MCP server：
- client 层用官方 SDK 的内存流（create_client_server_memory_streams +
  低层 Server）做端到端握手/发现/调用测试
- session / middleware 层用 Fake 客户端（monkeypatch 注入）
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager

import pytest

from wuwei.core.message import HumanMessage, SystemMessage
from wuwei.graph.state import State
from wuwei.middleware.base import MiddlewareContext
from wuwei.mcp.client import (
    BaseMCPClient,
    MCPClientError,
    SDKSessionMCPClient,
    StdioMCPClient,
)
from wuwei.mcp.config import MCPConfig, MCPServerConfig
from wuwei.mcp.session import MCPSessionManager
from wuwei.mcp.tool_search import ToolSearchIndex, build_tool_search_tool
from wuwei.mcp.tools import MCPToolAdapter
from wuwei.plugin import PluginContext
from wuwei.plugin.builtin.mcp import (
    MCPToolDiscoveryMiddleware,
    build_server_overview,
)
from wuwei.tools.registry import ToolRegistry
from wuwei.tools.tool import Tool, ToolParameters


# ──────────────────────────── 工具/桩 ────────────────────────────


def make_tool(name: str, description: str = "test tool") -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters=ToolParameters(),
        handler=lambda **kw: None,
    )


class FakeMCPClient(BaseMCPClient):
    """假客户端：返回预置的工具清单"""

    def __init__(self, config: MCPServerConfig, tools: list[dict] | None = None):
        self.config = config
        self.tools = tools or []
        self.connected = False
        self.disconnected = False
        self.list_tools_calls = 0

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def list_tools(self) -> list[dict]:
        self.list_tools_calls += 1
        return self.tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        return {"content": [{"type": "text", "text": f"called {name}"}], "isError": False}


class FailingMCPClient(FakeMCPClient):
    """连接失败的假客户端"""

    async def connect(self):
        raise RuntimeError("simulated connection failure")


READ_FILE_TOOL = {
    "name": "read_file",
    "description": "读取文件内容",
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

READ_ONLY_TOOL = {
    "name": "list_dir",
    "description": "列出目录",
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {"readOnlyHint": True},
}


def patch_session_clients(monkeypatch, factory):
    """把 session 模块的客户端构造替换为 factory(config)"""

    class _Factory:
        def __call__(self, config):
            return factory(config)

    f = _Factory()
    monkeypatch.setattr("wuwei.mcp.session.StdioMCPClient", f)
    monkeypatch.setattr("wuwei.mcp.session.HTTPMCPClient", f)


def make_config(**servers: dict) -> MCPConfig:
    config = MCPConfig()
    for name, kwargs in servers.items():
        config.add_server(MCPServerConfig(name=name, **kwargs))
    return config


# ──────────────────────────── 配置 ────────────────────────────


class TestMCPServerConfig:
    def test_stdio_config(self):
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        assert config.transport == "stdio"
        assert config.command == "npx"
        assert config.enabled is True
        assert config.timeout == 60.0
        assert config.env == {}

    def test_http_config(self):
        config = MCPServerConfig(
            name="remote",
            transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer xxx"},
        )
        assert config.transport == "http"
        assert config.url == "https://example.com/mcp"
        assert config.headers["Authorization"] == "Bearer xxx"

    def test_sse_config_kept_for_compat(self):
        """sse transport 保持向后兼容（实现上映射到 streamable http）"""
        config = MCPServerConfig(name="legacy", transport="sse", url="https://e.com/sse")
        assert config.transport == "sse"


class TestMCPConfig:
    def test_load_dual_scope_merge(self, monkeypatch, tmp_path):
        """项目级 + 用户级双作用域合并（沿用原语义：user 后加载、覆盖同名项）"""
        project_file = tmp_path / "project.mcp.json"
        user_file = tmp_path / "user.mcp.json"
        project_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "proj": {"command": "proj-cmd", "transport": "stdio"},
                    }
                }
            )
        )
        user_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "user": {"command": "user-cmd", "transport": "stdio"},
                        "proj": {"command": "user-override", "transport": "stdio"},
                    }
                }
            )
        )
        monkeypatch.setattr(
            MCPConfig,
            "_get_config_path",
            classmethod(lambda cls, scope: str(project_file if scope == "project" else user_file)),
        )

        config = MCPConfig.load()
        assert set(config.mcp_servers) == {"proj", "user"}
        # 兼容原实现：作用域顺序 ["project", "user"]，后加载的 user 覆盖同名配置
        assert config.mcp_servers["proj"].command == "user-override"
        assert config.mcp_servers["user"].command == "user-cmd"

    def test_add_remove_and_enabled(self):
        config = MCPConfig()
        config.add_server(MCPServerConfig(name="enabled", command="test", enabled=True))
        config.add_server(MCPServerConfig(name="disabled", command="test", enabled=False))
        assert "enabled" in config.mcp_servers
        assert [s.name for s in config.get_enabled_servers()] == ["enabled"]

        config.remove_server("enabled")
        assert "enabled" not in config.mcp_servers
        assert config.get_enabled_servers() == []

    def test_to_dict_roundtrip(self):
        config = MCPConfig()
        config.add_server(
            MCPServerConfig(
                name="test",
                transport="http",
                url="https://example.com/mcp",
                headers={"k": "v"},
            )
        )
        d = config.to_dict()
        assert d["mcpServers"]["test"]["transport"] == "http"
        restored = MCPServerConfig(**d["mcpServers"]["test"])
        assert restored.url == "https://example.com/mcp"


class TestStdioEnvWhitelist:
    def test_env_whitelist_and_override(self, monkeypatch):
        fake_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "USER": "u",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "TMPDIR": "/tmp",
            "SHELL": "/bin/zsh",
            "OPENAI_API_KEY": "sk-secret",
            "AWS_SECRET_ACCESS_KEY": "leak-me",
        }
        monkeypatch.setattr(os, "environ", fake_env)
        config = MCPServerConfig(
            name="s",
            command="cmd",
            env={"CUSTOM_VAR": "1", "PATH": "/custom"},
        )
        env = StdioMCPClient.build_env(config)
        # 白名单全部保留
        assert env["PATH"] == "/custom"  # config.env 可覆盖白名单
        assert env["HOME"] == "/home/u"
        assert env["USER"] == "u"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["LC_ALL"] == "en_US.UTF-8"
        assert env["TMPDIR"] == "/tmp"
        assert env["SHELL"] == "/bin/zsh"
        assert env["CUSTOM_VAR"] == "1"
        # 敏感变量不泄漏
        assert "OPENAI_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env


# ──────────────────────────── SDK 客户端（内存端到端） ────────────────────────────


def build_in_memory_server():
    """用 SDK 低层 Server 构建内存 MCP 服务器"""
    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server("memory-test")

    async def list_tools_handler(ctx, params):
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="add",
                    description="Add two numbers.",
                    inputSchema={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                ),
                types.Tool(
                    name="status",
                    description="Read system status.",
                    inputSchema={"type": "object", "properties": {}},
                    annotations=types.ToolAnnotations(readOnlyHint=True),
                ),
            ]
        )

    async def call_tool_handler(ctx, params: types.CallToolRequestParams):
        if params.name == "add":
            text = str(params.arguments["a"] + params.arguments["b"])
        else:
            text = "ok"
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    server.add_request_handler("tools/list", types.PaginatedRequestParams, list_tools_handler)
    server.add_request_handler("tools/call", types.CallToolRequestParams, call_tool_handler)
    return server


class InMemoryMCPClient(SDKSessionMCPClient):
    """走 SDK ClientSession 真实路径、但用内存流的客户端（测试专用）"""

    def __init__(self, config: MCPServerConfig, server):
        super().__init__(config)
        self._server = server

    @asynccontextmanager
    async def _open_streams(self):
        from mcp.shared.memory import create_client_server_memory_streams

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            task = asyncio.create_task(
                self._server.run(
                    server_streams[0],
                    server_streams[1],
                    self._server.create_initialization_options(),
                )
            )
            try:
                yield client_streams
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, BaseExceptionGroup):
                    pass


class TestSDKSessionClient:
    """基于官方 SDK ClientSession 的端到端测试（内存流，无外部进程）"""

    @pytest.fixture
    def client(self):
        config = MCPServerConfig(name="mem", transport="http", url="memory://test")
        return InMemoryMCPClient(config, build_in_memory_server())

    async def test_connect_initialize_list_tools(self, client):
        await client.connect()
        try:
            assert client.session is not None
            # initialize 握手已完成
            assert client.session.protocol_version
            tools = await client.list_tools()
            assert [t["name"] for t in tools] == ["add", "status"]
            # 协议层 camelCase 键，供 MCPToolAdapter 使用
            assert tools[0]["description"] == "Add two numbers."
            assert tools[0]["inputSchema"]["required"] == ["a", "b"]
            assert "annotations" not in tools[0] or tools[0].get("annotations") in (None, {})
        finally:
            await client.disconnect()

    async def test_call_tool_result_shape(self, client):
        await client.connect()
        try:
            result = await client.call_tool("add", {"a": 2, "b": 3})
            assert result["content"] == [{"type": "text", "text": "5"}]
            assert result["isError"] is False
        finally:
            await client.disconnect()

    async def test_call_without_connect_raises(self, client):
        with pytest.raises(MCPClientError, match="未连接"):
            await client.call_tool("add", {})
        with pytest.raises(MCPClientError, match="未连接"):
            await client.list_tools()

    async def test_disconnect_idempotent(self, client):
        await client.disconnect()  # 未连接时断开不报错
        await client.connect()
        await client.disconnect()
        await client.disconnect()
        assert client.session is None

    async def test_adapter_maps_read_only_hint(self, client):
        """SDK 返回的 readOnlyHint 注解正确映射到 execution.side_effect"""
        await client.connect()
        try:
            adapter = MCPToolAdapter(client, "mem")
            tools = await adapter.discover_tools()
            by_name = {t.name: t for t in tools}
            assert set(by_name) == {"mcp__mem__add", "mcp__mem__status"}
            assert by_name["mcp__mem__add"].execution.side_effect is True
            assert by_name["mcp__mem__add"].execution.requires_approval is True
            assert by_name["mcp__mem__status"].execution.side_effect is False
            assert by_name["mcp__mem__status"].execution.requires_approval is False
            assert by_name["mcp__mem__add"].parameters.required == ["a", "b"]
        finally:
            await client.disconnect()


# ──────────────────────────── 会话管理（失败隔离） ────────────────────────────


class TestMCPSessionManager:
    def _make_manager(self):
        """两个好 server + 一个坏 server"""
        config = make_config(
            good1=dict(transport="stdio", command="a"),
            bad=dict(transport="stdio", command="b"),
            good2=dict(transport="http", url="https://example.com/mcp"),
        )

        def factory(server_config):
            if server_config.name == "bad":
                return FailingMCPClient(server_config)
            tools = (
                [READ_FILE_TOOL, READ_ONLY_TOOL]
                if server_config.name == "good1"
                else [READ_FILE_TOOL]
            )
            return FakeMCPClient(server_config, tools=tools)

        return config, factory

    async def test_connect_all_isolates_failures(self, monkeypatch):
        """单 server 连接失败不炸 connect_all，失败记录到 failures"""
        config, factory = self._make_manager()
        patch_session_clients(monkeypatch, factory)
        manager = MCPSessionManager(config)

        await manager.connect_all()  # 不应抛异常

        assert set(manager.list_servers()) == {"good1", "good2"}
        assert set(manager.failures) == {"bad"}
        assert isinstance(manager.failures["bad"], RuntimeError)
        assert manager.get_client("good1").connected

        tools = manager.get_all_tools()
        assert {t.name for t in tools} == {
            "mcp__good1__read_file",
            "mcp__good1__list_dir",
            "mcp__good2__read_file",
        }
        assert [t.name for t in manager.get_tools_by_server("good1")] == [
            "mcp__good1__read_file",
            "mcp__good1__list_dir",
        ]

    async def test_discover_only_once(self, monkeypatch):
        """工具只发现一次：重复 connect_all 不重新 discover 已连接 server"""
        config, factory = self._make_manager()
        patch_session_clients(monkeypatch, factory)
        manager = MCPSessionManager(config)

        await manager.connect_all()
        await manager.connect_all()  # 第二次应跳过已连接的

        assert manager.get_client("good1").list_tools_calls == 1
        assert manager.get_client("good2").list_tools_calls == 1

    async def test_disconnect_all(self, monkeypatch):
        config, factory = self._make_manager()
        patch_session_clients(monkeypatch, factory)
        manager = MCPSessionManager(config)
        await manager.connect_all()
        await manager.disconnect_all()
        assert manager.list_servers() == []
        assert manager.get_all_tools() == []

    async def test_all_disabled_no_op(self):
        config = MCPConfig()
        config.add_server(MCPServerConfig(name="off", command="x", enabled=False))
        manager = MCPSessionManager(config)
        await manager.connect_all()
        assert manager.list_servers() == []
        assert manager.failures == {}

    def test_overview_includes_failures(self, monkeypatch):
        config, factory = self._make_manager()
        patch_session_clients(monkeypatch, factory)
        manager = MCPSessionManager(config)

        # 同步侧直接构造状态用于概览文本测试
        manager.failures = {"bad": RuntimeError("simulated connection failure")}
        manager._tools = {
            "good1": [make_tool("mcp__good1__read_file", "读取文件"), make_tool("mcp__good1__list_dir", "列出目录")]
        }
        manager._clients = {"good1": object()}
        overview = build_server_overview(manager)
        assert "- good1（2 个工具）: mcp__good1__read_file, mcp__good1__list_dir" in overview
        assert "- bad（连接失败）: simulated connection failure" in overview

    def test_overview_truncates_long_tool_lists(self):
        manager = MCPSessionManager(make_config())
        manager._clients = {"srv": object()}
        manager._tools = {
            "srv": [make_tool(f"mcp__srv__tool_{i:02d}") for i in range(15)]
        }
        overview = build_server_overview(manager)
        assert "- srv（15 个工具）:" in overview
        assert "mcp__srv__tool_09" in overview
        assert "mcp__srv__tool_10" not in overview  # 超出 10 个被省略
        assert "等 15 个工具" in overview


# ──────────────────────────── tool_search ────────────────────────────


class TestToolSearchIndex:
    def _index(self):
        index = ToolSearchIndex()
        index.add_tools(
            [
                make_tool("mcp__fs__read_file", "Read a file from disk and return its contents"),
                make_tool("mcp__fs__write_file", "Write text content to a file on disk"),
                make_tool("mcp__web__search_web", "Search the web for a query"),
                make_tool("mcp__db__query", "Run a SQL query against the database"),
            ],
            server="unused",  # add_tools 的 server 参数逐条写入
        )
        return index

    def test_substring_in_name_scores_highest(self):
        index = ToolSearchIndex()
        index.add(make_tool("mcp__fs__read_file", "Read a file"), "fs")
        index.add(make_tool("mcp__db__query", "Run a read query"), "db")

        results = index.search("read")
        assert results[0]["name"] == "mcp__fs__read_file"
        # 名称子串(+5)+词命中(+2+1)，db 那条只有描述词命中(+2+1)
        assert results[0]["score"] > results[1]["score"]

    def test_token_overlap_matches_without_substring(self):
        index = ToolSearchIndex()
        index.add(make_tool("mcp__web__search_web", "Search the web for a query"), "web")
        results = index.search("web search")
        assert results and results[0]["name"] == "mcp__web__search_web"

    def test_limit_and_no_match(self):
        index = self._index()
        assert len(index.search("file", limit=1)) == 1
        assert index.search("完全无关的查询xyz") == []
        assert index.search("") == []

    def test_get_tool_and_len(self):
        index = ToolSearchIndex()
        tool = make_tool("mcp__a__b")
        index.add(tool, "a")
        assert len(index) == 1
        assert index.get_tool("mcp__a__b") is tool
        assert index.get_tool("missing") is None


class TestBuildToolSearchTool:
    async def test_search_registers_matched_tools(self):
        index = ToolSearchIndex()
        read_tool = make_tool("mcp__fs__read_file", "Read a file from disk")
        write_tool = make_tool("mcp__fs__write_file", "Write a file to disk")
        index.add(read_tool, "fs")
        index.add(write_tool, "fs")

        registry = ToolRegistry()
        registered = []
        def register_fn(t):
            registry.register(t)
            registered.append(t.name)
            return True

        tool = build_tool_search_tool(index, register_fn)
        assert tool.name == "tool_search"

        result = await tool.invoke({"query": "read file"})
        assert result["matches"][0]["name"] == "mcp__fs__read_file"
        assert result["matches"][0]["server"] == "fs"
        assert "mcp__fs__read_file" in result["registered"]
        assert registry.get("mcp__fs__read_file") is read_tool

    async def test_already_registered_not_counted(self):
        index = ToolSearchIndex()
        read_tool = make_tool("mcp__fs__read_file", "Read a file")
        index.add(read_tool, "fs")

        registry = ToolRegistry()
        registry.register(read_tool)
        tool = build_tool_search_tool(index, lambda t: False)
        result = await tool.invoke({"query": "read file"})
        assert result["registered"] == []
        assert result["matches"]  # 匹配仍然返回

    async def test_no_match(self):
        index = ToolSearchIndex()
        index.add(make_tool("mcp__db__query", "Run SQL"), "db")
        tool = build_tool_search_tool(index, lambda t: True)
        result = await tool.invoke({"query": "nothing matches here"})
        assert result["matches"] == []
        assert result["registered"] == []


# ──────────────────────────── 发现中间件（概览注入 + 按需注册） ────────────────────────────


class TestMCPToolDiscoveryMiddleware:
    def _setup(self, eager_tools=None):
        """三个 server：filesystem（2 工具）、math（1 工具）、broken（连接失败）"""
        config = make_config(
            filesystem=dict(transport="stdio", command="fs-server"),
            math=dict(transport="stdio", command="math-server"),
            broken=dict(transport="stdio", command="bad"),
        )
        registry = ToolRegistry()
        plugin_ctx = PluginContext(
            tool_registry=registry,
            mcp_manager=MCPSessionManager(config),
            config={"eager_tools": eager_tools} if eager_tools else {},
        )
        middleware = MCPToolDiscoveryMiddleware(plugin_ctx)
        ctx_state = MiddlewareContext(state=State(messages=[HumanMessage(content="hi")]))
        return middleware, ctx_state, registry, plugin_ctx

    @staticmethod
    def _factory(server_config):
        if server_config.name == "broken":
            return FailingMCPClient(server_config)
        tools = {
            "filesystem": [READ_FILE_TOOL, READ_ONLY_TOOL],
            "math": [
                {
                    "name": "add",
                    "description": "Add two numbers.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                }
            ],
        }
        return FakeMCPClient(server_config, tools=tools.get(server_config.name, []))

    async def test_first_before_llm_injects_overview_and_not_all_tools(self, monkeypatch):
        """首次 before_llm：注入 server 概览、注册 tool_search、不注册全部工具"""
        middleware, mctx, registry, plugin_ctx = self._setup()
        patch_session_clients(monkeypatch, self._factory)

        await middleware.before_llm(mctx)

        # 1) 注入 server 概览 SystemMessage
        first = mctx.state.messages[0]
        assert isinstance(first, SystemMessage)
        assert "filesystem" in first.content
        assert "math" in first.content
        assert "tool_search" in first.content
        # 失败的服务器也在概览中标注
        assert "broken" in first.content and "连接失败" in first.content

        # 2) 未注册全部 MCP 工具（治理上下文膨胀的核心）
        assert registry.get("mcp__filesystem__read_file") is None
        assert registry.get("mcp__filesystem__list_dir") is None
        assert len([t for t in registry.list_tools() if t.name.startswith("mcp__")]) == 0

        # 3) tool_search 已注册
        assert registry.get("tool_search") is not None

        # 4) 第二次调用不再注入（无重复）
        await middleware.before_llm(mctx)
        assert sum(isinstance(m, SystemMessage) for m in mctx.state.messages) == 1

    async def test_eager_tools_wildcard(self, monkeypatch):
        """eager_tools 支持通配符，启动即注册指定工具"""
        middleware, mctx, registry, plugin_ctx = self._setup(
            eager_tools=["mcp__filesystem__*"]
        )
        patch_session_clients(monkeypatch, self._factory)
        await middleware.before_llm(mctx)

        assert registry.get("mcp__filesystem__read_file") is not None
        assert registry.get("mcp__filesystem__list_dir") is not None
        # 通配符只命中 filesystem，math 未注册
        assert registry.get("mcp__math__add") is None

    async def test_tool_search_call_registers_tool(self, monkeypatch):
        """调用 tool_search 工具后，匹配的 MCP 工具被注册且可调用"""
        middleware, mctx, registry, plugin_ctx = self._setup()
        patch_session_clients(monkeypatch, self._factory)
        await middleware.before_llm(mctx)

        search_tool = registry.get("tool_search")
        result = await search_tool.invoke({"query": "read file"})

        assert result["registered"] == ["mcp__filesystem__read_file"]
        registered_tool = registry.get("mcp__filesystem__read_file")
        assert registered_tool is not None
        # 注册后的工具可直接调用（走 FakeMCPClient.call_tool）
        output = await registered_tool.invoke({"path": "/tmp/a.txt"})
        assert output == "called read_file"

    async def test_no_mcp_manager_short_circuit(self):
        """无 mcp_manager 时直接跳过"""
        registry = ToolRegistry()
        plugin_ctx = PluginContext(tool_registry=registry, mcp_manager=None)
        middleware = MCPToolDiscoveryMiddleware(plugin_ctx)
        mctx = MiddlewareContext(state=State(messages=[HumanMessage(content="hi")]))
        await middleware.before_llm(mctx)
        assert len(mctx.state.messages) == 1
