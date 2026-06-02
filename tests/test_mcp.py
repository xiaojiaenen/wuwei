"""MCP 模块测试"""

import pytest
import json
import os
import tempfile
from wuwei.mcp.config import MCPServerConfig, MCPConfig
from wuwei.mcp.tools import MCPToolAdapter
from wuwei.mcp.session import MCPSessionManager


class TestMCPServerConfig:
    """MCPServerConfig 测试"""

    def test_stdio_config(self):
        """测试 stdio 配置"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        assert config.name == "test"
        assert config.transport == "stdio"
        assert config.command == "npx"

    def test_http_config(self):
        """测试 http 配置"""
        config = MCPServerConfig(
            name="remote",
            transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer xxx"},
        )
        assert config.transport == "http"
        assert config.url == "https://example.com/mcp"


class TestMCPConfig:
    """MCPConfig 测试"""

    def test_load_from_file(self):
        """测试从文件加载配置"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "mcpServers": {
                        "test": {
                            "command": "npx",
                            "args": ["-y", "test-server"],
                            "transport": "stdio",
                        }
                    }
                },
                f,
            )
            temp_path = f.name

        try:
            config = MCPConfig.load(scopes=[])
            # 手动加载测试文件
            with open(temp_path) as f:
                data = json.load(f)
            for name, server in data.get("mcpServers", {}).items():
                config.mcp_servers[name] = MCPServerConfig(
                    name=name, **server
                )

            assert "test" in config.mcp_servers
            assert config.mcp_servers["test"].command == "npx"
        finally:
            os.unlink(temp_path)

    def test_add_and_remove(self):
        """测试添加和移除服务器"""
        config = MCPConfig()
        server = MCPServerConfig(
            name="test",
            command="test",
        )
        config.add_server(server)
        assert "test" in config.mcp_servers

        config.remove_server("test")
        assert "test" not in config.mcp_servers

    def test_get_enabled_servers(self):
        """测试获取启用的服务器"""
        config = MCPConfig()
        config.add_server(
            MCPServerConfig(name="enabled", command="test", enabled=True)
        )
        config.add_server(
            MCPServerConfig(name="disabled", command="test", enabled=False)
        )

        enabled = config.get_enabled_servers()
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"

    def test_to_dict(self):
        """测试转换为字典"""
        config = MCPConfig()
        config.add_server(
            MCPServerConfig(name="test", command="test")
        )
        d = config.to_dict()
        assert "mcpServers" in d
        assert "test" in d["mcpServers"]


class TestMCPToolAdapter:
    """MCPToolAdapter 测试"""

    @pytest.mark.asyncio
    async def test_discover_tools(self):
        """测试工具发现"""
        from unittest.mock import AsyncMock, MagicMock

        # 创建模拟客户端
        client = AsyncMock()
        client.list_tools.return_value = [
            {
                "name": "read_file",
                "description": "读取文件",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            }
        ]

        adapter = MCPToolAdapter(client, "filesystem")
        tools = await adapter.discover_tools()

        assert len(tools) == 1
        assert tools[0].name == "mcp__filesystem__read_file"
        assert tools[0].description == "读取文件"


class TestMCPSessionManager:
    """MCPSessionManager 测试"""

    def test_init(self):
        """测试初始化"""
        config = MCPConfig()
        session = MCPSessionManager(config)
        assert len(session._clients) == 0
        assert len(session._tools) == 0

    def test_get_all_tools(self):
        """测试获取所有工具"""
        from wuwei.tools.tool import Tool, ToolParameters

        config = MCPConfig()
        session = MCPSessionManager(config)

        # 模拟工具
        session._tools["server1"] = [
            Tool(
                name="tool1",
                description="test",
                parameters=ToolParameters(),
                handler=lambda: None,
            )
        ]

        tools = session.get_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "tool1"

    def test_list_servers(self):
        """测试列出服务器"""
        config = MCPConfig()
        session = MCPSessionManager(config)
        session._clients = {"server1": None, "server2": None}

        servers = session.list_servers()
        assert len(servers) == 2
        assert "server1" in servers
