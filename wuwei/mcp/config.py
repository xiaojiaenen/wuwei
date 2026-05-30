"""MCP 配置管理"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
import json
import os


class MCPServerConfig(BaseModel):
    """MCP 服务器配置"""
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"

    # stdio 配置
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # http/sse 配置
    url: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)

    # 通用配置
    timeout: float = 60.0
    enabled: bool = True


class MCPConfig(BaseModel):
    """MCP 配置管理

    支持多作用域配置（项目级/用户级）。
    """
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @classmethod
    def load(cls, scopes: list[str] = None) -> "MCPConfig":
        """从多个作用域加载配置

        Args:
            scopes: 配置作用域列表，默认 ["project", "user"]
        """
        if scopes is None:
            scopes = ["project", "user"]

        config = cls()
        for scope in scopes:
            path = cls._get_config_path(scope)
            if path and os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for name, server in data.get("mcpServers", {}).items():
                    config.mcp_servers[name] = MCPServerConfig(
                        name=name,
                        **server,
                    )
        return config

    @classmethod
    def _get_config_path(cls, scope: str) -> Optional[str]:
        """获取配置文件路径"""
        if scope == "project":
            return ".mcp.json"
        elif scope == "user":
            return os.path.expanduser("~/.wuwei/.mcp.json")
        return None

    def add_server(self, config: MCPServerConfig):
        """添加服务器配置"""
        self.mcp_servers[config.name] = config

    def remove_server(self, name: str):
        """移除服务器配置"""
        if name in self.mcp_servers:
            del self.mcp_servers[name]

    def get_enabled_servers(self) -> list[MCPServerConfig]:
        """获取所有启用的服务器"""
        return [s for s in self.mcp_servers.values() if s.enabled]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "mcpServers": {
                name: server.model_dump()
                for name, server in self.mcp_servers.items()
            }
        }

    def save(self, path: str = None):
        """保存配置"""
        if path is None:
            path = ".mcp.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
