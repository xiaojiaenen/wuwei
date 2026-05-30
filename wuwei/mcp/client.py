"""MCP 客户端"""

from abc import ABC, abstractmethod
import asyncio
import json
import os
from typing import Any
from wuwei.mcp.config import MCPServerConfig


class BaseMCPClient(ABC):
    """MCP 客户端基类"""

    @abstractmethod
    async def connect(self):
        """连接到 MCP 服务器"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict]:
        """列出可用工具"""
        ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具"""
        ...


class StdioMCPClient(BaseMCPClient):
    """Stdio 传输的 MCP 客户端"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process = None
        self._reader = None
        self._writer = None
        self._request_id = 0

    async def connect(self):
        """启动 MCP 服务器子进程"""
        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.config.env},
        )
        self._reader = self.process.stdout
        self._writer = self.process.stdin

    async def disconnect(self):
        """关闭子进程"""
        if self.process:
            self.process.terminate()
            await self.process.wait()

    async def list_tools(self) -> list[dict]:
        """列出工具"""
        response = await self._send_request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具"""
        response = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        return response

    async def _send_request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        message = json.dumps(request)
        self._writer.write(f"{message}\n".encode())
        await self._writer.drain()

        response = await self._receive()
        if "error" in response:
            raise Exception(f"MCP 错误: {response['error']}")
        return response.get("result", {})

    async def _receive(self) -> dict:
        """接收 JSON-RPC 响应"""
        line = await asyncio.wait_for(
            self._reader.readline(),
            timeout=self.config.timeout,
        )
        return json.loads(line.decode())


class HTTPMCPClient(BaseMCPClient):
    """HTTP/SSE 传输的 MCP 客户端"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._client = None
        self._request_id = 0

    async def connect(self):
        """建立 HTTP 连接"""
        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError(
                "使用 HTTP MCP 客户端需要安装 httpx 包：\n"
                "pip install httpx"
            )

        self._client = AsyncClient(
            base_url=self.config.url,
            headers=self.config.headers,
            timeout=self.config.timeout,
        )

    async def disconnect(self):
        """关闭连接"""
        if self._client:
            await self._client.aclose()

    async def list_tools(self) -> list[dict]:
        """列出工具"""
        response = await self._send_request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具"""
        response = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        return response

    async def _send_request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        resp = await self._client.post("/mcp", json=request)
        data = resp.json()

        if "error" in data:
            raise Exception(f"MCP 错误: {data['error']}")
        return data.get("result", {})
