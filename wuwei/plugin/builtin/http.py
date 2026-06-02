"""HTTP 请求工具插件"""

from __future__ import annotations

from wuwei.plugin import PluginContext


def setup(ctx: PluginContext) -> None:
    @ctx.tool_registry.tool(
        name="http_get",
        description="发送 HTTP GET 请求",
        display_name="HTTP GET",
    )
    async def http_get(url: str, headers: str = "{}") -> str:
        """发送 HTTP GET 请求

        Args:
            url: 请求 URL
            headers: JSON 格式的请求头
        """
        try:
            import json
            from httpx import AsyncClient

            headers_dict = json.loads(headers) if headers else {}

            async with AsyncClient(timeout=30) as client:
                resp = await client.get(url, headers=headers_dict)
                return f"状态码: {resp.status_code}\n内容: {resp.text[:5000]}"
        except ImportError:
            return "需要安装 httpx: pip install httpx"
        except Exception as e:
            return f"请求失败: {e}"

    @ctx.tool_registry.tool(
        name="http_post",
        description="发送 HTTP POST 请求",
        display_name="HTTP POST",
    )
    async def http_post(url: str, data: str = "{}", headers: str = "{}") -> str:
        """发送 HTTP POST 请求

        Args:
            url: 请求 URL
            data: JSON 格式的请求体
            headers: JSON 格式的请求头
        """
        try:
            import json
            from httpx import AsyncClient

            headers_dict = json.loads(headers) if headers else {}
            data_dict = json.loads(data) if data else {}

            async with AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=data_dict, headers=headers_dict)
                return f"状态码: {resp.status_code}\n内容: {resp.text[:5000]}"
        except ImportError:
            return "需要安装 httpx: pip install httpx"
        except Exception as e:
            return f"请求失败: {e}"
