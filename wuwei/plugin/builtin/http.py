"""HTTP 请求工具插件

错误契约：连接失败/超时等异常直接抛出（由执行器包装为 ``ok:false`` 的
结构化错误结果，模型能明确看到失败）；HTTP 4xx/5xx 的响应体本身是合法
结果内容，按原样返回并携带状态码，不静默截断——超长内容会标注截断字数。
"""

from __future__ import annotations

from wuwei.plugin import PluginContext

_MAX_BODY_CHARS = 5000


def _format_response(status_code: int, text: str) -> str:
    if len(text) > _MAX_BODY_CHARS:
        return (
            f"状态码: {status_code}\n"
            f"内容（已截断，原文 {len(text)} 字符）: {text[:_MAX_BODY_CHARS]}"
        )
    return f"状态码: {status_code}\n内容: {text}"


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
        import json

        from httpx import AsyncClient

        headers_dict = json.loads(headers) if headers else {}

        # 失败不吞：异常交给执行器转成结构化错误（status=error），
        # 而不是把"请求失败: ..."当成功内容喂给模型。
        async with AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers_dict)
            return _format_response(resp.status_code, resp.text)

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
        import json

        from httpx import AsyncClient

        headers_dict = json.loads(headers) if headers else {}
        data_dict = json.loads(data) if data else {}

        async with AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=data_dict, headers=headers_dict)
            return _format_response(resp.status_code, resp.text)
