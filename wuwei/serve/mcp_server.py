"""MCP 出口：把 wuwei Agent 暴露为 MCP Server。

基于官方 ``mcp`` SDK（2.x：``mcp.server.mcpserver.MCPServer``，即 1.x 的
FastMCP 更名后的形态）：

- ``build_mcp_server(agent, ...)`` 返回进程内可测的 MCPServer 对象；
- ``serve_mcp(agent, ...)`` 以 stdio 传输阻塞运行（供宿主进程接入）；
- ``python -m wuwei.serve.mcp_server --stdio`` 命令行入口。

暴露的工具：

- ``run_task(task)``：运行 agent，返回 ``{content, usage, status}``；
- ``get_status()``（可选）：返回 agent 元信息。
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

__all__ = ["build_mcp_server", "serve_mcp", "main"]


def _result_field(result: Any, key: str, default: Any = None) -> Any:
    """从 agent.run 的返回值中按鸭子类型取字段（支持对象与 dict）。"""
    if result is None:
        return default
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _result_to_payload(result: Any) -> dict[str, Any]:
    """把 agent.run 的结果归一化为 {content, usage, status, ...}。"""
    content = _result_field(result, "content")
    if content is None:
        content = "" if result is None else str(result)
    usage = _result_field(result, "usage") or {}
    status = _result_field(result, "status") or "completed"
    payload: dict[str, Any] = {
        "content": content,
        "usage": dict(usage) if isinstance(usage, dict) else dict(usage or {}),
        "status": status,
    }
    latency_ms = _result_field(result, "latency_ms")
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    for key in (
        "interrupt",
        "termination_reason",
        "usage_available",
        "output",
        "content_parts",
        "metadata",
    ):
        value = _result_field(result, key)
        if value is not None:
            payload[key] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    from pydantic_core import to_jsonable_python

    return to_jsonable_python(payload)


def build_mcp_server(
    agent: Any,
    *,
    name: str = "wuwei-agent",
    version: str | None = None,
    instructions: str | None = None,
    include_status_tool: bool = True,
):
    """把 agent 包装为 MCP server（进程内对象，可单测）。

    Args:
        agent: 任意具备 ``async run(user_input)`` 的 agent（含 wuwei Agent）。
        name: MCP server 名称。
        version: server 版本；缺省尝试读取 agent 的 ``version`` 属性。
        instructions: MCP instructions；缺省给出工具说明。
        include_status_tool: 是否额外暴露 ``get_status`` 工具。
    """
    from mcp.server.mcpserver import MCPServer

    server_version = version or getattr(agent, "version", "") or ""
    server = MCPServer(
        name=name,
        version=server_version,
        instructions=instructions or "wuwei agent exposed over MCP. Use run_task to run a task.",
    )

    async def run_task(task: str, session_id: str | None = None) -> dict:
        """运行任务；session_id显式复用会话，省略则隔离上下文。审批仅交给宿主控制面。"""
        if callable(getattr(agent, "create_session", None)):
            session = (
                agent.create_or_get_session(session_id=session_id)
                if session_id is not None
                else agent.create_session()
            )
            try:
                result = await agent.run(task, session=session)
            finally:
                release = getattr(agent, "release_session", None)
                unresolved = getattr(session, "metadata", {}) or {}
                can_evict = getattr(agent, "session_store", None) is not None or not (
                    unresolved.get("_blocked_run") or unresolved.get("_pending_interrupt")
                )
                if session_id is None and callable(release) and can_evict:
                    await release(session.session_id)
        else:
            result = await agent.run(task)
        return _result_to_payload(result)

    server.add_tool(
        run_task,
        name="run_task",
        title="Run Task",
        description="Run a task with the wuwei agent. "
        "Returns the agent's final answer (content), token usage and run status.",
    )

    if include_status_tool:

        async def get_status() -> dict:
            """返回 wuwei agent 的元信息（名称/版本/工具数等）。"""
            info: dict[str, Any] = {"name": name, "server_version": server_version}
            for attr in ("version", "default_system_prompt", "default_max_steps"):
                value = getattr(agent, attr, None)
                if value is not None:
                    info[attr] = value
            registry = getattr(agent, "tool_registry", None)
            if registry is not None:
                try:
                    info["tool_count"] = len(registry.list_tools())
                except Exception:  # noqa: BLE001 —— 元信息尽力而为
                    info["tool_count"] = None
            return info

        server.add_tool(
            get_status,
            name="get_status",
            title="Get Status",
            description="Return metadata about the wuwei agent (name, version, tool count, ...).",
        )

    return server


def serve_mcp(
    agent: Any,
    *,
    name: str = "wuwei-agent",
    version: str | None = None,
    include_status_tool: bool = True,
) -> None:
    """以 stdio 传输阻塞运行 MCP server（同步，不返回）。

    供宿主进程通过 stdio 与 wuwei agent 通信：
    ``server.run(transport="stdio")`` 内部使用 anyio 事件循环。
    """
    server = build_mcp_server(
        agent,
        name=name,
        version=version,
        include_status_tool=include_status_tool,
    )
    server.run(transport="stdio")


def main(argv: list[str] | None = None) -> int:
    """``python -m wuwei.serve.mcp_server --stdio`` 入口。"""
    parser = argparse.ArgumentParser(
        prog="python -m wuwei.serve.mcp_server",
        description="Serve a wuwei agent as an MCP server over stdio.",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="run over stdio transport (currently the only supported transport)",
    )
    parser.add_argument("--name", default="wuwei-agent", help="MCP server name")
    parser.add_argument("--no-status-tool", action="store_true", help="skip the get_status tool")
    args = parser.parse_args(argv)

    if not args.stdio:
        parser.error("only --stdio transport is supported; pass --stdio")

    from wuwei.agent import Agent

    agent = Agent.from_env()
    print(
        f"wuwei MCP server starting on stdio (name={args.name!r})",
        file=sys.stderr,
    )
    serve_mcp(agent, name=args.name, include_status_tool=not args.no_status_tool)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
