"""wuwei.serve —— 「协议出口」：把 wuwei agent 暴露为网络公民。

三个出口：

- ``mcp_server``：MCP Server（官方 mcp SDK，stdio 传输）；
- ``a2a``：A2A（Agent2Agent）协议最小 ASGI 实现（Agent Card + tasks/send）；
- ``ag_ui``：AG-UI 事件映射与 SSE/NDJSON 事件流 ASGI 应用。

均为惰性导入（``mcp`` SDK 缺失只影响 MCP 出口），HTTP 出口零第三方依赖。
"""

from wuwei.serve.a2a import (
    build_a2a_app,
    build_agent_card,
    extract_text_from_parts,
)
from wuwei.serve.ag_ui import (
    ag_ui_events,
    build_ag_ui_app,
    map_agent_event,
)
from wuwei.serve.mcp_server import build_mcp_server, main, serve_mcp

__all__ = [
    "build_mcp_server",
    "serve_mcp",
    "main",
    "build_a2a_app",
    "build_agent_card",
    "extract_text_from_parts",
    "ag_ui_events",
    "build_ag_ui_app",
    "map_agent_event",
]
