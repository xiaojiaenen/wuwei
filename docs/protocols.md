# 协议与部署


MCP client 支持 stdio、Streamable HTTP 和独立旧 HTTP+SSE，transport 分别为 stdio/http/sse。`MCPSessionManager.connect_all()` 隔离服务器失败；使用完 `disconnect_all()`。动态发现只缓存发现结果，每次模型调用注入概览，tool_search 按需注册工具。远端 readOnlyHint 不是本地授权依据。

`build_a2a_app(agent, name=..., description=..., url=...)` 返回 ASGI 应用。声明 **A2A 0.3.0 JSON-RPC**，向卡片 URL 发标准 `message/send`。`tasks/send` 仅为弃用兼容入口，不声明 1.0：

```json
{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"kind":"message","messageId":"m1","role":"user","parts":[{"kind":"text","text":"你好"}]}}}
```

interrupted 映射 input-required，恢复数据在任务 metadata.interrupt 中。没有实现的协议方法返回错误，不假装支持完整 A2A 方法集。

`build_ag_ui_app(agent)` 的 POST /run 接收 threadId/messages，输出 SSE 或 NDJSON；`build_mcp_server`/`serve_mcp` 提供 MCP Server 出口。应用服务器部署、认证、TLS、限流由宿主负责。
