# 边界修复记录

## 已落地

- 工具注册使用 Pydantic 参数模型生成 schema，保留数组元素、嵌套定义和必填字段；执行前验证 JSON schema、签名和严格类型。非对象参数不能进入 handler。
- 同步 handler 通过线程执行，不阻塞事件循环。副作用工具不自动重试；超时结果标记 `status=unknown`、`execution_may_continue=true`，要求对账，不能将超时解释为动作停止。
- `bind_tool_execution(callback)` 通过 ContextVar 注入同一运行的治理执行入口。Code Mode 嵌套工具只使用该入口；直接 API 可显式传 `governed_execute`，无入口时拒绝嵌套执行。`run_code` 必填 code schema 已补全。它仍是可信代码的进程内执行，不是 OS 沙箱，不能隔离任意 Python，也不能强制终止同步死循环。
- 账本 read/pin schema 必填 entry_id；持久化条目被内存淘汰后可补读，pin 状态落盘；纯内存模式将淘汰内容留在独立原文存储，不丢仍被引用的内容（不是有界总内存存储）。
- graph 流式节点字典更新应用与 invoke 相同 reducer，公开导出 END；空 async generator 不引用未定义变量。
- MCP sse 使用独立旧版 SSE transport，HTTP 只向 Session 传读写二元流，兼容 SDK transport 三元组；发现失败也关闭已连接客户端。
- AG-UI 将 threadId 映射到支持 session_id 的 stream_events；保留 duck fake 兼容；错误不伪装成功，意外 EOF 报 IncompleteStream，提前退出关闭底层流；HTTP 发送异常也关闭响应流。
- A2A 明确只声明 0.3.0：根 URL 接受标准 `message/send` JSON-RPC，保留 `tasks/send` 弃用兼容。卡片不再伪装 1.0。标准错误包含 jsonrpc/id/error，id=0 正确保留；interrupted 映射 input-required 并携带恢复 interrupt。
- Streaming Pydantic Config 迁移至 ConfigDict。

## 验证

新测试先复现了数组 schema、阻塞同步 handler 和账本丢原文，再修复。

`pytest tests/test_boundary_hardening.py tests/test_mcp.py -q`：37 passed。
新增 8 条边界测试覆盖上述关键路径，包括标准 A2A ASGI 请求、错误信封、恢复数据、AG-UI session/关闭、超时后真实副作用仍完成但未重试、Code Mode 治理与图 reducer 等价。

`tests/test_serve.py`：23 passed，1 条旧测试仍要求虚假的 1.0 supportedInterfaces，需随明确版本策略更新。
`tests/test_v31_core.py tests/test_v32_features.py tests/test_v32_kernel.py`：41 passed，1 条旧直接 Code Mode 测试未显式提供治理入口，需迁移为 `run_code(..., governed_execute=...)`。

## 集成与实际缺口

主执行 owner 负责 Runner 绑定治理 callback、MCP 每次调用注入概览和 registry schema 刷新、运行依赖声明与旧测试迁移。

官方 A2A typed SDK 未安装，当前仅完成 ASGI 离线协议测试；没有远程 MCP、官方 A2A 客户端、真实模型或 OS 沙箱互操作验收。Code Mode 任意 Python 是可信执行入口；审批恢复不能靠重放整段代码来假装 exactly-once，嵌套挂起需要明确的恢复边界。

## 协议验收补充（推送前）

经授权在本地虚拟环境安装 **a2a-sdk 0.3.26**（`uv pip install --python .venv/bin/python 'a2a-sdk>=0.3,<0.4'`），没有安装 1.x，也没有发送远程请求。

用该官方 SDK 的 AgentCard、SendMessageRequest、SendMessageResponse、SendMessageSuccessResponse、JSONRPCErrorResponse 验证真实 ASGI 请求/响应：卡片版本 0.3.0、标准 message/send、RPC id=0、Task.history 的 kind/messageId、标准错误信封以及 completed/interrupted/incomplete/unknown/cancelled 五种状态映射均通过。generated agent message 当前已经带 kind 与 messageId；官方类型验证确认有效。标准入口补充验证 messageId/role/parts，tasks/send 继续保留旧形状兼容。

AG-UI 非流式 fallback 已修复：支持 run(session_id=...) 或 create_or_get_session + run(session=...) 的线程映射；先发 RUN_STARTED，异常与 incomplete/unknown/error/cancelled 发 RUN_ERROR，不再宣告成功；interrupted 发 approval_required 自定义事件。dict 与对象结果均支持。

`pytest tests/test_boundary_hardening.py tests/test_serve.py -q`：**42 passed**，其中官方 A2A 类型校验按五种状态参数化。SDK fixture 在没有可选 SDK 的环境显式 skip，仅本地已安装版本的此次结果证明官方 0.3 类型互操作。尚未运行远程服务器/官方网络客户端流式互操作；本出口未声明 streaming 或 A2A 1.0。
