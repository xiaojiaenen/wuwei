# 存储、调试与运维


`StateGraph` 节点返回 State 或字段更新 dict；`writes={"field": reducer}` 指定归并。未知字段写入 state.metadata，invoke/stream 使用相同 reducer。完整示例见 [离线示例](examples/offline.md)。

`await agent.fork(checkpoint_id)` 创建隔离会话与新 run，不修改原分支，默认模拟副作用。明确确认需要真实动作后才传 `allow_side_effects=True`。它是反事实调试，不是跨系统 exactly-once 重放。

## 上下文与原文

- `Context` 保存会话消息；窗口与压缩中间件减少模型上下文，不是原文存储。
- `StorageMiddleware` / `FileStorage` 以 session 归档；业务应定义保存期限和访问策略。
- `ContextLedger(persist_dir=...)` + `LedgerMiddleware` 外置大块结果；注册 `ledger_tools(ledger)` 供 read_context/pin_context/list_context 召回。持久原文与缓存分离；不持久模式为保留引用仍保留原文，总内存不保证有界。
- `MemoryMiddleware` / `CuratedMemoryStore` 用于记忆策展；`RagMiddleware` / KnowledgeStore 用于检索。vectorstore 提供内存及 Chroma、Qdrant、pgvector、Milvus 可选后端。

## 事件与集成

EventBus 订阅运行事件，ActionLog 记录动作；OpenTelemetry 需 observability extra。`wuwei-evals` 运行数据集与轨迹/输出/rubric 断言；LLM judge 需要模型调用预算。`wuwei-debug` 浏览检查点。多平台 gateway 包含飞书、钉钉、Telegram、微信与 Webhook 适配器；DecisionClient 提供决策服务入口。每种远程服务都应做部署环境独立验收。
