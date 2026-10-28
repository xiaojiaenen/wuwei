# 更新日志

## 3.2.0 — 执行契约加固

- 普通/流式/结构化输出与委派共享治理生命周期；会话连续、累计预算与调用统计贯通。
- 多工具审批持久化完整批次，原子 claim 防止重复恢复；恢复继续执行权限检查；不确定副作用保留 unknown。
- 工具类型 schema 与执行前校验；同步 handler 不阻塞事件循环；副作用不盲目重试。
- 模型参数与 SDK client 注入分离、Responses 参数映射、终止原因/usage/reasoning 保留、分类重试和流清理。
- PlanExecutor.resume、resume_async_task 与非完成委派结果；后台句柄仍为进程内资源。
- 图流式 reducer 与 invoke 一致；END 公开导出。
- 账本召回 schema、持久 pin 与缓存淘汰后的原文召回。
- A2A 明确 0.3 message/send，tasks/send 弃用兼容；MCP 旧 SSE 与 HTTP 分开；AG-UI thread/session 映射和流关闭。
- fork 使用隔离分支，默认模拟副作用；Code Mode 嵌套工具经治理且明确为可信进程内执行。
- 公开指南、离线文档示例与迁移说明更新；在线测试显式 opt-in。

## 3.2 能力范围

本版本还提供 DAG 规划、插件与技能、虚拟文件系统、记忆/RAG/可选向量后端、预算/权限中间件、事件/动作日志、评估 CLI、决策客户端、平台 gateway 以及本地/Docker/E2B 执行驱动。远程服务和部署环境需独立验收；检查点不提供跨系统副作用 exactly-once。

## 兼容与展示

工具保留 display_name 中文展示名；类型化注册支持可选参数与列表元素 schema。

## [1.0.0] - 2026-05-08

### 新增

- 首个正式版本发布。
- 记忆衰减与自动清理机制。
- 长期记忆与 RAG 知识库功能。
- Skill 引用资源与缓存机制。
