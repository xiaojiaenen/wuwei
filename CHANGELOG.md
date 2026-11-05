# 更新日志

## 3.3.1 — 2026-11-05 · 流式兼容与运行时可中断性

- 修复流式请求从不设置 `Accept` 头：SDK 默认发 `application/json`，按 Accept 门控响应格式
  的兼容网关会返回裸 JSON，SSE 解码器得到 0 个事件，最终报出与根因无关的
  "stream ended without a terminal event"。OpenAI/Responses 适配器改为仅在 `stream=True`
  的请求上以请求级头强制 `Accept: text/event-stream`，不影响非流式路径；零事件流改报
  可定位的错误。
- 修复批处理用 `isinstance(result, Exception)` 判定失败：`asyncio.CancelledError` 属于
  `BaseException`，取消时原始异常对象被塞进结果列表并在后续序列化时崩溃。改为普通异常
  转工具错误消息，取消等 `BaseException` 原样向上传播。
- 修复流式消费方中断后会话锁永久占用的 R-9：运行改为在任务中持锁，事件经有界队列转交；
  消费方消失（队列满且超时未读）时取消该次运行并落 `cancelled` 检查点。Agent 层的 store
  锁与租约同步改为任务持有，`stream_idle_timeout=None` 可退回旧行为。
- 修复生成器被关闭时内层运行生成器不被关闭的隐患：取消清理不再晚于锁释放执行。
- 中断时区分"已派发"与"尚未派发"的工具：前者标 `unknown` 需对账，后者补记 `not_executed`
  结果并可安全重试，不再为从未执行的工具封锁会话。
- 工具超时语义明确：框架超时与工具自身抛出的 `TimeoutError` 分开判定；仅当工具声明
  `side_effect=True` 且未声明 `idempotent=True` 时标 `unknown`，其余标失败并允许重试。
- 工具批次并发加 `max_concurrency`（默认 8）上限；`is_concurrency_safe` 未显式声明时按
  `side_effect` 推导，写类工具默认串行；批次缺结果直接报错而非静默缩短结果列表。
- 工具输出序列化加字符上限（默认 100k）并标注截断长度。
- 运行异常收尾补发 `done` / `run_end`（status=error），客户端不再无限等待终止事件。
- 模型侧非契约异常统一包装为 `LLMError`，保留 `status_code`、分类与 `__cause__`；
  `ValueError` / `TypeError` / `TimeoutError` 等契约错误保持原样。
- 新增 `RepeatCallGuard` 中间件：连续同参调用超过阈值时返回错误结果，让模型改变策略。
- 事件类型声明与实际发射对齐（移除从未发射的 `task_start` / `task_end`），并移除运行器
  上未被使用的 `task` 形参。

## 3.3.0 — 2026-10-29 · 框架契约与持久执行

- 修复工具业务JSON误触发unknown、评估非完成结果假绿、YAML与Python模型参数不一致。
- 新增ModelProfile/OutputSpec：原生、工具及提示式结构化输出；严格schema转换、拒答语义与恢复保留。
- 新增类型化图片/文件/引用、owner隔离ArtifactStore及过期维护，贯通Agent、协议与持久会话。
- 新增可信ToolOutcome、瞬态RunContext依赖、类型化子代理输入/输出、上下文及产物选择。
- 新增版本化RunState、原子对账、SQLiteSessionStore、JobService与可选DBOS持久队列适配。
- 评估支持实际工具轨迹、多轮、限制、必需judge、baseline比较及资源关闭；完善运行追踪和安全诊断。
- 修复子进程取消清理，Docker按owned cidfile尝试清理；统一协议JSON类型化序列化。

## 3.2.2 — 2026-10-28

- OpenAI SDK 最低版本更新至 3.40.0，Responses 支持显式配置 prompt_cache_options / prompt_cache_key。
- 缓存读取、缓存写入和推理 token 细分统计贯通普通/流式运行、委派与规划汇总，不重复计入总 token。
- Responses 默认省略 temperature；保留显式配置、Chat Completions 默认值和适配器原有位置参数兼容性。
- 补充缓存配置、预算估算边界及离线 SDK 契约测试。

## 3.2.1 — 2026-10-28

- 更新补丁版本号，承接执行契约、模型接入及协议兼容性加固。

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
