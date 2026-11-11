# 更新日志

# 更新日志

## 3.4.0 — 2026-11-12 · 审计修复：检查点瘦身、结构化输出收敛与跨平台修复

依据 2026-11 只读复审（`docs/audits/2026-11-11/`）对低效实现、重复逻辑、过度设计与
未解决问题的系统性修复。正确性内核（六态结果、unknown 对账、claim 租约、恢复语义）未动。

### 检查点与热路径（P1）

- **检查点 v2 格式**：此前同一份 metadata 在 `state` 列与 `metadata` 列各序列化一次，
  消息另以 `State.messages` 与 `metadata.session.context` 双份落盘。现在 runner 快照
  只写一份，`checkpointer.load` 从会话副本重灌 messages，所有消费方（resume、
  fork、debug.list_steps）零改动；v1 旧检查点与 StateGraph 直存路径行为不变。
- **保存次数**：`preparing`/`running` 两次 `tool_running` 快照在无治理中间件时合并；
  同一步内 `tool_result` 与 `step` 快照间无状态变化，合并为一条。
  实测 40 步工具循环：**201 次保存 / 13.32 MB → 121 次 / 5.18 MB（写出量 -61%）**。
  进一步压缩需 ActionLog 增量持久化（schema 级改动，本轮为保审计完整性未做）。
- `usage_records` 保留完整逐条记录：`BudgetMiddleware` 按记录逐条估算成本，
  截断会破坏预算语义。
- 工具参数校验收敛：`jsonschema` validator 编译一次、`get_type_hints`/`signature`/
  `TypeAdapter` 反射缓存一次（此前每次调用重建四份）。
- 流式热路径：移除逐 delta 无消费者的 `provider_delta` 全量序列化（runner 侧同名键
  覆盖只保留最后一个）；工具参数分片改列表一次性 `join`（大参数 O(n²) → O(n)）；
  流式终止 JSON 解析错误统一为 `Malformed tool arguments`（与适配器契约一致）；
  Anthropic 适配器的 SDK 能力反射探测缓存为一次。

### 结构化输出（P0）

- `auto` 降级链从 native→prompted 扩展为 **native→tool→prompted**：兼容端点
  （自定义 base_url）不再默认落到最不可靠的提示词模式；带可执行工具的请求保持
  prompted（结构化输出工具必须独占 tool_choice）。
- prompted 注入与 runner 恢复路径共用唯一实现 `llm/output.py:inject_prompted_schema`：
  并入首条 system 而不是追加到消息尾部（尾部 system 是严格端点 400 的根因，
  3.3.3 只修了 runner 一侧）。两条路径注入文案一致且幂等。
- 新增 `tests/test_structured_output_matrix.py`（provider×mode 矩阵）。

### 跨平台与缺陷修复（P0）

- 修复 `ArtifactStore.put` 在 Windows 必抛 `PermissionError`（目录 fsync 是 POSIX
  惯用法），且失败发生在 blob/meta 写盘之后、留下孤儿文件——8 个测试转绿。
- 修复 SQLite 连接从不关闭（`with sqlite3.connect` 只提交不关闭）：Windows 文件锁
  导致临时目录清理失败、生产环境句柄泄漏。涉及 graph/checkpoint、persistence/sessions、
  persistence/jobs。
- 修复 Windows 进程树终止：shell 启动的命令是孙进程，只 kill 直接子进程不传播，
  超时/取消后 descendant 存活；改用 `taskkill /F /T`（POSIX 沿用进程组）。
- `tests/test_quality_contracts.py`、`test_sandbox.py`、租约测试跨平台化
  （`os.kill(pid,0)` 语义、POSIX 引号、`ps`/`pwd` 依赖、时间余量）。
- CI windows 矩阵首次运行即捕获 11 个存量平台缺陷并修复：python/skill 工具与
  `LocalPythonDriver` 子进程加 `-X utf8`（Windows 控制台 cp1252 打印中文直接崩溃）、
  evals CLI 输出强制 UTF-8 stdio、文档硬化测试按 UTF-8 读取中文文档、
  docker 集成测试在 Windows runner 跳过（Windows 容器模式无法运行 Linux 镜像）。

### 委派与治理（P2）

- 新增 `wuwei/agent/child.py:build_child_agent` 作为受治理子代理的单一构建点，
  收敛 4 处各自为政的 `Agent(...)` 构造（sub_agent / multi_agent / PlanExecutor /
  内置规划）。计划任务此前丢失 event_bus 透传，现已统一接入。
- 修复 `SubAgent.to_tool_schema` 对模型撒谎：声明 `input_model` 时 schema 仍广播
  "自然语言 task"，实际要求 JSON 序列化串——模型 schema 现已写进参数描述。
- `inherit_middleware` 迁至 `child.py`，`sub_agent` 保留再导出。
- multi_agent 文档订正：图编排本身无恢复入口，成员会话可单独恢复。

### 缺陷与契约（N）

- `LLMError.details["provider"]` 恒为类名 "LLMGateway"，现反映真实供应商。
- http 插件不再把异常吞成 `"请求失败: ..."` 成功文本（违反框架自身 `ok:false`
  错误契约）；连接失败直接抛出由执行器包装为结构化错误；截断标注原文长度。
- skill 指引按 session 注入（此前按中间件实例一次性，跨会话静默丢失），并改为
  并入首条 system 而非再插一条。
- `handle_task_tool` 的 ToolMessage call_id 不再拿工具名充数。

### 死代码与文档（P3）

- 删除零引用实现：`memory/context_compressor.py`、`memory/context_window.py`、
  `Context.keep_last_turns`、`session.summary` 僵尸字段、`multi_agent.HandoffSignal`、
  `Swarm` 别名、`TeamMember.tools`（写入后无读者）、`create_planner`、
  `gateway._create_fallback_adapter`、sub_agent `before_llm` 死写、
  `run_plan` 不可达分支。
- `_extract_json` 迁至 `runtime/agent_runner.py`（消除 runtime→agent 反向导入），
  `agent.py` 保留再导出。
- 文档订正：tiktoken 注释与实现不符、decision 端点文档与代码不一致
  （统一 `{base_url}/systemone`）、compression 文档夸大接入状态、
  README 将 sandbox（Docker/E2B）标注为实验性未接入模块。

### 测试与 CI

- `tests/test_coverage_gaps.py` 从模块级脚本（10 个无断言 `test_` 函数在 CI 恒通过、
  Windows 上 import 即 `sys.exit(1)` 炸掉收集）改写为 22 条真断言用例。
- 新增 `tests/test_audit_regressions.py`（12 条）：检查点 v2/v1 兼容、Ledger 全量保真、
  schema 契约、委派工厂语义、http 错误契约、skill per-session 注入、provider 字段。
- `docs/audits/2026-11-11/reproduce_efficiency_gaps.py` 断言化：40 步 ≤130 次保存 /
  ≤6 MB、归档 == 原文、prompted 角色断言，作为改造门禁可重复执行。
- CI 增加 **windows-latest** 矩阵（py3.12）：Windows 专属缺陷（目录 fsync、进程树
  终止、SQLite 锁、路径语义）此前从未被 ubuntu-only CI 捕获；发布链（DBOS/docs/build）
  仍由 ubuntu 承载。
- 本地基线：16 failed（Windows）→ **0 failed**；全量 598 passed / 12 skipped。


## 3.3.3 — 2026-11-10 · 结构化输出与 system 消息

- 修复结构化输出（`response_model` / `output_spec` 的 JSON 提示路径）把 schema 说明
  `insert(0, SystemMessage(...))` 到消息视图最前面，与会话原有 system prompt 形成
  相邻两条 system，被 OpenAI 兼容端点（DeepSeek / vLLM / 内网网关）以
  `400 System message must be at the beginning` 拒绝的问题。
  - 注入改为并入首条 system 消息：system 条数与顺序不变，prompt 前缀稳定，重复注入幂等。
  - 适配器边界合并开头连续的 system（`Anthropic`、`Responses` 一直如此，补齐 OpenAI 一侧），
    覆盖 RAG / 记忆等中间件注入的 system；对话中段的 system 保持原样。
  - 归因订正：注入只作用于本轮 LLM 调用的消息视图（`_ctx()` 深拷贝），不会写入会话历史、
    也不随轮次累加——会话历史始终只有一条 system。
- `AgentRunResult` 新增 `steps_used` / `max_steps`，`max_steps` 截断场景可在 done 事件中
  程序化判断，不必匹配 content 文案；截断文案同时带上具体步数。
- 测试：新增 `tests/test_structured_system_message.py`（20 条），覆盖注入合并、幂等、
  多模态 system、适配器合并、中段 system 保留、历史不被污染与 max_steps 载荷。

## 3.3.2 — 2026-11-09 · ActionLog 恢复类型收敛

- 修复从旧/外部 checkpoint 恢复后 `ActionLog` 退化为普通 dict、首次 `append` 抛
  `AttributeError`，导致 agent 工具调用后静默无最终回复的问题。
  - 归因订正：`graph/checkpoint.py` 的 `dict(metadata)` 处理的是 checkpoint 级元数据
    （`run_id` / `parent_checkpoint_id`），不是 `session.metadata`。后者从 3.2 起就由
    `AgentSession.to_dict/from_dict` 用 `__type__` 标记还原，正常恢复路径不复现。
  - 缺陷本体在读侧入口 `get_or_create_log` 只认 `None`：3.2 前的无标记快照、
    第三方 SessionStore、外部代码写入的裸 dict 会被原样放行。
  - 新增 `coerce_action_log` 作为唯一类型收敛点，同时识别带标记形态与裸形态；
    未知形态在入口报 `TypeError`，不再拖到工具执行收尾才崩。
  - `ActionLog.from_dict` 兼容两种落盘形态（只认裸形态会把带标记 dict 当成空日志，
    静默丢历史）；`AgentSession.to_dict/from_dict` 与 `get_or_create_log` 共用同一入口，
    写侧也规范化，避免脏形态在快照之间一直传下去。
  - `get_or_create_log` 回填实例，避免每次重跑全量反序列化、子代理写到不同实例。
- 测试：新增 `tests/test_action_log_restore.py`（17 条），覆盖官方往返、旧 checkpoint、
  裸 dict、带标记 dict、identity 回填与异常形态。

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
