# 框架改造交付记录

完成日期：2026-10-29。基线：3.2.2 / bb7b671。代码在 codex/framework-hardening-20261028 功能分支；新增能力未发布到PyPI。此次仅修改Wuwei。

## 审查项与实际落地

| 审查项 | 落地与使用入口 |
|---|---|
| F1业务正文误触发unknown | ToolOutcome信封决定执行事实，普通status/ok正文无控制权；运行、CodeMode和委派均使用显式结果 |
| F2评估假绿 | completed默认、expected_status、实际工具轨迹、required judge、空断言策略、JUnit skipped、资源关闭失败断言 |
| F3 YAML脱节 | effective_llm_config保留unset/null差异；provider_options、缓存、reasoning、profile；Python/YAML等价请求测试 |
| A1会话生命周期 | SQLiteSessionStore namespace/TTL/租约/续期/提交fencing；Agent.load_session/release_session/aclose；MCP成功临时会话释放 |
| A2持久任务 | SQLiteJobStore/JobService查询、幂等、审批signal、cancel、unknown对账；DBOSJobBackend和真实workflow/step适配；过期运行不重放 |
| A3模型/消息契约 | ModelCapabilities/ModelProfile；text/image/file/citation；ArtifactRef/Store；Agent→gateway→session→协议保留类型化内容及provider metadata |
| A4原生结构输出 | OutputSpec auto/native/tool/prompted；OpenAI严格schema转换、开放字典明确拒绝、refusal非成功、审批恢复保留策略；PlanSpec不再混入任务执行状态 |
| A5类型化状态/依赖 | RunState v1→v2与未来版本拒绝、工具状态不变量；RunContext依赖不进入schema和检查点；依赖继承显式opt-in |
| A6观测与取消 | run/model/tool层级、trace/run/session关联、用量/重试事实、默认不采集正文、脱敏与诊断summary；拥有的子进程组kill/drain/reap；Docker cidfile定向清理 |
| 类型化委派 | input_model/output_model、context_keys/artifact_keys、owner校验、类型化媒体输入与结构结果，不复制全部父上下文 |
| 文件生命周期 | owner/媒体类型/大小/过期校验、私有文件、路径/伪造ref/符号链接拒绝；显式delete/purge_expired |
| 协议JSON | HTTP/SSE/NDJSON共用已知类型编码，日期/UUID/Decimal/dataclass/Pydantic不崩；未知对象不默认str()泄漏 |

## 验收证据

- Python3.12本地完整套件：500 passed、10 skipped（最后集成候选）。
- Python3.11 / 3.13的新依赖环境全量通过；最新提交的三版本验收以GitHub CI为准。
- DBOS3.3真实SDK+本地SQLite队列+fake LLM测试通过；不是只mock一个enqueue调用。CI的3.12任务新增这一独立验收步骤。
- 新增专项测试：model_contracts、runtime_contracts、quality_contracts、durable_sessions_jobs、dbos_jobs、docker_cleanup、professional_integration、protocol_serialization。
- 原三个复现：YAML缓存/null均接受、Responses不发送默认温度、incomplete评估False、业务unknown的成功lookup完成。
- Ruff关键错误、git diff --check、MkDocs --strict、uv build、Twine --strict通过。
- 当前wheel独立隔离导入及exports、artifact读/删smoke通过；包含新模块，不依赖源码路径替代wheel。
- 独立审查发现的“未缓存blocked会话可删除”“close异常丢失已完成job”“外部传入session执行中被close”均修复并回归。租约过期的旧clean快照也不能绕过unknown保护。

## 使用条件

- 跳过项为可选SDK/外部执行环境相关测试，不能计作通过。没有调用真实模型、远端MCP、Docker守护进程、E2B或生产DBOS/Postgres服务。
- 原生schema/媒体格式用真实SDK的MockTransport及可序列化SDK对象验证，不代表每个模型和账号均有相应能力。
- Docker取消/超时尝试以私有cidfile中的完整ID执行rm -f；未知ID、守护进程故障或删除失败时清理不被冒充为成功。
- namespace、artifact owner和approval operator由可信宿主鉴权后提供。持久化分区不替代用户/租户权限检查。
- 未知外部动作需要目的端证据/幂等策略；没有宣称跨系统exactly-once。

使用入口：[持久运行](../../guides/durable-execution.md)、[模型契约](../../model-contracts.md)、[运行契约](../../runtime-contracts.md)、[质量与生命周期](../../guides/quality-and-lifecycle.md)。

首次GitHub CI发现shell后代取消测试把Linux zombie当作仍在运行。取消验收现与已有timeout验收一致：检查后代真实进程状态，只接受已消失或已终止的Z状态；直接拥有的子进程仍要求wait完成后PID不可查。没有跳过测试或增加时间来掩盖运行中的进程。
