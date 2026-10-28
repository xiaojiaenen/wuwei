# 主流框架比较：下一阶段应补齐能力契约，而非增加包装层

核查日期：2026-10-28。对照 Wuwei 3.2.2 / `bb7b671`；研究只读取代码与官方文档，没有真实模型调用。引用的是当日移动中的官方文档，不等于所有历史版本均有相同能力。

## 结论

Wuwei 已有统一 Runner、审批恢复、SQLite 检查点、工具验证、预算统计、MCP、规划与 OTel 接入，不能再将这些列成“没有实现”。下一阶段最值得学的是 **Pydantic AI 的能力/输出分层、LangGraph 的状态与恢复模型、OpenAI Agents SDK 的会话/交接/追踪契约**。保留轻量内核，不建议复制一个 LangChain 式大集成层，或再造平台控制面。

以下是工程适配度选择，不是下载量排行榜。本轮重点核查这三个互补框架，不用其它框架的营销功能数替代实际契约比较。

## 1. 模型能力档案 + 输出模式：优先级最高

**外部证据。** Pydantic AI 区分 ToolOutput、NativeOutput 与 PromptedOutput；原生 schema 输出依赖模型/接口能力，提示词输出可靠性较低，仍需验证。它也将 provider 原生工具单独建模，而不是一律转换为本地函数工具。[输出模式](https://pydantic.dev/docs/ai/core-concepts/output/)、[原生工具](https://pydantic.dev/docs/ai/tools-toolsets/native-tools/)

**Wuwei 现状。** `wuwei/runtime/agent_runner.py:297-319,424-438` 的结构化输出仍是 schema 提示词 + 事后校验；`wuwei/core/message.py:63-71` 只有字符串 content；Responses 适配器主要归一化 output_text/function_call/reasoning。这使图片/文件、引用、服务端工具结果很难完整进入会话、检查点和结果。

**建议。** 建立小型 `ModelCapabilities`，声明原生 schema、工具、内容块和 token 参数支持情况；增加 `output_mode=auto/native/tool/prompted`，选定模式后将其记录在运行结果中，显式要求 native 而不支持时立即报错。然后增加向后兼容的 typed content parts（text/image/file/citation），再接 hosted web/file search。不能只给 build_request 增加 kwargs，就宣称功能已被 Runner 支持。

**验收。** 同一 schema 在 Responses/Chat/Anthropic 上走不同请求契约；不支持组合明确失败；文本旧调用不变；多模态/原生工具的普通、流式、持久化往返不丢字段。先实现 2–3 个真实适配器契约，不做动态“猜支持”网络探测。

## 2. 持久作业接口：按长期任务需求引入，不扩大 SQLite 承诺

**外部证据。** LangGraph 明确区分单 thread 的检查点与跨 thread 的 store；内存 saver 无法跨进程恢复。Pydantic AI 提供 durable execution 后端集成，使模型/工具执行挂接到成熟运行引擎。[LangGraph 持久化](https://docs.langchain.com/oss/python/langgraph/persistence)、[Pydantic AI durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/)

**Wuwei 现状。** 审批 checkpoint 已能持久化，但 `wuwei/agent/async_sub_agent.py:68-108` 的后台句柄包含 asyncio.Task，任务清单在进程内；因此“能存快照”与“后台作业重启后继续”是两个层次。此处属于明确能力边界，不是说已有审批恢复失效。

**建议。** 提供 `RunStore/JobBackend` 最小协议，区分 session history、run state、task lease 与业务 artifact。默认仍用本地实现；有跨进程/多 worker 需求时优先适配 Temporal/DBOS 等已有引擎之一，而不是在内核堆租约、心跳、定时器、队列。先提供统一 job status/result/cancel API。

**约束与验收。** 外部副作用仍要幂等键或权威对账；迁移到 durable 引擎不会自动带来 exactly-once。杀进程、换 worker、重复投递、过期 lease 和未知副作用分别测试；unknown 不自动重放。

## 3. 运行状态强类型化 + 版本迁移：维护性优先

**外部证据。** LangGraph 通过 checkpoint history 选择重放点和建立分支；分支之后的 LLM/API 节点会再次执行，不能把 time travel 理解为纯缓存读取。[Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)

**Wuwei 现状。** `AgentRunner._metadata` 同时容纳预算、审批、工具状态、structured output 和父子状态，`agent_runner.py:254-278` 虽写 `schema_version:1`，加载仍依赖开放字典的 update/get。功能已闭环，但跨模块共享字符串键、后续 schema 升级和状态不变量难以维护。

**建议。** 用内部 `RunState` / `PendingToolBatch` / `ApprovalDecision` 模型替换核心散落字典访问；持久序列化单独有版本、迁移与拒绝未来版本的规则。保留 middleware 自由 metadata，不把所有扩展字段硬塞进同一个庞大 Pydantic 模型。将现有 fork 与 checkpoint 历史暴露为只读 timeline；恢复、模拟重放、真实重放三类动作必须区分。

**验收。** 旧快照 fixture 可迁移；未来版本快照明确拒绝；状态约束校验不允许 succeeded 又 pending 等矛盾；原有 approval/unknown/replay 回归全部保留。

## 4. 可观测性连接评估：先让“通过”可信

**外部证据。** OpenAI Agents SDK 追踪 run、agent、model、tool、handoff、guardrail 等层次，并支持自定义导出。现有 OTel 资源清理只是基础，还需可关联的运行事实。[官方 tracing](https://openai.github.io/openai-agents-python/tracing/)

**Wuwei 现状。** `wuwei/observability/tracing.py` 有 model/tool span，但没有成套 run/session/parent-run 属性及状态、重试、审批耗时关联。`wuwei/evals/runner.py:91-110` 的通过判断只聚合内容/工具断言，不检查 run_result.status；incomplete/interrupted/error 即使文字匹配也可能被算为通过。主审已通过同目录 `reproduce_design_gaps.py` 离线复现该误判；应先修准确性，而不是再加一个观测后端。

**建议。** 默认评估断言 `status == completed`，可显式配置预期中断；记录模型/提示词/工具 schema 版本与本次 run id。扩展 trace 标准属性及统一脱敏/长度上限策略；CI 做 baseline diff（成功率、工具轨迹、token/延迟），确定性断言优先，LLM judge 独立计费且标注不确定性。

**验收。** 非完成状态不能因命中文字而通过；父子 trace 可串联；敏感参数不被日志/导出旁路泄漏；离线回归可读取同一运行事实，不靠扫描自然语言推断状态。

## 5. 类型化交接与上下文筛选：复杂协作比增加团队拓扑更有价值

**外部证据。** OpenAI Agents SDK 区分 agent-as-tool 与 handoff；handoff input filter 可以改变下一个 agent 看到的输入，同时保留 session history。其 Sessions 文档明确客户端历史与 previous_response_id 等服务端续接不能混为同一机制。[Handoffs](https://openai.github.io/openai-agents-python/handoffs/)、[Sessions](https://openai.github.io/openai-agents-python/sessions/)

**Wuwei 现状。** `wuwei/agent/sub_agent.py` 主要通过一个 task 字符串派发任务；恢复、预算已继承，缺少面向业务的输入/输出类型、artifact 引用和可配置 context selection。应补交接质量，而非再加一套“群聊/主管/蜂群”包装。

**建议。** SubAgent 支持可选 input_model/output_model、context_selector、artifact refs；manager delegation（返回给父级）与 handoff（接管当前会话）保持不同 API。将被筛选的模型输入与完整审计历史分开存储，避免复制整个父上下文或压缩后丢失工具关联。

**验收。** 下游只收到授权且必要的数据；结构结果仍经同一个 Runner/预算/审批；artifact 引用有所有者、类型、来源和生命周期；不同租户不能通过上下文选择器串数据。

## 推荐实施顺序

1. 先修评估状态误判，建立可信 baseline。
2. 模型能力档案 + native structured output；随后 typed content / 引用 / hosted tools。
3. 内部 RunState 类型化与快照迁移，补 run-level trace 关联。
4. 类型化交接/上下文筛选/轻量 artifact 引用。
5. 有明确跨进程长期任务需求时，再接一种 durable backend。

暂不优先：自建向量数据库、更多同质 provider 别名、可视化拖拽平台、实时语音、复制完整群聊生态。它们各有场景，但当前不能解决模型数据丢失、运行状态演进与评估真值这些更基础的约束。
