# Wuwei 应借鉴的框架契约与 API 设计

核查日期：2026-10-27。本文依据访问到的官方文档和官方仓库；移动中的文档/main 链接用于架构评估，落地实现时仍需对目标依赖版本做兼容测试。Wuwei 的本地复现见同目录 REVIEW.md，本文不把其他框架的设计直接当作 Wuwei 已具备的能力。

## 判断：学执行契约，不学功能清单

最值得借鉴的是三个不同层面的成熟做法：

| 参考 | 精选做法 | 对 Wuwei 的作用 | 不宜照搬 |
|---|---|---|---|
| LangGraph | 显式 state/reducer、thread/checkpoint、重执行边界 | 把恢复状态和普通会话历史分开建模 | 不为模仿 API 增加一套空 Channels/Pregel 抽象 |
| OpenAI Agents SDK | 统一 Runner 生命周期、RunState 审批恢复、流消费和清理边界 | 统一普通/流式/子代理/恢复的行为 | 不将 provider 专属 hosted 工具能力泛化为所有模型能力 |
| Pydantic AI | 输出模式、验证重试、provider/client 注入、父子 usage | 让工具参数/模型参数/输出都具有可检验契约 | 不把仅提示词 JSON 提取包装成原生结构化输出保证 |

以上是结合当前问题的工程选择，不是框架排名。建议继续维护薄内核，先完成一条“模型调用 → 完整工具批次 → 权限/审批 → 结果持久化 → 后续调用”的纵向闭环，再扩展后端数量。

## 1. LangGraph：检查点并不自动带来副作用 exactly-once

官方将 checkpointer 的 thread-scoped 图状态与跨线程 store 分开；同一个 thread_id 指向同一条持久状态，内存 saver 不能跨进程重启保存。Wuwei 可以保留轻量 SQLite 后端，但需要明确 Session（会话）、Run（执行）、Checkpoint（运行状态快照）之间的归属，而不是只保存消息后仍依赖内存 session 字典恢复。[LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

`interrupt()` 恢复会从所在节点开始重新执行，因此节点内中断前的动作可能再次发生。官方要求这些动作幂等，或拆到独立任务/节点，且不要改变中断调用的匹配顺序。Wuwei 应持久化整个 pending 批次和已完成结果，不能只记住触发审批的一个工具；审批恢复也不能跳过所有权限中间件。[LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

节点状态更新由 schema 和 reducer 共同定义，图的执行算法围绕更新传播运行。对 Wuwei，更经济的采用方式是让 invoke/stream 复用同一节点调用及 reducer 归并函数；不需要先复制 LangGraph 的整个调度引擎。[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

**落地约束：**外部动作的可靠性还需幂等键/权威结果对账。即使本地存了“开始执行”，也无法凭这个标志知道远端是否已成功；无法确定结果时应保留 unknown 并停止自动重放。这是跨系统原子性边界，而不是多存一次 checkpoint 就能消除。

## 2. OpenAI Agents SDK：恢复审批状态不等于授权绕过或防重完成

官方 RunState 包含执行、审批和待执行工具状态。长时审批应把完整快照保留在服务端，客户端仅提交被授权的 opaque decision 标识；服务端验证审核人、所属 run、具体 pending 项，并在执行前原子消费请求。官方还明确：审批通过不保证副作用恰好执行一次，外部成功、本地快照未更新的崩溃窗口需要目的端幂等或权威对账；结果不确定时不要自动重放。[Human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)

对 Wuwei 应采用：
- approval 决定绑定 tool_call_id、原始参数及运行归属；不是任意字典加 `approved=True`。
- resume 先校验/消费审批决定，再进入正常的工具执行治理链。
- 工具状态至少区分 planned、awaiting_approval、running、succeeded、failed、unknown；挂起不是失败重试。
- fork 创建新 run/session 与父 checkpoint 关联，不回写原分支；副作用重放策略单独控制。

这些是本项目应建立的状态契约；无需一开始模仿 SDK 所有序列化选项。

## 3. 流式：最后一个 token 不是运行结束，停止消费也不等于取消

官方要求持续消费 `stream_events()` 到迭代结束，最后可见 token 之后还可能完成会话存储、审批记账和压缩；取消后也应继续排空以完成清理。消费者提前退出与后台 run 生命周期不同。[OpenAI Agents Streaming](https://openai.github.io/openai-agents-python/streaming/)

Wuwei 的事件契约应明确：
1. text_delta 是显示进度，不代表最终状态。
2. terminal 事件在状态/统计/检查点提交后才发出；调用者看到完成即可读取最终状态。
3. 取消必须传播到当前模型/工具及子任务，并定义清理和已发生副作用的保存方式。
4. 同一 session 并发运行必须有串行化或冲突策略，不能只依靠共享可变列表。

OpenAI SDK 的官方生命周期约束还要求：相同模型行为下，普通/流式在最终输出、生成 items、usage、guardrail、session history 和 interruptions 上等价；恢复继续暂停的 turn，不能重置计步/预算。[官方 Runner lifecycle](https://github.com/openai/openai-agents-python/blob/main/.agents/references/runner-lifecycle.md)

这正是 Wuwei 应优先写成组合测试的要求，比每个新模块只测“可构造”更有价值。

## 4. 重试：区分传输失败、模型校验修复与业务副作用

Pydantic AI 将 transport/provider 重试、工具/输出修复与 durable 重放视作不同层；多个层级的尝试次数会相乘。官方给出由一个 transport 负责重试时关闭 SDK 重试的例子，也区分 Retry-After 和不可修复请求错误。[Pydantic AI Retries](https://pydantic.dev/docs/ai/core-concepts/retries/)

OpenAI Agents 的 runner 重试模型显式带 `response_started`、`replay_safety`、`stateful_request`。已开始不安全输出的流、abort 及本地副作用重放否决，不会因普通 retry=True 被绕过；状态化请求在重放安全性未知时默认拒绝重放。[OpenAI Agents Models: Runner-managed retries](https://openai.github.io/openai-agents-python/models/#runner-managed-retries)

**建议 Wuwei 采用的最小政策：**
- TypeError、参数错误、解析/校验程序错误不做同请求传输重试。
- 429/暂时性连接错误/部分 5xx 依据规范化错误事实处理，记录 SDK 与框架各自的次数。
- 流式重试要看是否产生可观察输出或副作用、是否为状态化 provider 请求；“尚未吐字”只是必要条件之一。
- 已发生工具写入但结果缺失，不作为可重试工具异常一概重跑。
- 输出 schema 校验失败可以提示模型纠正，但计入同一个 run 的预算与统计。

## 5. Pydantic AI：结构化输出应是运行配置，不是裸 LLM 的旁路

Pydantic AI 区分 ToolOutput、NativeOutput 和 PromptedOutput：原生 schema 输出依赖模型能力；PromptedOutput 只是提示模型遵守 schema，可靠性较低，仍需验证和有限重试。这些都属于同一 Agent 的输出配置，而不是另开不受治理的 llm.generate 入口。[Pydantic AI Output](https://pydantic.dev/docs/ai/core-concepts/output/)

Wuwei 应把 `run_structured` 收敛为正常 run 的一个参数/门面：保留既定 system、session、middleware、usage、checkpoint 和工具能力；返回统一 RunResult 的 typed output 字段。文本提取 JSON 可以保留为兼容后备，但不能声称等价于 provider 原生严格 schema。

## 6. 子代理：继承预算与取消域，不是无条件复制全部可变中间件

Pydantic AI 的委派示例显式传 `usage=ctx.usage`，使子调用计入父总量；也说明多模型总成本不能仅用聚合 token 和单一价格重建。依赖通常复用或取子集，取消作用域也有明确语义。[Pydantic AI Multi-agent Applications](https://pydantic.dev/docs/ai/guides/multi-agent-applications/)

OpenAI Agents 的 usage 文档将每个会话 turn 的使用量与会话历史区分；RunState 捕获当前 usage 快照，恢复在其基础上累加，独立分叉的 usage 不互相污染；嵌套 agent-as-tool 的后续 usage 再归入活跃外层 run。[OpenAI Agents Usage](https://openai.github.io/openai-agents-python/usage/)

**Wuwei 的合理边界：**
- 子代理共用执行内核，run_id 独立、parent_run_id 明确。
- 权限可收紧，扩权需显式授权；预算可子分配，但同时受父级剩余额度约束。
- 注入的 middleware 配置与单次执行的可变状态分离，避免父子并发覆写 `_executor`、spend 或 taint。
- 保留按每次请求的 model/provider/usage 记录，再汇总成本；不要用父模型的单一价格表估算所有孩子。

## 7. 审批、内容 guardrail、权限、OS 沙箱是四种边界

OpenAI Agents 文档清楚区分 agent 输入/输出检查与每次 function-tool 调用检查。工具审批通过后，输入工具检查默认仍在实际执行前运行；即使选择审批前检查，通过后也再次检查。不是所有 hosted/built-in 工具都进入同一条 guardrail 管线，不能把“装了一个 guardrail”当成全系统授权保证。[OpenAI Agents Guardrails](https://openai.github.io/openai-agents-python/guardrails/)

对 Wuwei 的判断：
- HITL 解决人对一次动作的决定；Permission 解决该动作在当前能力授权内是否允许；两者不能互相替代。
- 模型分类器拒绝某些内容，不提供文件系统/进程隔离保证。
- OS 沙箱是实际执行环境约束。进程内 `exec`、文件工具的路径检查和文本上的“不可信”标记，都不等价于 OS 隔离。
- 不应把成熟 SDK 默认的并行输入 guardrail 直接复制为写入权限检查：检查尚未完成时业务动作就可能已发生。

MCP 规范要求实现者验证工具输入、控制访问，并提醒非可信服务器的 tool annotations 不应被当作安全事实。Wuwei 的 concurrency/read-only/side-effect 元数据可以用于调度提示，但授权必须基于本地可信配置和真实参数，不能信任远端自报“只读”。[MCP Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

## 8. Provider 和 client 注入：迁移痛点要留在模型边界

Pydantic AI 的 OpenAIProvider 接受现成的 AsyncOpenAI client，支持组织、代理地址等客户端配置，不需要调用者重写全部消息/响应转换。[Pydantic AI OpenAI provider](https://pydantic.dev/docs/ai/models/openai/#custom-openai-client)

Wuwei 可采用三层配置：

```text
ProviderConfig / client             认证、base_url、连接池、SDK重试、关闭责任
ModelSettings                      模型、预算、推理和输出能力
Request / provider-specific extras 每次调用的合法请求字段
```

自定义企业认证应替换 client 或 provider factory；不应修改 `gateway.adapter` 私有运行假设，也不应把 max_retries/default_headers/http_client 展开进模型请求。模型专属能力用 profile/capabilities 明确表达；未知参数早失败，而不是静默删掉后宣称兼容。

## 9. 用户 3.0.2 反馈中，两处修法需要校正

### 连续 user 不是“必然 roles must alternate 400”

Anthropic SDK 的 Messages 参数说明明确写到：连续 user 或 assistant turns 会合并成一个 turn。所以仅凭“为两个 tool_result 分别生成两条连续 user 消息”，不能证明服务端必然因角色不交替报错。[Anthropic SDK message_create_params.py](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message_create_params.py)

不过官方并行工具指南要求：下一条 user 消息一起返回每个 tool_use 对应的 tool_result，按 tool_use_id 配对，tool_result 位于文本之前；即使不执行某个调用，也返回 is_error 解释。Wuwei 主动合并并检查完整批次仍然是正确的序列化策略，但验收依据应是工具配对/邻接契约，而不是未经验证的角色交替报错。[Claude Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)、[Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)

### “pending 非空就组装执行”“未知终止原因默认 stop”都不安全

官方把 max_tokens 和 context-window-exceeded 当截断，而不是自然完成；不完整 tool_use 被截断时应先取得完整调用，不能直接执行部分 JSON。[Claude Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)

Wuwei 应保留 raw_finish_reason，并规范化为 completed/tool_calls/incomplete/refused/aborted/unknown 等明确状态。只有工具调用已完整终止、参数 JSON 和 schema 验证通过、provider 请求成功且权限通过才进入执行。某兼容网关返回 stop 但确实提供完整工具调用，可以作为针对该 provider 的兼容规则；不能推广到 length、abort 或断连 EOF。JSON 解析失败要产生不可执行的工具解析错误，不要替换成 `{}` 或添加 `_raw_arguments` 后照常调用。

## 10. 如何认定“过时”，以及本轮不确定点

- **官方弃用/移除：**要有官方 deprecated/removal 记录或目标版本源码依据；不能因为 Responses 新于 Chat Completions 就判后者已移除。
- **内部陈旧 API：**文档导入已删除对象、旧 response 形状、流/非流契约不一致，是 Wuwei 自己的兼容性问题。
- **设计欠缺：**缺少 provider 注入、裸字典流事件、不可恢复执行状态，并非上游 deprecated，但应因已确认的故障收敛。
- **实验性和版本变动：**官方文档的实验能力不应直接变成 Wuwei 核心依赖；先固定目标版本和薄适配层。

本轮没有替 Wuwei 真实调用多个 provider、没有证明远端代理实现的完整工具/重试/多模态兼容。普通 Chat Completions、Responses、MCP SDK 的具体版本下限和正式迁移记录，由同轮主报告按 Wuwei 当前依赖声明另行核对。本文不据移动文档声称所有新特性在 Wuwei 声明的最低 SDK 版本上都可用。

## 采用优先级

1. **现在采用：**完整 RunState、统一工具执行协议、审批后仍检查权限、终止状态可靠、provider/client 参数分层、失败白名单重试。
2. **随后采用：**同内核子代理/typed output、父子预算归因、流式清理和普通/流式等价测试、版本化持久化 schema。
3. **暂不扩张：**新持久化引擎、新图调度抽象、新 hosted 协议、新后端列表。先让已有公开入口进入同一治理链路。

## 补充：A2A 是真实协议契约落后，不是“加两个别名就兼容”

当前 `/Users/xiaojia/code/wuwei/wuwei/serve/a2a.py` 声明 0.3.x/1.0 卡片兼容，却仅识别 `tasks/send` 方法；HTTP 同时挂载 `/tasks/send` 和根路径 `/`。标准 JSON-RPC 客户端可以到达卡片根 URL，但 `message/send` 或 `SendMessage` 会被 handler 判为不支持。应按下面两种版本分别实现/校验：

| 契约 | A2A 0.3 | A2A 1.0 |
|---|---|---|
| JSON-RPC 发消息方法 | `message/send` | `SendMessage` |
| JSON-RPC HTTP 入口 | POST 到卡片声明的服务 URL；不是把 method 当 REST 路径 | POST 到所选 supportedInterface 的 URL |
| HTTP+JSON/REST 发消息 | `POST /v1/message:send` | `POST /message:send` |
| 卡片接口发现 | `url` / `preferredTransport` / `additionalInterfaces` | `supportedInterfaces[]`，每项含 `url`、`protocolBinding`、`protocolVersion` |

0.3 的 JSON-RPC 请求/响应必须遵守 2.0 信封，发送操作不是 `tasks/send`；URL 来自 AgentCard，并没有要求统一固定为根路径或 `/rpc`。因此路由是否正确要结合卡片中的 URL 判断，不应只比较路径拼写。[A2A 0.3 specification](https://a2a-protocol.org/v0.3.0/specification/)

1.0 接口项要求 `protocolBinding`，值可为 JSONRPC/GRPC/HTTP+JSON；不是 `transport`。JSON-RPC 方法改为 PascalCase。REST 使用不同的路径与直接请求对象，不能把 JSON-RPC method 和 REST endpoint 混在一起。[A2A 1.0 specification，4.4.6、9、11 节](https://a2a-protocol.org/v1.0.0/specification/)

1.0 还移除了旧 kind discriminator，更新了状态/角色等枚举，并把卡片顶层协议声明移到接口项。当前同时附带 `kind`/`type`、顶层 0.3 和错误形状的 1.0 接口，不能证明两版兼容。更合理的方案是确定一个首选版本、使用目标版本官方 SDK/生成类型，再为确有下游需求的旧版本提供显式适配与独立互操作测试。[A2A 官方 1.0 迁移说明](https://a2a-protocol.org/v1.0.1/whats-new-v1/)
