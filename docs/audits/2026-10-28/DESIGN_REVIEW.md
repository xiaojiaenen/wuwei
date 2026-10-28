# Wuwei 3.2.2：从功能覆盖走向可依赖的框架

日期：2026-10-28。基线：`bb7b671` / PyPI 3.2.2。此次是调研和评估，不修改运行时代码、不发布版本、不调用真实模型或付费服务。

## 结论与推荐方向

Wuwei 已有统一 Runner、工具治理、预算、审批检查点、委派、图、MCP/A2A/AG-UI 与评估等模块。上一轮已经修复的审批消费、流式统计、父子预算等问题不应重新当作“尚未实现”。

当前短板不是功能少，而是**同一能力在 Python/YAML/评估/协议之间的契约没有完全闭合**，以及轻量进程内框架与生产持久任务平台的边界不够易用。建议定位为：**轻量、跨模型、可审计和可恢复的 Python Agent 框架**。保留薄内核，不照搬另一套图调度引擎或先做低代码平台。

首先修复三个已复现缺陷；随后建设模型能力契约、类型化消息/工具结果、持久 RunStore 和评估门禁。扩展到实时语音、群体代理、可视化编辑器应后置。

## 一、已复现的缺陷

复现命令（仓库根目录）：

```bash
PYTHONPATH=. .venv/bin/python docs/audits/2026-10-28/reproduce_design_gaps.py
```

脚本只使用 fake LLM / 未联网客户端。输出：

```text
yaml rejects: ('prompt_cache_options',) extra_forbidden
yaml rejects: ('temperature',) float_type
yaml Responses temperature: 0.2
incomplete eval passed: True
successful lookup run status: unknown tool_outcome_unknown
```

### F1 / P1：控制状态从工具正文猜测，普通业务数据可阻塞会话

证据：`wuwei/runtime/agent_runner.py:697-707`，`_commit_tool` 除看 metadata，还对 `final.content` 搜索字符串 `"status": "unknown"`。复现中一个正常成功、只读的 lookup 返回该业务字段，run 就以 unknown 终止，后续模型调用没有发生。

影响：业务状态与运行控制状态耦合；返回 JSON 的排版还能改变执行行为。不是必须发生外部写入才会触发。

建议：引入类型化 ToolOutcome，把 success/error/unknown、是否发生副作用、重试/对账信息放在受信任执行信封；正文只作业务数据。对子代理/协议结果做显式转换，不再让内核解析任意正文决定执行状态。

验收：同一个业务 JSON 用不同空格/嵌套/普通文本均不会改变治理；执行器明确报告 unknown 时仍阻塞并保留恢复信息。

### F2 / P1：EvalRunner 没有检验执行完成状态，产生假绿

证据：`wuwei/evals/runner.py:91-110`。passed 只由内容、工具、schema 和 rubric 断言组成。复现 status=incomplete、content=expected，contains 断言匹配即 passed=True。缺少断言的 case 也可能空泛通过。工具断言从 assistant.tool_calls 取名，是请求轨迹，并不证明工具执行成功。

影响：发布验证可能奖励未完成任务；工具被拒绝/失败也可能被计作“调用过”。rubric 已配置却没有 judge 时会 skipped，报告没有充分区分“未评估”和成功。

建议：默认要求 completed；需要测试 interrupted/unknown 时由 case 显式设置 expected_status。区分 requested/executed/succeeded 的工具断言。为 required judge、空断言、预算/时延阈值建立明确策略；报告保留 status、termination_reason、usage_available。

验收：各终止状态矩阵、拒绝/超时工具、缺 judge 和无断言用例都得到可解释结果，JUnit 不把未评估伪装为业务通过。

### F3 / P1：YAML 与 Python 接口的模型配置脱节

证据：`wuwei/config/yaml_loader.py:33-47`，LLMConfig extra=forbid，temperature 仍为非空 float=0.2，字段没有 prompt_cache_options/key 和 reasoning 等扩展配置。`create_agent_from_config` 会序列化这些默认值后传给网关。

影响：Python 中可用的新功能在 YAML 中报错；即便不填温度，YAML Responses 仍显式发送0.2，与3.2.2新的适配器默认不一致。不能仅靠“升级SDK”解决。

建议：公共 ModelSettings 与 provider-specific settings 分层；保持未知顶层字段严格报错，用受控 provider_options 承载厂商参数。区分 unset、None、具体值，不能用 exclude_none 抹平“用户明确要求省略”与“没有指定”。

验收：同一配置从 Python/YAML 构建的有效请求一致；Responses 省略温度、Chat 旧默认、显式None、新缓存参数均有组合测试。

## 二、架构设计缺口（代码证据与风险，不冒充已复现线上事故）

### A1：会话缓存、持久历史、运行状态需要更清晰的生命周期

`wuwei/agent/base.py:50-85` 将所有会话存在进程内 dict；`create_session` 即使复用同一个 id 也会创建新对象和新 asyncio.Lock。没有统一关闭/淘汰入口。MCP省略session_id时每次都create_session，长期服务仍会保留这些会话。

当前锁能保护同一个进程中的同一个session对象；不能据此承诺多个Agent实例或worker对同一会话串行。SQLite checkpointer 的 claim() 防止重复消费一个恢复点，不等于 session 级并发控制。

建议 SessionStore（load/save/delete/TTL）、SessionHandle 与 run 版本/租约分层。先实现本地持久会话与release_session，不立即要求所有使用者部署数据库。

### A2：有检查点，但不是完整的持久任务运行平台

Memory/SQLite checkpoints 已能持久恢复特定审批流程；后台任务仍是进程内句柄。`run_heads` 以run_id为键，没有跨worker的session lease。检查点写入 schema_version=1，却没有公开的版本拒绝/迁移策略。保存的state、session history、run_state还存在重复序列化。

建议定义可选 RunStore/JobBackend 协议，统一 queued/running/waiting_approval/unknown/completed、查询、取消及外部signal；默认保持轻量本地实现，跨进程需求优先适配一种成熟 durable backend，而非自建队列和调度平台。后端负责lease/heartbeat，业务工具仍需幂等键和对账记录。数据库快照版本化，跨版本迁移显式处理；大正文/文件改引用。先让重启后能查询任务、恢复等待中的审批。不要承诺仅靠本地检查点实现外部副作用 exactly-once。

### A3：模型抽象仍偏“文本+函数调用”，能力边界靠参数透传

`BaseMessage.content` 是str；Responses转换主要处理文本/function_call/reasoning，输出只归一化message文本、函数和reasoning。SDK支持某字段不等于框架能展示其输出、保存状态、计费和恢复。

建议 ModelCapabilities/ModelProfile 标明 native schema、图像/文件输入、工具类别、stream及token参数。按配置和可验证profile处理，不只按模型名字猜测。未知扩展输出保留为typed/provider items而不是静默丢掉。注册provider时随同提供能力和契约测试。

### A4：结构化输出仍是提示JSON+本地校验，没有输出策略

`Agent.run_structured` 已走统一治理，这是优势；但Runner在模型请求前插入JSON Schema提示，未选择provider native schema。原生structured output与提示后解析不是相同保证；拒答/截断需要独立结果语义。

建议 OutputSpec(mode=auto/native/tool/prompted)。能力支持时用原生严格schema，明确选择prompted后备；验证重试继续沿用现有预算/审批/会话生命周期。不再另造旁路运行链。

### A5：中间件字典承担了过多隐式协议

Runner约809行、网关约551行，长度只是信号。实质问题是run_state、permission_state、parent_context、runtime resources、structured_schema等通过字符串键在多个组件传播，变更容易漏掉某条入口。run_plan还自己创建根治理metadata。

建议在保持单一Runner的前提下提取有类型的RunContext、RunState、ModelCallOutcome和ToolOutcome，运行时资源不序列化，持久状态显式版本化。不要拆出第二个“更简单”的Runner。用路径契约矩阵覆盖run/stream/structured/plan/delegation/resume × success/error/cancel/unknown。

### A6：追踪和资源释放还没有形成完整生产契约

TracingMiddleware已有span生命周期管理，但字段主要是llm.content_length、llm.tokens_used等，没有完整的根run、模型/提供商、缓存细分、错误与重试事实，以及明确的脱敏策略。可对接OTel不等于已提供统一可检索的运行追踪。

LocalSandbox._communicate和DockerSandbox._run对TimeoutError清理子进程，但对CancelledError没有对称的finally清理路径；这是取消时的资源风险，尚未在此次评估中复现残留进程。此前GitHubCI存在Event loop is closed子进程析构警告，不能直接断言就是这条路径造成。

建议补齐取消测试、统一Agent/依赖生命周期和OTel映射。内容采集默认关闭，敏感参数脱敏，标准字段版本要固定；GenAI semantic conventions会演进，不能声称无限兼容。

## 三、值得新增的功能与实现次序

| 顺序 | 功能 | 用户能直接得到什么 | 最小验收 |
|---|---|---|---|
| 1 | 类型化工具结果 + unknown对账接口 | 不会因业务JSON误停；能对异常外部动作录入权威结果 | 已成功动作不重放；缺证据仍保持unknown；对账有操作者、证据、审计与并发消费保护 |
| 2 | ModelProfile + OutputSpec + 配置一致性 | 新模型、新缓存和严格结构化输出不再逐入口踩坑 | Python/YAML请求等价；native/prompted与拒答/截断/不支持能力测试 |
| 3 | 类型化ContentPart + ArtifactStore | 图片、PDF和工具生成文件能贯通模型、会话、事件与协议 | 保留旧strAPI；文件引用/媒体类型/大小及归属校验；重启后文件可读；不把大base64塞入每个检查点 |
| 4 | SessionStore + 可选持久RunStore/Worker | 服务重启后任务可查询、审批可恢复、会话可清理 | 双worker抢占、崩溃、lease到期、取消、重复signal、版本迁移演练；未知动作不自动重放 |
| 5 | 真正可当发布门禁的Evals | 比较模型/提示/框架升级是否退化，而不是只有测试数量 | 终止状态、成功工具轨迹、多轮会话、预算/时延、baseline diff及缺judge策略 |
| 6 | 类型化业务依赖 RunContext[Deps] | 数据库、用户身份、HTTPclient不必藏在闭包里，也不会当模型参数暴露 | 注入参数不进入schema/检查点；父子依赖边界、单测替身、生命周期与租户隔离 |
| 7 | 统一运行追踪与诊断CLI | 从run定位慢请求、重试、工具失败和缓存效果 | 根run+子span、异常状态、token/cost estimate、trace id关联，内容默认不采集 |

1-2属于下一轮可靠性底座；3与5最能提升开发者实际价值；4按需要后台长任务的真实使用场景实现。6-7可与底座共用类型设计，避免后补第二套上下文。

## 四、暂不优先做

- 不先做可视化低代码编辑器：目前输出/配置/结果协议仍会变，UI会固化错误抽象。
- 不为追赶数量再加一批模型专用适配器：已有OpenAI兼容入口，优先做能力profile和provider契约套件。
- 不先做实时语音、视频、自治群体：输入输出与持久任务底座尚不完整，扩展面会放大缺口。
- 不为了“企业级”强制Redis/Postgres/Temporal：保持轻量默认，持久worker作为可选能力。
- 不把memory、RAG、技能或MCP列成“缺失功能”：这些已存在；后续按业务样例评估召回质量、权限与互操作。

## 五、研究参照与验收边界

框架对照见同目录 FRAMEWORK_COMPARISON.md；本报告的推荐是结合本地代码与参考设计的工程判断。

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)：用于区分原生schema契约和仅JSON提示/解析。
- [OpenAI Images and vision](https://developers.openai.com/api/docs/guides/images-vision)：多模态消息结构的实际需求。
- [OpenTelemetry GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：追踪映射参照，采用时固定语义版本。

本次做代码路径检查与三个离线复现，没有重跑全套测试来冒充新的框架质量评测。未评估真实模型质量、真实远端协议、生产吞吐或数据库恢复SLA。报告优先级按后果与使用范围排序，不是对“没有某个功能”的统一缺陷判定。
