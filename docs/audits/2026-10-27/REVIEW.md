# Wuwei 当前工作树深度排查（2026-10-27）

## 结论：先收敛执行契约，再扩展功能

当前最严重的问题不是模块数量不够，而是**模块单独能跑，组合后不再遵守同一套执行、权限、会话和恢复契约**。3.2 新增能力的方向值得保留，但目前不能把 durable、deny-by-default、统一成本治理当作已经闭环的生产保证。

建议保留 AgentRunner 单一事件循环、插件注册表、工具并发分组和可组合中间件；优先让普通运行、流式、结构化输出、子代理、计划任务、恢复/分叉共享同一条受治理的执行路径。不要再建立第三套运行内核。

## 本次依据和边界

- 已执行 `git fetch origin` 和 `git pull --ff-only`，结果 `Already up to date`；HEAD 与 origin/master 同为 `889eb7d`。
- 评估对象是 **HEAD 加当前未提交/未跟踪的 3.2 工作树**，不是仅远端发布版。原有改动很多，本次未提交、覆盖或修复业务源码。
- 阅读了 Agent/Runner、同步与后台子代理、图执行与检查点、LLM 网关与适配器、中间件、权限、MCP 动态发现、计划执行、协议出口、会话序列化、相关测试与设计文档。不是逐行穷尽全部后端的安全审计。
- 全量 `pytest -q` 收集失败：三个集成测试文件导入未安装的 `dotenv`。`pyproject.toml` 的 dev 依赖也未声明它。
- 按项目 CI 的离线范围运行：**320 passed、9 skipped、1 warning，4.81s**。未调用真实模型、远程 MCP、Docker/E2B 或向量数据库。
- 新增的本地复现脚本得到 23 条观察记录；部分观察属于同一个根因，不代表 23 个独立缺陷。使用 fake 模型/工具、临时目录，Responses 探针在 SDK 参数检查时失败，没有发出 HTTP 请求。

证据文件：
- [复现脚本](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/reproduce.py)
- [实际观察 JSONL](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/observations.jsonl)
- [配置错误的原始 stderr](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/observations.stderr.txt)

运行：在 `/Users/xiaojia/code/wuwei` 执行 `PYTHONPATH=. .venv/bin/python docs/audits/2026-10-27/reproduce.py`。脚本记录缺陷现象，不是断言修复成功的回归测试。

## 一、优先阻断的权限与恢复问题

### 1. 权限污染标记和工具配额不能跨步骤保留【高风险，已复现】

位置：[Runner:327](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L327)、[Permission:151–202](/Users/xiaojia/code/wuwei/wuwei/security/permissions.py#L151)。

每一步都创建新的 `MiddlewareContext`，权限中间件却把 `untrusted_seen` 和 `_quota_*` 放在这个短命对象的 metadata 中。复现中第一步读取网络内容，第二步写文件，记录是 `[http, write]`，没有按声明冻结后续写入。配额也同样按步骤重新计算。即使同一批工具，所有 before_tool 先执行、所有 after_tool 后执行，也不能用前面网络结果的污染标记约束后面的写入。

**修复方向：**明确 run 级状态容器，跨步骤、检查点、恢复保存治理状态；对有依赖的工具批次，在结果返回后再决定后续动作。

### 2. 权限路径匹配不是目录边界【高风险，已复现】

位置：[Permission:208–216](/Users/xiaojia/code/wuwei/wuwei/security/permissions.py#L208)。

`lstrip('./')` 不会规范化路径，额外的 `p.rstrip('/*') + '*'` 会扩大白名单。复现：白名单 `out/**` 同时放行 `out/../../secret.txt` 和 `outside/secret.txt`。内置 file 插件另有 workspace 根目录约束，但它不能保证 `out` 子目录权限；自定义工具也不能依赖这个额外约束。

此外未分类工具归为 `other` 后默认放行，与 README 的 deny-by-default 不一致。

**修复方向：**基于固定根目录、规范化后的目标路径检查真实目录边界；工具能力应显式声明，未知能力的默认策略要清晰。

### 3. 多工具审批会丢调用，重复 resume 会重复执行【高风险，已复现】

位置：[Runner:559–576](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L559)、[Runner:454–469](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L454)、[resume:140](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L140)。

模型一次产生 `a、b` 两个调用，在 `a` 审批时挂起。保存的 pending 只有 `a`，恢复后只执行 `a`，`b` 没有结果。若中断前已经通过其他调用的过滤，其待执行队列/中间件结果也未一起保存。

同一个 interrupt 再 resume 一次，`a` 再执行一次。没有消费标记或防重入状态。涉及部署、发送、写入时，这是实际副作用重复风险，不只是多花一次 token。

**修复方向：**持久化完整工具批次和每个调用状态；审批与执行状态分离；恢复领取需要原子防重。外部工具已成功但结果尚未落库的崩溃窗口还需要幂等键或 unknown 状态，不能只靠重放。

### 4. 恢复把“审批通过”扩大为“跳过全部工具中间件”【高风险，源码确认】

位置：[Runner:305–316](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L305)、[Runner:563–565](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L563)。

恢复使用 `skip_before_tool=True`，跳过的不只是 HITL，还包括权限检查、参数改写和子代理拦截。把 Permission 放在 HITL 后面时，初次运行尚未走到权限检查，恢复就直接执行。

检查点也只保存消息、step 和少量 metadata，没有累计 usage、治理状态、完整会话配置；新进程 `Agent.resume()` 先查内存 `_sessions`，调用者必须自行补建 session。README 所说“崩溃后恢复”目前更接近有限的审批续跑。

**修复方向：**只消费指定调用的审批决定，其余治理照常执行；明确完整持久化 RunState 和重建会话入口。

### 5. YAML 中间件配置出错会降级为无审批 Agent【高风险，已复现】

位置：[yaml_loader:153–157](/Users/xiaojia/code/wuwei/wuwei/config/yaml_loader.py#L153)。

配置 `hitl.policy=typo`，构造 HITL 抛异常被捕获后只记日志，最终成功创建 `middleware_count=0` 的 Agent。安全组件缺失不能当作非关键扩展失败继续启动。

**修复方向：**配置/安全组件构造错误应直接启动失败；显式 optional 扩展才能选择跳过。

## 二、已经存在的功能性 Bug

### 6. Responses API 默认配置不可调用【阻断该入口，已复现】

位置：[gateway:83–85](/Users/xiaojia/code/wuwei/wuwei/llm/gateway.py#L83)、[responses:33、104](/Users/xiaojia/code/wuwei/wuwei/llm/adapters/responses.py#L33)。

网关统一传 `max_tokens=4096`，ResponsesAdapter 却需要 `max_output_tokens`。前者进入 `default_params`，最终和后者一起传给 SDK。实际错误：`AsyncResponses.create() got an unexpected keyword argument 'max_tokens'`。fallback 的 Responses 构造也有相同映射问题。

**修复方向：**在 provider 边界做显式参数映射；测试从 LLMGateway 到 SDK 调用签名，不只单测 Adapter。

### 7. 中间件返回 ToolMessage 后，栈仍把它当 ToolCall 传下去【已复现】

位置：[stack:122–126](/Users/xiaojia/code/wuwei/wuwei/middleware/stack.py#L122)。

HITL 拒绝返回 ToolMessage，后接 Permission 访问 `.function`，直接 AttributeError。Runner 支持的新三态返回协议，Stack 仍是旧的 ToolCall/None 协议；子代理直接返回结果也会遇到同一类组合问题。

**修复方向：**把 before_tool 的执行/完成/挂起语义定义成统一契约；完成结果应停止后续 before_tool 并进入结果阶段。明确安全中间件和实际执行者的顺序，不能仅凭调换顺序掩盖漏洞。

### 8. 动态发现工具已经注册，但当前运行看不到 schema【已复现】

位置：[Agent.create_runner:151–161](/Users/xiaojia/code/wuwei/wuwei/agent/agent.py#L151)、[Runner:101、356](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L101)、[MCP discovery:119–155](/Users/xiaojia/code/wuwei/wuwei/plugin/builtin/mcp.py#L119)。

Runner 在运行开始时复制工具列表，MCP 却在 before_llm 才注册 tool_search。复现：首轮带 MCP 概览但 tools 为空，下一轮 tools 才有 tool_search，而概览又消失。运行中注册 new_tool 后，下一步 schema 仍只有 discover。

**修复方向：**每次模型调用时从 registry/当前能力视图取得有效工具；发现缓存与每次调用的上下文注入分开管理。

### 9. 流式入口和 AG-UI 的会话连续性断裂【已复现】

位置：[Agent:179–188](/Users/xiaojia/code/wuwei/wuwei/agent/agent.py#L179)、[ag_ui.ag_ui_events](/Users/xiaojia/code/wuwei/wuwei/serve/ag_ui.py)。

普通 `run()` 复用 `_default_session`，`stream_events()` 不传 session_id 时另建 session。同一个 Agent 先 run 再 stream，第二次只看到新的用户问题。AG-UI 的 thread_id 只用于输出事件，没有传给 Agent 会话；同一 thread 连续两次请求仍创建两个会话，只取最后一条用户消息进一步丢掉客户端传来的历史。

**修复方向：**统一默认会话规则；协议 thread_id 映射明确的 AgentSession，并在服务端约束同会话并发和归属。

### 10. 图的 reducer 仅非流式生效【已复现】

位置：[graph.invoke:191–198](/Users/xiaojia/code/wuwei/wuwei/graph/graph.py#L191)、[graph.stream:260–271](/Users/xiaojia/code/wuwei/wuwei/graph/graph.py#L260)。

同一个节点返回 `{'value': 42}`：invoke 正常归并到 metadata；stream 把 dict 直接当 State，之后 `.step` 报 AttributeError。AgentRunner 虽统一了流式与非流式，图层的双轨漂移仍存在。

**修复方向：**节点调用和状态归并共用一段逻辑，只让事件输出形式分流。

### 11. Anthropic 流式输入 token 统计丢失【已复现解析器】

位置：[anthropic:223–249](/Users/xiaojia/code/wuwei/wuwei/llm/adapters/anthropic.py#L223)、[gateway 流式 usage 合并](/Users/xiaojia/code/wuwei/wuwei/llm/gateway.py#L342)。

message_start 的 usage 位于 `event.message.usage`，当前读 `event.usage`，100 input 被解析成 0；后续 20 output 最终仅报告 total=20。即使修正起始读取，按字段 max 合并两个分离的 input/output subtotal 也不能保证 total=input+output。

另外 build_request 对多条 system 消息只保留最后一条，RAG/Memory 等注入消息可能覆盖其他系统约束（源码确认）。

**修复方向：**适配器统一事件语义；聚合后重新计算 total，保留全部有效 system 指令。

### 12. 存储和会话序列化仍然不闭环【已复现】

位置：[storage:32–40](/Users/xiaojia/code/wuwei/wuwei/middleware/storage.py#L32)、[Runner:327](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L327)、[session:44–55](/Users/xiaojia/code/wuwei/wuwei/agent/session.py#L44)、[action_log:130–135](/Users/xiaojia/code/wuwei/wuwei/runtime/action_log.py#L130)。

- Storage 从 ctx 读取 session_id，但 Runner 从未注入；会话 A、B 均写进 `default` 目录。
- 正常运行后 metadata 包含 ActionLog 对象；`json.dumps(session.to_dict())` 报 `ActionLog is not JSON serializable`。
- SQLite 检查点同秒写入 3 条，分页第一页取 1 条后下一页得到 0 条，因为只用秒级 created_at 做严格小于游标。

**修复方向：**运行上下文统一带 session_id/run_id；持久数据与运行对象分离；分页按时间+唯一 ID 稳定排序。

### 13. 账本召回工具 schema 缺参数【已复现】

位置：[ledger:205–208](/Users/xiaojia/code/wuwei/wuwei/middleware/ledger.py#L205)。

read_context、pin_context 的 Python handler 必须接收 entry_id，但给模型的 properties/required 都为空。内容虽归档成功，模型没有正确的参数契约召回它。Code Mode 的 run_code 同样直接使用空 ToolParameters，未声明必需的 code 参数（源码确认）。

**修复方向：**统一通过带类型签名的工具构造器生成 schema，并测试模型可见参数与 handler 签名一致。

## 三、设计脱节与错误的成功语义

### 14. “单一内核”尚未覆盖子代理，安全和成本会绕开【已复现】

位置：[sub_agent:216–330](/Users/xiaojia/code/wuwei/wuwei/agent/sub_agent.py#L216)、[async_sub_agent:371–419](/Users/xiaojia/code/wuwei/wuwei/agent/async_sub_agent.py#L371)。

同步子代理仍自行搭建 LLM/Tool 状态图；后台子代理复用了它，但没有复用 AgentRunner。SubAgent 文档称继承父 middleware，实际只加载 sub_agent.middleware，也不走 wrap_model_call、统一 usage/checkpointer/EventBus。

复现：父 Permission 明确 denied_tools=['danger']，通过 task_worker 委派后子工具 danger 仍执行；实际 4 次 LLM 调用，父结果仅报告 2 次。当前子循环把 ToolMessage 拒绝结果继续交给 executor，新增的拒绝回喂契约也没有接入。

**修复方向：**子代理使用同一 Runner，继承权限/预算的作用域应显式；子代理可以收紧而不能无意丢掉父约束，统计统一回传。

### 15. run_structured 是另一个裸 LLM 入口【已复现】

位置：[Agent:216–276](/Users/xiaojia/code/wuwei/wuwei/agent/agent.py#L216)。

创建/选择了 session 却不用，构造独立消息直接 llm.generate，绕过系统提示、会话历史、工具、中间件预算、检查点与运行统计。实际返回 `(model, LLMResponse)`，不是 docstring 的 `(model, AgentRunResult)`；运行后 session 历史仍只有 system。

**修复方向：**结构化输出是正常运行的输出约束和校验重试策略，不应独立绕过治理链路。

### 16. 未完成被报告成功，计划执行器无法信任结果【已复现】

位置：[Runner:496–514](/Users/xiaojia/code/wuwei/wuwei/runtime/agent_runner.py#L496)、[PlanExecutor:96–122、178–186](/Users/xiaojia/code/wuwei/wuwei/planning/executor.py#L96)。

达到 max_steps 输出“任务未完成”，status 仍 completed。复现 max_steps_per_task=1，计划 task.status=completed，result 却是“任务未完成，已达到最大步骤限制。”，下游可以把未完成当成有效输入。Executor 也未验证 DAG：循环依赖直接返回全部 pending，没有错误。

审批 interrupted 被当普通失败重试，丢掉恢复 payload；可能再次请求模型/审批。`run_plan(session=...)` 明确忽略 session，规划调用的 usage 也没有合入汇总（源码确认）。

**修复方向：**完成、步数耗尽、失败、挂起分别表示；DAG 执行前验证；挂起应保留可恢复任务，不进入自动失败重试；规划和执行统一成本归因。

### 17. fork 实际回滚原会话并重做副作用，不是真分支【已复现】

位置：[debug:150–173](/Users/xiaojia/code/wuwei/wuwei/debug.py#L150)。

fork 优先取原 session，resume 替换原上下文并沿用旧 run_id。复现 fork 前后 session 总数仍是 1，原历史改变，deploy 执行次数从 1 变 2。`apply_overrides()` 修改 pending 参数却不修改历史 assistant.tool_calls，模型可见记录也可能与实际执行不一致（源码确认）。

**修复方向：**新 session/new run、parent lineage 与重放策略；副作用工具默认 dry-run/mock 或单独授权，不能把“调试”直接等同于线上再次执行。

### 18. 账本的“全量真相”与淘汰策略冲突【源码确认】

位置：[ledger:74–108](/Users/xiaojia/code/wuwei/wuwei/middleware/ledger.py#L74)。

默认内存账本超过 max_entries 会删除原文，而上下文还可能留着它的引用；持久化模式 get 不会从磁盘补读被淘汰条目；pin 没有把 pinned 状态重新写入磁盘，重启后可能失效。README/模块说明里的“任何时刻完整还原”“钉住永不淘汰”超出当前实现。

**修复方向：**区分持久源数据、内存缓存与窗口引用；明确定义保留期和失效反馈，而不是用缓存冒充保真存储。

## 四、文档、测试和产品边界已经落后

### 文档不能作为当前 API 的可靠入口

[README](/Users/xiaojia/code/wuwei/README.md) 中可直接定位到：
- graph 示例导入 `END`、`wuwei.graph.channels`，当前包不导出 END，channels 目录也已删除。
- PluginLoader/PluginRegistry 示例对应已替换的插件系统。
- 多 Agent 示例定义 team 后调用 swarm；变量不存在。
- 列出 graph/middleware/memory/sandbox extras，但 pyproject 不再声明；“all”也不含新增 sandbox/vectorstore 等全部扩展。
- “6 个适配器”“72 个测试”等结构说明与 3.2 工作树不同。
- 若干中间件示例使用 `MiddlewareStack(Middleware(...))`，当前构造器不接受参数。

[OPTIMIZATION_PLAN.md](/Users/xiaojia/code/wuwei/OPTIMIZATION_PLAN.md) 保留的是较早诊断：旧适配器故障、AgentRunner 无测试、并行工具死代码、没有 PlanExecutor 等结论已经不适用于当前工作树。应把历史诊断、已实现内容、未验收缺口分开，避免继续按旧故障清单开发。

### 测试成功主要证明模块局部，缺少跨模块契约

现有 kernel 测试的 interrupt 主要是单工具 happy path；测试甚至接受没有 checkpointer 却返回空 checkpoint_id 的 interrupted 状态。没有覆盖多工具、重复恢复、跨步权限、HITL+Permission、真实 SDK 参数、动态工具 schema 刷新、同线程连续请求、普通/流式图 reducer 一致性等关键组合。

CI 的 `ruff --exit-zero` 不具备 lint 阻断能力；离线 CI 范围明确排除了三个集成测试，不能据此推导真实模型、MCP、沙箱、数据库或协议互操作已经验收。

## 建议实施顺序与验收

### 第一批：阻断危险结果和直接不可用入口

1. before_tool 结果协议贯通 Stack/Runner；多工具中断完整状态、防重复恢复、恢复时继续权限检查。
2. run 级权限状态与规范路径边界；安全配置构造失败应终止启动。
3. 修正 Responses 参数映射、MCP 工具视图刷新、最大步数终止状态。

**验收：**HITL+Permission 可组合；两工具审批不丢不重；重复恢复不再次执行副作用；网络后写入受限、配额跨步有效；错误配置无法启动；Responses SDK 接受默认参数；发现后下一步模型收到工具 schema；计划不再将耗尽判为成功。

### 第二批：真正统一内核与会话

1. 子代理和 run_structured 接入共享生命周期；父子权限/预算/统计作用域明确。
2. run/stream/AG-UI 会话映射统一；Storage/session 序列化闭环。
3. 持久化完整 RunState、恢复入口、fork 新分支与副作用策略。

**验收：**父子调用可归因总费用；同一线程跨请求连续；重建进程后可恢复；fork 不修改原会话，调试默认不执行真实副作用。

### 第三批：一致性、保真和发布质量

修复图 reducer 流式差异、Anthropic usage/system 合并、SQLite 分页、账本 schema/保留策略；将 README 示例变为可执行的无网络 smoke 测试，划分核心离线测试与在线/协议兼容验收。

此阶段优先完成纵向闭环，不建议继续增加新模型、新协议、新向量后端。新增能力应以“进入同一治理链路并通过契约测试”为条件，而不以“增加了一个类和构造测试”为完成标准。
