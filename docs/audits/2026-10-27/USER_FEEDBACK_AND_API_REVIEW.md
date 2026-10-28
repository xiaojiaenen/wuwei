# 下游用户反馈复核、API 时效性与架构取舍

日期：2026-10-27。对象：Wuwei 当前本地 3.2 工作树，HEAD `889eb7d` 加未提交改动。

## 结论

下游反馈说明的是用户实际接入框架时付出的兼容成本，应当优先解决。当前重构已经修了一部分，但不能拿原始 3.0.2 故障总评当作当前结果，也不能因为增加了模块就宣布架构闭环。

**推荐方向：保留轻量自有运行内核，收敛为统一、类型化、可恢复的执行契约；向成熟框架学习状态与边界，不复制其全部产品面。** 先修用户主路径和兼容层，再做新能力。Chat Completions 与国内兼容端点继续保留；Responses 作为独立 provider；两者都经过同一治理内核。

关联材料：
- 原始用户反馈：[AgenticOS 的 3.0.2 问题清单](/Users/xiaojia/code/AgenticOS/docs/wuwei-3.0.2-问题清单.md)。该文档作为审查证据，本次未改动其内容，也没有把它的补丁/PR 建议视为执行授权。
- 上轮发现：[当前工作树深度排查](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/REVIEW.md)。
- 本轮探针：[feedback_probes.py](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/feedback_probes.py)、[实际输出](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/feedback_observations.jsonl)。
- 官方框架比较：[FRAMEWORK_REFERENCES.md](/Users/xiaojia/code/wuwei/docs/audits/2026-10-27/FRAMEWORK_REFERENCES.md)。

## 1. 用户 3.0.2 反馈逐项核对

“已修”仅表示当前源码/本地探针对应的具体问题已解决，不代表线上提供商全部验收。“源码确认”与动态复现分开列示。

| 用户编号 | 当前判断 | 当前代码与证据 |
|---|---|---|
| P0-1 四家流式未传 stream | 原故障已消除，兼容迁移待补 | Anthropic 已写入 stream；zhipu/dashscope/ollama 专用 adapter 已删除，统一走 OpenAI-compatible。旧 provider 字符串目前报不支持，不能称无缝修复。 |
| P0-2 三家 parse_response 必崩 | 原路径删除，根因仍残留 | 删除的 adapter 不再构造错误响应，但 core.types 与 llm.types 仍有两份不同 LLMResponse，且 wuwei.core 仍导出旧类型；探针确认不是同一类。 |
| P0-3 OpenAI 流式 usage=0 | 对 OpenAI 已修 | 使用真实 SDK ChatCompletionChunk 构造 usage-only 帧，正确得到 10/5/15。Anthropic 输入 usage 漏计是另一个仍存在的问题，见上轮报告。 |
| P0-4 finish_reason 严格 Literal | 仍存在 | 真实 SDK ChatCompletion 构造 finish_reason=eos，框架解析 ValidationError。应保留 raw reason，并区分成功/截断/中止/未知。 |
| P1-1 流式无统一重试/fallback | 仍存在 | fake adapter 故障：primary=1、fallback=0。SDK 可能在请求建立阶段自行重试，但网关没有声明的统一流式恢复策略。 |
| P1-2 字符串分类重试 | 分类方式已修，策略仍不完整 | status_code=401、正文含 :5000，现判 fatal。未实现 Retry-After；还需区分可重试状态和参数/解析错误。 |
| P1-3 确定性错误也重试 | 仍存在 | TypeError 被 primary 调用3次，再调用 fallback 1次；不应对本地程序错误、ValidationError 重放模型请求。 |
| P1-4 客户端参数摊进请求 | 仍存在 | max_retries=0 出现在请求 dict，SDK 签名校验 TypeError；真实 client.max_retries 仍为2。 |
| P1-5 工具调用静默丢弃/空参 | 两处均仍存在 | tool delta 完整但 finish=stop，所有 chunk.tool_calls_complete 均 None；参数非法 JSON 被解析为 {}。修复时不能不看终止原因就执行残缺调用。 |
| P1-6 usage 缺失导致崩溃 | 对 OpenAI 已修 | 真实 SDK 响应 usage=None，正确解析文本与0 usage；不过“缺失”和“真实0消耗”当前没有区分。 |
| P1-7 Anthropic 参数碎片当正文 | 已修 | input_json_delta 进入 tool_calls_delta，不再进入 content；已有适配器契约测试。 |
| P1-8 Anthropic/网关 usage 契约 | 部分修复 | 网关已只读适配器 data.usage，但 message_start 读错嵌套层级、分离 input/output 的 total 聚合仍错。 |
| P1-9 连续 user 必然400 | 原结论需要修正 | 官方 SDK 说明连续同角色消息会合并；不能把“连续 user”本身判成必然 roles must alternate。parallel tool results 仍应按官方规则成组、紧邻对应调用，不能丢任何结果。详见官方比较文档。 |
| P1-10 截断误报 stop | 部分修复 | Anthropic 流式 max_tokens 已映射 length；非流式真实 SDK Message(stop_reason=max_tokens) 仍被映射 stop。 |
| P1-11 Anthropic 无 base_url | 仍存在且入口报错 | 网关现会传 base_url，但 Adapter 构造签名不收，实际 TypeError。 |
| P2-1 客户端无关闭入口 | 仍存在 | 网关没有 aclose/__aenter__，缺乏共享客户端所有权约定；不能以“必然泄漏”替代负载证据，但显式生命周期缺失已经确认。 |
| P2-2 工厂硬编码/无注入 | 部分改善 | 内部 if/else 改成私有注册表，但没有公开 register_provider/adapter_factory/client 注入；下游仍需替换 gateway.adapter。 |
| P2-3 fallback 只拿 adapter | 仍存在 | set_fallback 丢掉备用 gateway 自己的 timeout/retry/策略；流式也不用 fallback。 |
| P2-4 retry 配置易 KeyError | KeyError 已修，校验不完整 | .get(max_attempts,3) 已兜底；retry 仍是自由 dict，无清晰配置模型、整体deadline/jitter策略。 |
| P2-5 流式无整体超时 | 仍存在 | 配置timeout=.01s，fake流不产出；直到外层.05s取消才结束。真实SDK默认read timeout不等于框架整体deadline。 |
| P2-6 llm_end 延迟累计 | 仍存在 | Runner:393累计后，llm_end继续发送total_latency_ms。字段语义应明确；新增step_latency_ms，别静默改变下游已依赖字段。 |
| P2-7 chunk 无类型契约 | 仍存在 | ABC 只返回 dict[str,Any]；LLMResponseChunk.tool_calls_delta 还混合 list/str。应使用类型化增量事件，adapter只做一次规范化。 |
| P2-8 content 仅 str | 属于能力边界，不是已证实的Chat输出Bug | 当前安装官方SDK的 ChatCompletionMessage.content 与 ChoiceDelta.content 均是 Optional[str]。多模态输入/content block、Responses items未完整表达是扩展能力缺口，不能据此声称正常官方Chat流必崩。 |
| P2-9 两套消息序列化 | 仍存在 | BaseMessage.to_openai 只保留role/content，带tool_calls的基础消息会丢字段；Adapter另写一套转换。专用adapter删除减轻了旧触发路径，未消除双契约。 |
| P2-10 reasoning无条件回传 | 可移植性风险仍在 | 探针确认发回 provider-specific reasoning_content。不能从DeepSeek可用推断所有端点都支持，也未在线证明官方端点必400；应通过provider capability决定序列化。 |

源码入口：[LLM 网关](/Users/xiaojia/code/wuwei/wuwei/llm/gateway.py)、[OpenAIAdapter](/Users/xiaojia/code/wuwei/wuwei/llm/adapters/openai.py)、[AnthropicAdapter](/Users/xiaojia/code/wuwei/wuwei/llm/adapters/anthropic.py)、[LLM 类型](/Users/xiaojia/code/wuwei/wuwei/llm/types.py)、[旧类型](/Users/xiaojia/code/wuwei/wuwei/core/types.py)。

### 本轮额外发现：OpenAI 生成配置被静默忽略

OpenAIAdapter.__init__ 接收 temperature/max_tokens，却没有保存或放入 default_params。用 temperature=.73、max_tokens=17 构造后，build_request 中两者都不存在。实际模型行为、成本与用户配置不一致。这是功能Bug，不是“API已经弃用”的同义词。

更具体的工具 schema 问题：[ToolRegistry](/Users/xiaojia/code/wuwei/wuwei/tools/registry.py) 把所有带 __args__ 的类型当作 Union 处理，`list[str]` 被错误生成为 `type:string`。因此应使用Pydantic/TypeAdapter或成熟schema生成器，不继续扩展手写类型名映射。

## 2. 哪些 API 真过时，哪些只是实现错误

### 2.1 Chat Completions 还受支持，不应强制删除

OpenAI 官方说明 Chat Completions 仍支持，Responses 推荐给新项目。**保留兼容模型路径、修复 Responses 独立路径**比“一刀切迁移”更适合 Wuwei 的国内模型定位。两者消息/工具/流事件不同，不应只换 endpoint。[官方迁移说明](https://developers.openai.com/api/docs/guides/migrate-to-responses)

### 2.2 max_tokens 的弃用必须限定 provider

OpenAI Chat Completions 的 max_tokens 已弃用，推荐 max_completion_tokens，且有 reasoning 模型兼容限制；Responses 用 max_output_tokens。Anthropic 与部分兼容端点仍使用自己的 max_tokens 参数，不能全仓字符串替换。

建议公开层使用中立的输出预算，适配器依据capability映射；原 max_tokens 参数保持明确的兼容入口/弃用周期。[OpenAI Python API参考](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)

### 2.3 SDK 客户端参数/重试/关闭不是可随意透传的模型字段

本地安装：openai 3.23.0、anthropic 1.11.0、mcp 2.2.0、pydantic 2.13.5。它们是本次运行版本，不是本报告承诺的全版本支持范围。

OpenAI SDK 默认自动重试2次，流消费中断不自动重放；支持显式客户端配置和关闭。Wuwei自己的3次重试与SDK叠加，在相应错误条件下可能变成最多9次底层尝试，不能只看网关计数宣称调用3次。

应由一个层负责重试，并提供client/http_client注入与owned/borrowed生命周期。网关超时应覆盖整体deadline，而不仅一次调用。[官方SDK重试与生命周期说明](https://github.com/openai/openai-python#retries)

依赖下限也存在真实脱节：pyproject仍声明openai>=1.0.0，但官方在1.66.0才引入/v1/responses。当前对外提供Responses入口，却允许安装不包含该入口的旧SDK。应按实际使用功能建立最低版本回归矩阵后提高依赖下限；1.66.0只是Responses引入版本，不等于本实现所有功能的已验证最低版本。[官方1.66.0发布记录](https://github.com/openai/openai-python/releases/tag/v1.66.0)

### 2.4 MCP 2.x 的迁移已有进展，但 SSE 说明不正确

当前客户端使用官方SDK、initialize握手、2.x float timeout、httpx2和2元组stream；MCPServer出口也用新类名。不能套用旧文章再把 timedelta、三元组或 FastMCP 1.x写回去。

但配置 transport='sse' 被直接映射为Streamable HTTP，并声称官方已将SSE合入。**Streamable HTTP可以承载SSE响应，不等于兼容旧HTTP+SSE握手端点**。官方2.x迁移指南仍列有独立sse_client。应真正适配legacy transport，或明确拒绝并给迁移错误，不能静默改传输语义。

另需统一mcp extra与serve extra的版本支持范围，目前前者>=2.2、后者>=2.2,<3，缺少一致支持边界。[MCP官方迁移指南](https://py.sdk.modelcontextprotocol.io/migration/)

### 2.5 Pydantic 的旧 class Config 是已弃用写法

当前运行已报告 `PydanticDeprecatedSince20`，来自 streaming/types.py 的class Config；应改model_config/ConfigDict。优先级低于执行/权限Bug，但属于有明确证据的API清理项。

### 2.6 Wuwei 自身公开 API 已漂移，不能让用户承担迁移猜谜

README中的graph.channels、PluginLoader/PluginRegistry、未导出END等已不对应当前代码。旧provider别名删除、两份LLMResponse并存、Agent.run与stream_events默认会话规则不同，不是外部SDK过时，而是Wuwei自己的版本契约缺失。

应保留可转换旧入口、显式DeprecationWarning、迁移示例和破坏性版本说明。尤其不要突然删除下游依赖的gateway.adapter公开替换方式，应先提供更稳定的client/adapter注入点。

### 2.7 A2A 宣称的版本与实际方法不匹配，已本地复现

当前实现接受旧的 tasks/send，却声明0.3/1.0兼容。对卡片默认根URL发送标准JSON-RPC请求：0.3的message/send与1.0的SendMessage均得到404、-32601 method not supported，旧tasks/send反而200；所以是方法契约错误，不是根URL缺路由。1.0 supportedInterfaces还使用transport而非规定的protocolBinding。

这是比内部命名旧更直接的协议互操作阻断。应选择目标版本，用官方请求/响应模型实现，再独立提供旧入口适配；不能把不同版本字段混在一个dict里就宣称兼容。[A2A 0.3规范](https://a2a-protocol.org/v0.3.0/specification/)、[A2A 1.0规范](https://a2a-protocol.org/v1.0.0/specification/)

AG-UI目前已确认的是会话thread映射问题，未执行官方客户端端到端兼容验收，不把“未验证”写成“已经不兼容”。

## 3. 架构是否合理：保留骨架，改掉四个根因

### 合理且应保留

- Agent门面 + Session + Runner 分层：调用者不需要直接操纵执行循环。
- async-first事件流，以及非流式由事件聚合得到结果：方向正确。
- Plugin注册入口、工具执行策略与并发安全分组：可以承载轻量扩展。
- 可选检查点与中间件：适合按需求组合，不必强制每个聊天请求都部署工作流平台。

### 根因 A：同一能力有多套真相

LLMResponse两份、消息序列化两份、AgentRunner与子代理循环两份、graph invoke/stream归并两份。修复一条路径不会自动修好另一条。

**决策：**内核只有一份消息/结果/工具调用语义。图调度可以独立存在，但节点执行归并与模型/工具调用仍复用统一服务；子代理和structured output不能绕过治理。

### 根因 B：状态作用域没有建模

run/session/step、父子任务、持久化/临时对象全混在metadata中。权限跨步丢失、Storage不知道session、ActionLog破坏JSON、resume清零成本，都是同一根因。

**决策：**明确 SessionState（会话历史/归属）、RunState（run_id/预算/安全状态/终止原因）、StepContext（本次模型调用/工具批次）和持久化快照。共享的是明确状态接口，不是随手约定字符串键。

### 根因 C：执行控制协议隐含在返回类型和异常里

before_tool 可能返回ToolCall、ToolMessage、None或抛Interrupt；Stack、Runner、子代理对这些含义不一致。把“审批通过”解释成“所有中间件都跳过”是最危险的例子。

**决策：**类型化Execute/Reject/Handled/Suspend；审批决定绑定tool_call_id和参数版本，权限仍检查。外部副作用由状态机/幂等约束，不由“发生异常就重试”处理。

### 根因 D：以能力标签代替可验收协议

有checkpoint不等于durable；有JSON端点不等于A2A兼容；有成本中间件不等于覆盖子代理与planner；有schema不等于handler参数正确。

**决策：**每项卖点对应一条跨模块验收链路和版本矩阵。文档由实际可执行入口支撑，删除“业界无开源对标”等无法由当前证据支持的自我定位。

## 4. 用户补丁：哪些吸收，哪些不能原样合

| 反馈建议 | 采纳判断 |
|---|---|
| C-1 usage-only帧 | 思路已进入当前OpenAI适配器；保留真实SDK形状回归。不要再把网关写成读取某一家原始chunk。 |
| C-2 tool收尾纠正 | 修复真实问题，但不要在共享adapter实例保存 _stream_saw_tool_calls，多个并发请求会串状态。聚合状态放在单次stream局部变量；正常终止时校验完整调用，length/abort/断连不能盲目执行。 |
| C-3 finish_reason归一化 | 接受保留raw reason/统一终止原因；不接受将abort或未知一律映射stop，失败会被伪装成成功。 |
| C-4 缺字段兜底 | 接受可选usage兜底，区分unknown与0。必需的choices/调用完整性缺失应返回明确协议错误，不能将所有坏响应都“兜底成功”。 |
| C-5 未吐字才重试 | 方向合理，边界应为“尚未提交任何可观察输出/动作”而不只是文本；按错误白名单和总deadline执行。中途失败应显式结束，不能重播整轮。 |
| C-6 客户端/请求参数分离 | 应优先吸收；client可注入、SDK重试所有权明确、资源所有权明确，才能覆盖下游自建认证网关而不掺其Cookie业务。 |
| C-7 工具并发 | 当前已有分批执行；仍需统一钩子和实时事件语义。不能结果全部执行完后才发tool_start，用户会看到虚假的开始时间。 |
| C-8 分档审批 | 产品语义值得吸收，参数级读写能力也合理；full不应等价于取消组织级权限。先确定会话授权作用域和安全优先级，再增加模式。 |
| C-9 截断 | 需要结构化结果/原文引用/截断标记；不要用“含任意SVG就整体跳过”作为全局豁免，容易让大结果绕过限制。 |
| C-10 thinking兼容 | 用capability表达支持范围，保留provider要求的完整reasoning items/签名；不能全局丢弃或无条件回传。 |

## 5. 推荐落地顺序与迁移验收

### A. 先交付用户可感知的兼容修复

SDK客户端/生成参数分离，修复temperature/max_tokens丢失和Responses映射；确定性错误不重试；流式首个可观察输出前的有限重试/降级与整体deadline；stop+完整工具调用、未知finish reason、usage-only/usage=None的真实SDK响应测试。

验收不依赖线上密钥：使用SDK typed models、MockTransport和fake clock捕获实际请求、重试次数及事件序列。真实提供商验收另列，不把fake测试当作全模型兼容结论。

### B. 收敛运行内核和治理状态

贯通before_tool协议，子代理/structured/plan接入同一运行服务；run/session/step状态明确；多工具审批恢复、防重/unknown、最大步数终止状态与成本归因闭环。这个阶段优先于增加新provider/协议。

### C. 稳定公开API和协议出口

唯一响应模型加兼容转换；provider/client注入；统一close；补旧provider迁移与可运行文档；锁定MCP/A2A/AG-UI支持版本并用官方客户端做互操作测试。支持不了的特性明确拒绝，不靠拼接不同版本字段伪装兼容。

### D. 是否引入成熟工作流引擎

当前建议不立即重写成LangGraph/Pydantic AI包装层：那会同时改变Wuwei语义和下游扩展点，而眼下问题已有明确边界修复路径。若真实产品要求跨进程长任务、并行分支恢复、工作流升级和任务租约，则在稳定Runner之下评估成熟durable backend；以故障恢复测试和维护成本作决策，而不是以框架名气作决策。

本次只增加评估材料与离线探针，未执行用户清单里的补丁合并/PR方案。
