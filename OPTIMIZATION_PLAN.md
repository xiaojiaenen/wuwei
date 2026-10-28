> 历史设计记录：本文保留当时的设计与诊断，不代表当前实现。当前用法见 [README](README.md)，修复验收见 [状态清单](docs/audits/2026-10-27/FIX_STATUS.md)。

# Wuwei Agent 框架深度优化方案

> 版本 v1.0 · 2026-10-02
> 依据：① 全量源码审查（`wuwei/` 约 90 个 Python 文件、约 11,500 行，逐模块阅读）
> ② 2025–2026 主流 Agent 框架调研（LangGraph/Claude Agent SDK/OpenAI Agents SDK/MAF/CrewAI/ADK/Pydantic AI/smolagents/Mastra/Dify/AgentScope/Eino 等 20+）
> ③ 开发者痛点调研（GitHub issues / HN / Reddit / LangChain State of Agent Engineering 2025 等）

---

## 0. TL;DR

wuwei 当前是一个**架构蓝图优秀、方向正确，但生产可用性不足**的框架。三个最能说明问题的数字：

1. **5 个 LLM 适配器中只有 1 个（OpenAI）能完整工作**——zhipu/dashscope/ollama 的 `parse_response` 一调即抛 ValidationError，Anthropic 流式从未真正实现（`adapters/zhipu.py:66`、`adapters/dashscope.py:66`、`adapters/ollama.py:65`、`adapters/anthropic.py:78-91`）。
2. **对外宣称的能力约 1/3 是死代码或假能力**——并行工具执行（`ToolExecutor.execute(concurrent=True)` 全库零调用）、graph channels（约 440 行从未被 import）、checkpointer 恢复（只存不取）、Plan-and-Execute（有 plan 无 execute）、parsers（与主链路不通）、SSE transport（与 http 同一实现）、Docker/E2B 沙箱（只有 `create_subprocess_shell`）。
3. **最复杂的执行循环（`runtime/agent_runner.py`，706 行）零测试**，因此 `run(stream=True)` 调用不存在的方法直接 AttributeError、流式模式子代理整体失效这类炸弹级 bug 得以存活至今。

本方案给出一条五阶段路线：**修复地基 → 统一内核 → 补齐 Harness → 生产化 → 创新突围**，并明确差异化定位——不与大厂拼生态数量，做**「对国内模型生态最友好、生产化三件套（Durable / 成本治理 / 权限沙箱）内建、开放标准全对齐」的薄内核框架**。

---

## 1. 现状诊断

### 1.1 框架底子（先说好话）

经过 v1→v2→v3 三轮重构，框架的**骨架选择是对的**，这也是值得继续投入的理由：

- **中间件系统是全库最精致的部分**：6 个钩子点 + `wrap_model_call` 洋葱模型 + 预分拣优化（`middleware/stack.py:50-65`，子类未覆写的钩子零开销）。这与 LangChain 1.0 把 middleware 作为 `create_agent` 的灵魂特性是同一方向。
- **Plugin 系统（v3.0）概念正确**：PluginContext 聚合 Tool/Skill/MCP/Middleware 四种能力，13 个内置插件全部插件化，目录发现 + setup 约定 + plugin.yaml。
- **Skill 系统完成度最高**：SKILL.md + load_token 一次性令牌 + scripts/references 路径隔离 + 超时/截断防护，明显借鉴 Claude Code 且防护最细。
- **上下文压缩模块设计认真**（`middleware/context_compression.py`）：tool_call 配对修复、反抖动、迭代摘要、MD5 缓存、失败冷却。
- **async-first 贯彻彻底**，流式 tool_calls 增量聚合协议（`gateway._generate_stream`）设计合理。
- **记忆衰减公式**（`importance × 0.9^days × log2(2+access_count)`）是有特色的设计。

### 1.2 能力真实性矩阵

| 能力 | 宣称 | 实际 | 证据 |
|---|---|---|---|
| OpenAI 全链路 | ✅ | ✅ 可用 | `adapters/openai.py` |
| Anthropic | ✅ | ⚠️ 非流式可用；流式必炸 | `anthropic.py:78-91` 不设 stream 字段；`:178-179` 把 partial_json 当文本 |
| Zhipu/DashScope/Ollama | ✅ | ❌ 非流式即抛 ValidationError | `parse_response` 构造 `LLMResponse(content=...)` 但必填字段是 `message` |
| 流式输出 | ✅ | ⚠️ 仅 OpenAI 可用；`run(stream=True)` 直接 AttributeError | `agent_runner.py:179` 调用不存在的 `_run_stream` |
| 子代理 SubAgent | ✅ | ⚠️ 非流式可用；流式下失效 | 流式路径不执行 `wrap_model_call`（`agent_runner.py:301-305`），task_* 工具 schema 永远不注入 |
| 后台子代理 | ✅ | ⚠️ 可用但无持久化、无并发上限、与 SubAgent 90% 重复代码 | `async_sub_agent.py:375-466` |
| MultiAgentGraph | ✅ | ⚠️ 文本协议 `[DECISION]` 解析失败即静默退化为全员重复干活 | `multi_agent.py:307-323` |
| 并行工具调用 | ✅ | ❌ 假的 | `parallel_tool_calls` 配置贯穿全链路但执行处只串行 `execute_one` |
| Checkpointer | ✅ | ⚠️ 只能存，无 resume/get_state/replay API | `graph/graph.py:167-171` |
| Channels | ✅（文档列出） | ❌ 死代码 | 除注释外全库无 import |
| Plan-and-Execute | ✅ | ❌ 只有 Planner，无执行器 | `PlanRunResult` 全库除测试外零引用 |
| Parsers | ✅ | ❌ 孤岛模块，Agent 主链路不接受 parser | — |
| HITL | ✅ | ⚠️ 最低配：拒绝=异常=整个 run 终止，模型收不到拒绝反馈 | `hitl.py:57-68` + `agent_runner.py:118` |
| RAG | ✅ | ❌ 中间件必炸 | `rag.py:51-54` 传 `top_k=` 但协议参数是 `limit` → TypeError；`:60` 访问不存在的 `r.score` |
| Memory 中间件 | ✅ | ⚠️ 同样的 `top_k` bug；记忆消息只插不删无限堆积 | `memory.py:87-89` |
| MCP | ✅ | ⚠️ 手写 JSON-RPC 无 initialize 握手；SSE 未实现；未用官方 SDK | `mcp/client.py` |
| 沙箱 | ✅（README 写 Docker/E2B） | ❌ 只有 `create_subprocess_shell`，无隔离 | `sandbox/local.py:8` |
| 可观测性 | ✅（宣称 OTel） | ⚠️ span 无上下文传播、全是平级根 span、异常不 end | `tracing.py:35-80` |
| Examples | ✅ | ❌ 两个示例全部 import 已删除 API，开箱即崩 | `examples/full_flow.py:50-52` |
| 文档 | ✅ | ⚠️ 教已删除的 API（hooks.md、from_builtin）；Plugin/MCP/multi-agent 零文档 | `docs/tools/builtin.md` |
| 版本号 | — | ❌ 三处互相矛盾：README "2.1"、pyproject "3.0.1"、CHANGELOG "1.0.0" | — |

### 1.3 P0 缺陷清单（炸弹级，按危害排序）

1. **`run(stream=True)` → AttributeError**：调用不存在的 `_run_stream`（`agent_runner.py:171-180`）。
2. **流式路径绕过 `wrap_model_call` 与 StateGraph**：子代理在流式下整体失效；用 `type('State', (), {...})()` 伪造 State 传给中间件（`agent_runner.py:290, 301-305, 493`），中间件的修改不回写 session。
3. **非流式 usage 统计错误**：只合并最后一次 LLM 调用的 usage，多步任务 token 消耗被严重低估（`agent_runner.py:667-669`）——成本统计的地基是错的。
4. **三个国产/Ollama 适配器不可用 + 三份复制粘贴代码**：`zhipu/dashscope/ollama` 的 `parse_response` 必抛 ValidationError。
5. **Anthropic 流式从未实现**且工具参数碎片污染文本流。
6. **RAG/Memory 中间件运行时必炸**（`top_k`/`score` 两处 API 不匹配）——因为零测试才存活。
7. **HITL 拒绝即炸穿**：`ToolApprovalRejected` 抛出后整个 run 终止，LLM 永远看不到"被拒"，无法调整策略；`Tool.execution.requires_approval` 声明位与审批流程脱节（`hitl.py:43-70`、`agent_runner.py:118`）。
8. **并行工具执行是死代码**：`session.parallel_tool_calls` 与 `ToolExecutor._execute_batched`（`executor.py:41-98`）无调用方。
9. **`Agent.run()` 不持 session 时每轮新建 session**，多轮对话历史丢失（`base.py:73-89`）。
10. **CJK token 估算 bug**：`cjk_count` 是对"字符总数的十进制字符串"逐字判 CJK，几乎永远得 0，中文场景压缩阈值失真（`context_compression.py:90-107`）。
11. **MCP 安全问题**：stdio 子进程继承全部父环境变量（`client.py:44`），违背 DEVPLAN 自己写的"环境变量过滤"。
12. **`markitdown` 依赖未声明**：file 插件 ImportError 被 `load_all_builtin` 静默吞掉，表现为"file 工具莫名缺失"。
13. **Agent 默认静默注册全部 13 个内置插件**（含写文件/git/跑 Python），调用方不知情（`agent.py:59-60`）。
14. **RagMiddleware/MemoryMiddleware 检索消息只插不删**，长对话 system 消息无限堆积。
15. **StorageMiddleware 同步 IO 阻塞事件循环 + `datetime.now()` 文件名同秒覆盖 + session_id 永远落 "default"**（`storage.py`）。

### 1.4 结构性架构问题（修 bug 之外的根因）

1. **双轨 agent loop**：非流式走 StateGraph、流式手写 while 循环，两套生命周期手动各写一遍，行为持续漂移——这是大部分 P0 bug 的根因。
2. **可变 State 穿节点**：图执行是把一个可变 State 对象顺序穿过节点函数，无 reducer 归并语义；channels 抄了 LangGraph 的 API 面但从未接入执行引擎。要么真接入，要么删掉。
3. **中间件间靠共享可变 `ctx.metadata` 通信**：顺序敏感、互相踩踏（Memory 与 Rag 都 `insert(0, SystemMessage)`）。
4. **单包 + 空 extras**：REDESIGN.md 规划的 `wuwei-ext-*` monorepo 未落地，pyproject 里 graph/middleware/memory/sandbox 等 extras 全是空列表。
5. **弱测试掩盖问题**：129 个测试中大量是"构造对象即通过"的弱断言（如 `test_coverage_gaps.py:252-262` 只断言 issubclass）；无 CI。

---

## 2. 业界格局与痛点洞察（调研结论）

### 2.1 格局已收敛：成功框架的五层模型

2025–2026 年所有跑出来的框架（LangGraph 1.0、Claude Agent SDK、OpenAI Agents SDK、Mastra、MS Agent Framework）都在验证同一个分层：

```
┌─────────────────────────────────────────────────────────┐
│  协议出口层   MCP (工具) · A2A (代理互联) · AG-UI (前端) · OTel (观测)  │
├─────────────────────────────────────────────────────────┤
│  Harness 层   计划工具 · 虚拟文件系统 · 子代理 · Compaction · Skills      │
├─────────────────────────────────────────────────────────┤
│  持久化底座   Checkpoint · interrupt/resume · Durable Execution          │
├─────────────────────────────────────────────────────────┤
│  切面层       Middleware (before_model / wrap_model_call / wrap_tool_call)│
├─────────────────────────────────────────────────────────┤
│  瘦内核       模型驱动的 while 循环 + 状态 + 事件总线（几百行）            │
└─────────────────────────────────────────────────────────┘
```

关键趋势判断：
- **图编排退潮，"模型驱动循环 + 强 Harness"成为主流**（OpenAI Agents SDK / Strands / smolagents / AgentScope 2.0 共同验证）。但图编排的遗产——checkpoint、interrupt、审计——以持久化底座的形式保留下来成为标配。
- **协议层大统一**：MCP（2026-07-28 RC 起进化为分布式生产协议）、A2A v1.0（150+ 组织）、AG-UI 1.0（2026-09-30 冻结）、Agent Skills 开放标准（agentskills.io）。框架差异化从"私有协议"退到"运行时质量"。
- **deepagents 现象**（30k★）："agent = 模型 + 计划工具 + 文件系统 + 子代理 + 压缩"的 harness 组合被产品化复制，是涨星最快的赛道。

### 2.2 最值得借鉴的 10 个设计

| # | 设计 | 来源 | wuwei 对应 |
|---|---|---|---|
| 1 | 内核只做模型驱动 while 循环，复杂编排留给上层 | OpenAI SDK / smolagents | 瘦身 agent_runner |
| 2 | Middleware 四钩子切面，HITL/摘要/护栏全是中间件 | LangChain 1.0 | 已有，需统一到单一循环 |
| 3 | checkpoint + interrupt/resume 作为一等状态语义，存储可插拔 | LangGraph | checkpointer 已有 save，补 resume |
| 4 | Harness 三件套：todo 计划工具 + 虚拟文件系统 + task 子代理 | deepagents | 部分有，需补 FS |
| 5 | 子代理上下文隔离是唯一防污染手段（隔离 > 压缩 > 塞满） | Anthropic | SubAgent 已隔离，需强化 |
| 6 | Compaction 内建 + 工具结果清理 + 结构化笔记 | Claude Agent SDK | 已有压缩，需升级 |
| 7 | 类型安全工具/输出 + 校验失败自动回灌模型重试（ModelRetry） | Pydantic AI | 缺失，易补 |
| 8 | Code-as-action：写一段 Python 沙箱执行替代逐个 tool call | smolagents | 缺失 |
| 9 | Skills 渐进披露开放标准（只注入元数据，按需读正文） | agentskills.io | 已有雏形，对齐标准 |
| 10 | 统一事件总线，一条流同时服务控制台/OTel/前端三端 | OpenAI SDK / AG-UI | 事件类型已有，缺总线 |

### 2.3 用户痛点 Top 15 与 wuwei 的机会点

调研结论（详见附录来源）：长任务错误级联（每步 95% → 10 步 60%）、compaction 丢信息 / context rot、评估覆盖率低（89% 有 tracing 但只有 37% 做线上评估）、prompt injection 无解 + 权限模型缺位、记忆写入策展缺失、成本失控无法按任务归因、MCP 工具描述膨胀（5 个 server 吃掉 40% 上下文）、durable execution 太重（Temporal 运维成本高，轻量方案缺失）、HITL 中断恢复状态一致性、多 agent 上下文共享（Cognition："Don't Build Multi-Agents"）、time-travel 调试无开源实现。

**对 wuwei 最重要的判断**：Top 痛点中 #4（权限）、#6（成本）、#9（轻量 durable）、#13（time-travel 调试）、#5（记忆策展）目前**都没有好的开源标准解**——这正是小框架可以做出差异化、而不是重复造轮子的地方。第 7 章的创新功能全部锚定这些空白。

---

## 3. 定位与总体策略

### 3.1 差异化定位建议

**不做什么**：不拼 integrations 数量（LangChain 数百个 partner package，拼不过）；不做低代码画布（Dify/Coze 的主场）；不做全托管平台。

**做什么**——一句话定位：

> **wuwei：生产就绪的极简 Agent 框架，国内模型生态一等公民。**

三个支点：
1. **国内模型生态一等公民**：GLM / Qwen / DeepSeek / Kimi / 豆包 + OpenAI / Anthropic 全部走同一条高质量代码路径（现在恰恰是反的——国产适配器全是坏的）。中文文档、中文 display_name、国内可用的 embedding/RAG 链路已是现有特色，放大它。
2. **生产化三件套内建**：Durable（轻量 interrupt/resume）、成本治理（预算断路器 + 模型阶梯）、权限沙箱（capability 模型 + 真沙箱）。这三件事所有框架都做得不好，是中小框架的最佳突破口。
3. **开放标准全对齐**：Agent Skills（直接兼容 Claude Code 技能生态，白捡生态）、MCP（官方 SDK）、AG-UI / OTel（协议出口）。

### 3.2 目标架构（v4.0 蓝图）

```
                        wuwei
   ┌──────────────────────────────────────────────┐
   │ 出口: MCP Server 模式 · A2A · AG-UI · OTel    │
   ├──────────────────────────────────────────────┤
   │ Harness: Planner(todo) · VFS · SubAgents ·   │
   │          Compaction · Skills(开放标准)        │
   ├──────────────────────────────────────────────┤
   │ 底座: Checkpoint(SQLite/Postgres/Redis) ·    │
   │       interrupt/resume · Cost Governor ·     │
   │       Context Ledger · Action Log            │
   ├──────────────────────────────────────────────┤
   │ 切面: Middleware × N（全部钩子过单一循环）      │
   ├──────────────────────────────────────────────┤
   │ 内核: 单一 Step 状态机 + 事件总线 + StateGraph │
   │       （StateGraph 变成内核的一种消费方式）     │
   ├──────────────────────────────────────────────┤
   │ 接入: LLMGateway(统一 OpenAI 兼容基类) ·       │
   │       PluginSystem(entry-points) · MCP · 沙箱 │
   └──────────────────────────────────────────────┘
```

---

## 4. Phase 0：修复地基（v3.1，约 2–3 周）

原则：**先让已宣称的能力真实可用**，不改架构。这一阶段结束时，README 上写的每一句话都应该能跑。

### 4.1 LLM Provider 层重构（P0-4/5）

- 新增 `OpenAICompatibleAdapter(base_url, model, extra_headers)` 基类，把「构造请求 / 解析响应 / 解析流式 chunk / tool_calls 增量聚合」一处实现；zhipu、dashscope、ollama、moonshot、deepseek、qwen 全部变成 5–10 行的声明式子类（改 base_url + 模型映射 + 特有 header）。**一处修复全部受益，同时新增 3 家国产模型**。
- 修复 Anthropic 流式：`build_request` 写入 `stream=True`；`content_block_delta` 的 `input_json_delta.partial_json` 走 tool_calls_delta 通道而非文本通道；实现 prompt caching（`cache_control` 打在 system 与最后一条消息上，长对话成本立省 50–90%）。
- 错误分类废弃 `str(exc).lower() 里找 "429"` 的字符串匹配，改为「status_code 优先 + 异常类型映射表」。

### 4.2 Agent Loop 双轨合一（P0-1/2/3，本阶段最重要重构）

把 `agent_runner.py` 收敛为**单一 Step 状态机 + 事件总线**：

```
run(stream=False)  =  async for event in step_machine(): collect; 聚合为 AgentRunResult
run(stream=True)   =  async for event in step_machine(): yield
```

- 非流式与流式共享同一套：中间件生命周期（含 `wrap_model_call`）、usage 累计、State 构建、tool 执行。修一次，两条路径同时修好。
- 删除伪造的 `type('State', ...)()` 与全部死代码（`_execute_tool_calls`、`_execute_tool_calls_with_middleware`、`_emit_event` 空转等，约 200 行）。
- usage 累计改为每步累加（`state.metadata["_usage_total"]`），流式结束回填 `AgentRunResult`。
- 每次 run 不再重建 graph（`_build_graph()` 移出 run 路径），节点函数改为接收显式依赖的类方法。

### 4.3 真并行工具执行（P0-8）

- tool_node / stream_events 统一改调 `ToolExecutor.execute(tool_calls, concurrent=session.parallel_tool_calls)`（`executor.py` 已实现按 `is_concurrency_safe` 分批 gather，接上即可）。
- 并发分支补取消传播：外层 `asyncio.wait` + 超时取消；并发结果按原始顺序回填（已实现）。

### 4.4 HITL 拒绝回喂（P0-7）

- `HitlMiddleware.before_tool` 拒绝时不再抛异常，而是返回一条 ToolMessage：`{"ok": false, "error": {"type": "approval_rejected", "message": "用户拒绝了 <tool>，理由：<...>"}}`，让 LLM 看到拒绝并可调整策略。
- 接通 `Tool.execution.requires_approval`：HitlMiddleware 默认策略 = 审批所有 `requires_approval=True` 的工具（白/黑名单作为 override），消除"声明位与审批流程脱节"。
- 同步修 `ask_user_decision`：从"假工具"升级为走 interrupt 通道（Phase 1 落地 interrupt 后自动获得真实的等待-恢复能力）。

### 4.5 其余 P0 修复清单

| 项 | 修法 | 位置 |
|---|---|---|
| Session 每轮新建 | `run()` 默认复用 `self._default_session`；文档明确"传 session 才多轮"或反向 | `base.py:73-89` |
| RAG/Memory 中间件必炸 | `top_k=`→`limit=`、去掉 `r.score`；注入的检索消息改为"先删上轮注入再插本轮"（给注入消息打 marker） | `rag.py:51-60`、`memory.py:87-89` |
| CJK token 估算 | `cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')`；顺手换 tiktoken 可选依赖 | `context_compression.py:90-107` |
| MCP 环境变量泄漏 | `env = {PATH, HOME, LANG} | config.env` 白名单 | `mcp/client.py:44` |
| markitdown 未声明 | pyproject 声明可选依赖，file 插件 lazy import + 降级为纯文本读取 | `plugin/builtin/file.py:5` |
| Agent 默认全家桶 | `Agent(load_builtins="minimal")`：默认只装 calc/time/json/text；`"all"`/`"none"`/显式列表可选 | `agent.py:59-60` |
| MCPToolAdapter 静默丢字段 | `parameters` 做宽容转换保留扩展键；`side_effect` 映射到 `execution.side_effect` | `mcp/tools.py:33-41` |
| StorageMiddleware | `asyncio.to_thread` 包写盘；文件名加毫秒+uuid；PluginContext 显式传 session_id | `storage.py` |
| TracingMiddleware | 改用 `start_as_current_span` 上下文管理器获得调用树；异常路径 end；延迟初始化 provider 不污染全局 | `tracing.py` |
| LoggingMiddleware | status 读取加 getattr 防御 | `logging.py:64` |
| YAML loader | 支持 tools/middleware 两节；api_key 缺失给可读报错 | `yaml_loader.py:85-115` |

### 4.6 测试与 CI 基建（跟所有修复同步进行）

- **agent_runner.py 补测试**：用 fake adapter（录制的 OpenAI 响应序列）驱动全流程——多步循环、并行工具、流式、中断、HITL 拒绝。这是全库最需要测试的 706 行。
- 每个适配器一条契约测试：录制真实 API 的最小响应（json 文件），断言 parse_response/parse_stream_chunk 输出符合协议——适配器再烂也不至于 ValidationError 级炸弹。
- GitHub Actions：`pytest + ruff + mypy(宽松)` 三件套，Python 3.11/3.12/3.13 矩阵；coverage 门禁只卡 `wuwei/llm`、`wuwei/runtime`、`wuwei/tools` 三个核心包（≥80%）。
- 文档与示例同步：重写两个 examples（能跑通）；删除 hooks.md、修正 from_builtin 相关文档；补 Plugin/MCP/multi-agent 三篇；统一版本号（pyproject 3.0.1 为准，README/CHANGELOG 对齐），CHANGELOG 从现在开始维护。

**Phase 0 验收标准**：examples 一键跑通；5 家 provider 契约测试全绿；`run(stream=True)` 与 `stream_events()` 行为一致；HITL 拒绝后 agent 能继续对话；CI 全绿。

---

## 5. Phase 1：内核强化（v3.2，约 6 周）

### 5.1 Durable Execution：interrupt / resume 落地（对标 LangGraph，但做轻）

这是本框架从"玩具"到"生产"的分水岭，直接命中痛点 #9（轻量 durable 缺失）与 #10（HITL 恢复）：

```python
agent = Agent(..., checkpointer=SQLiteCheckpointer("./state"))
run1 = agent.run("帮我下单 3 件商品", session=s)      # 遇到 requires_approval 工具 → 挂起
run1.status        # "interrupted"
run1.interrupt     # InterruptPayload(tool="place_order", args={...}, checkpoint_id=...)

run2 = agent.resume(run1.interrupt, approved=True)    # 从断点继续，状态由 checkpointer 恢复
# 或者拒绝：agent.resume(run1.interrupt, approved=False, reason="库存不足")
```

实现要点：
- 中间件抛 `InterruptSignal(payload)`（唯一允许炸穿循环的异常），runner 捕获 → checkpointer.save(含 pending_tool_calls) → 返回 interrupted 结果。
- `resume()`：load checkpoint → 重建消息序列 → 把人审结果作为 ToolMessage 注入 → 继续循环。
- HitlMiddleware、ask_user_decision 全部改走 interrupt 通道（Phase 0 的回喂改造是它的前置）。
- 后端：Memory / SQLite（已有，补 resume 语义）/ Postgres（新）；API 与存储解耦。
- 附带收益：审计、回放（7.3 time-travel）、长时后台子代理的持久化全部长在同一个底座上。

### 5.2 状态模型：轻量 reducer，了结 channels 悬案

删除 channels 目录（约 440 行死代码），改为节点级声明式 reducer（保持极简，不引入 Pregel）：

```python
graph.add_node("research", research_fn, writes={"findings": merge_extend})
graph.add_node("draft", draft_fn, writes={"draft": overwrite})
```

节点返回 dict 而非原地改 State → 每步产出可序列化的 delta → 与 checkpointer 天然契合，也为多 agent 共享状态（5.6）打地基。

### 5.3 Plan-and-Execute 补完

把已有的 Planner（只有 plan）接到执行器上：

```python
result = agent.run_plan("调研三个竞品并写对比报告")
# 内部: Planner 产出 TaskList(DAG) → Executor 按拓扑序调度
#   - 无依赖任务并行跑（复用并行工具的 gather 底座）
#   - 每个任务可指定 sub_agent / tool / model
#   - 失败任务重试 n 次后标记 failed 并通知 planner 重规划（local replan）
```

同时给主循环暴露 deepagents 式的 `todo` 计划工具（write_todos / read_todos），让模型自己维护计划——两条路线并存（显式 DAG 给工程化场景，todo 给自主场景）。

### 5.4 结构化输出打通（盘活 parsers 孤岛）

- `Agent.run(..., response_model=MyModel)`：内部 = tool-call 强制模式（OpenAI strict schema / Anthropic tool 强制）→ 校验失败把 pydantic 错误信息回灌模型自动重试（借鉴 Pydantic AI 的 ModelRetry，默认 2 次）。
- JsonOutputParser 的"失败返回原文"静默降级改为可配置（strict 模式抛错）；修嵌套 JSON regex 问题（改用括号计数扫描）。

### 5.5 Session 与存储升级

- Session 持久化统一走 FileStorage（已有 meta.json + jsonl 追加，设计不错），Context 重启恢复；SQLite 后端可选。
- MiddlewareStack 按会话隔离：`ctx.metadata` 拆为 run 级与 session 级两个容器，消除跨 session 串台（MemoryMiddleware 的记忆 namespace 默认跟 session_id 走）。
- 异步化 SQLiteCheckpointer（aiosqlite）。

### 5.6 多 Agent 修订（回应 Cognition 痛点）

- **SubAgent 与 AsyncSubAgent 合并**为一个实现（现在 90% 复制粘贴），SubAgent = 同步等待的后台任务特例。
- 子代理上下文继承的"200 字符硬截断"改为真摘要（复用压缩中间件的 LLM 摘要），或干脆不继承（隔离优先，把必要事实写进 task 描述）。
- **共享动作日志（Action Log）**：一次 run 内所有 agent 往 append-only log 写 `(agent, step, action, result_digest)`；子代理可 `read_log(query)` 取相关片段——用最小成本回应"多 agent 决策碎片化"痛点 #11，也是 7.3 time-travel 的数据底座。
- MultiAgentGraph 的 `[DECISION]` 文本协议改为工具调用协议（leader 拿到 `assign_tasks` 工具，结构化输出，不再 regex 解析）；handoff 从"字符串手术"改为消息级控制流。

**Phase 1 验收标准**：interrupt/resume 跨进程重启可恢复；run_plan 并行分支真实并行；`response_model` 带自动重试；两个 SubAgent 类合并后测试覆盖 ≥80%。

---

## 6. Phase 2：Harness 与生态（v4.0，约 8 周）

### 6.1 Deep Harness：虚拟文件系统（deepagents 缺的第三件套）

- 内置 `VFS`（可落地为本地目录 / 内存 / checkpoint 后端三种 impl），暴露 `read_file / write_file / ls / grep` 工具。
- 用途：中间产物、长文档、搜索结果外置到文件系统，主上下文只留摘要与路径——这是 Anthropic "structured note-taking" 的工程化，直接缓解 context rot（痛点 #2）。
- 与 Skills 打通：skill 的 scripts/references 就挂载在 VFS 上。

### 6.2 Compaction 升级

现有压缩中间件保留，补三件事：
1. **工具结果清理**：超过 N 轮的旧工具输出替换为 `[工具 <name> 的结果已归档，id=<ledger_id>]` 占位（细节进 Context Ledger，7.1）。
2. **摘要消息改用 system/developer 角色或独立通道**，不再以 user 角色注入污染对话语义。
3. **压缩事件可观测**：compaction 发生时发 `context_compaction` 事件（丢了什么、剩多少），调试"压缩导致失忆"有据可查。

### 6.3 Skills 对齐开放标准（agentskills.io）

- FileSystemSkillProvider 换 python-frontmatter 解析（依赖已声明未使用）；字段名对齐开放标准（name/description/license/allowed-tools 兼容映射）。
- **直接兼容 Claude Code 技能目录**：`Agent(skills_dirs=["~/.claude/skills", "./skills"])`——Claude Code 生态的技能白拿，这是零成本借力最大的一步。
- `allowed_tools` 从声明变为强制：技能激活期间 before_tool 按白名单过滤。
- Skill 收编进 Plugin 体系（`PluginKind.SKILL`），消除 skill/plugin 双轨。

### 6.4 MCP 重构

- **弃手写 JSON-RPC，改用官方 `mcp` SDK**（extra 已声明未使用）：自动获得 initialize 握手、capability 协商、streamable HTTP（含 SSE）、OAuth 2.1、progress 通知；stdio stderr 消费防死锁。
- **工具描述膨胀治理**（痛点 #7，Kiro/Anthropic 都在做）：`MCPT discovery` 默认只注入 server 名单与工具计数；新增 `tool_search(query)` 工具按语义检索并按需注册——15 个 server 不再吃掉 40% 窗口。
- 连接治理：connect_all 并发 + 单 server 失败隔离；健康检查与自动重连；工具发现去重（消除现在的双重发现）。
- MCPToolAdapter 的 schema 转换保留扩展键，readOnlyHint → side_effect 映射。

### 6.5 RAG 与记忆重构

- 引入 `VectorStore` Protocol（Qdrant/Chroma/pgvector/内存 四实现，optional deps），替换内存态 KnowledgeStore；chunk 改为按句子/语义边界切分 + 可配置长度。
- 修复并保留记忆衰减模型，补上**自动调度**（每 N 轮触发 cleanup）。
- Embedder 加批量与 LRU 缓存；SimpleEmbedder 换稳定 hash（hashlib.blake2b）消除"两次运行向量不同"。
- （写入策展的完整愿景见 7.5 Memory Curator。）

### 6.6 沙箱与权限

- 沙箱驱动器抽象 `SandboxDriver`：`local`（现状）/ `docker`（默认镜像 + 资源限额 + 无网）/ `e2b`（optional）。`run_python_script`、`run_skill_python_script`、Code Mode（7.6）全部走沙箱。
- 默认拒绝外联：Docker 驱动默认 `--network none`，白名单域名走代理。
- 权限模型的完整愿景见 7.2（capability 模型），本阶段先落「策略中间件」雏形。

### 6.7 可观测性：事件总线 + 三端出口

- 内核加 `EventBus`：所有 AgentEvent 走总线，中间件、OTel exporter、前端订阅者并行消费（解决现在 `_emit_event` 空转、事件被丢弃的问题）。
- OTel 出口按 GenAI 语义约定（gen_ai.* attributes）导出，with span 传播形成调用树；接 Langfuse/Logfire 只需一个 exporter。
- 新增事件类型补全：`tool_error`、`approval_required`、`task_start/end`、`context_compaction`、`budget_warning`——现在 14 种事件类型有 5 种从未发出。
- 控制台美化输出（Rich），对标 Claude Code 的终端体验。

### 6.8 Evals 内建（痛点 #3，89% tracing vs 37% evals 的缺口）

```bash
wuwei eval run ./evals/support_agent/   # pytest 风格
```

- eval case = 数据集（yaml）+ 被测 agent + 断言器：**轨迹断言**（必须/禁止调用的工具序列）、**结果断言**（response_model 校验）、**rubric 评分**（LLM-as-judge，暴露 judge 模型与被测模型解耦选项，文档提示位置偏差）。
- 录制回放：用 checkpointer 的 checkpoint 做回放源，改输入重跑（time-travel 的第一个应用）。
- CI 集成：`--fail-under` 门禁 + 报告输出 junit/json。

**Phase 2 验收标准**：Claude Code 技能目录可直接加载；15 个 MCP server 场景上下文占用 < 10%；evals 跑通一个完整示例；OTel 到 Langfuse 一条链路可见。

---

## 7. Phase 3：创新功能（面向未来，差异化王牌）

> 以下功能锚定第 2.3 节调研出的「无好开源解」痛点，是 wuwei 从"又一个框架"变成"有独特主张的框架"的机会点。按想象力 × 可行性排序。

### 7.1 Context Ledger——上下文保真账本 ⭐ 王牌之一

**痛点**：compaction 是有损的，"压缩丢的信息导致 agent 回退到垃圾代码"（HN 高频抱怨）；context rot 实证存在（Chroma Research：18 个前沿模型全部退化）。

**设计**：把"上下文"从"窗口里的一段历史"升级为"一个可寻址的账本"：

- 所有 tool result / 长文档 / 被压缩掉的内容以内容寻址（hash）方式记入 Ledger，窗口内只留 `<ctx id="a3f2" kind="tool_result" digest="..."/>` 引用。
- 三个召回通道：① 自动——引用临近消息语义相关时由中间件展开；② 半自动——LLM 调 `read_context(id)` 工具按需取回；③ 手动——`ledger.pin(id)` 钉住永不淘汰。
- 淘汰策略可插拔（LRU / 重要性×新鲜度，复用现成的记忆衰减公式）。
- **无损承诺**：压缩只发生在"窗口视图"，账本永远全量——配合 checkpoint，任何时刻都能完整还原当时 LLM 看到了什么。

这是对"压缩=失忆"的正面回答，且与 checkpoint/time-travel 共享同一存储底座，边际成本低。

### 7.2 Capability 权限模型——Agent OS 的安全层 ⭐ 王牌之二

**痛点**：prompt injection 被判定"无完整解"（Simon Willison），业界共识是收缩爆炸半径而非根治；现有框架的权限要么全无（wuwei 现状）要么只有粗粒度审批。

**设计**：借鉴操作系统 capability 模型：

```yaml
# agent.yaml
permissions:
  filesystem:
    read: ["./docs/**"]
    write: ["./out/**"]          # 默认拒绝，白名单放行
  network:
    allow: ["api.github.com"]    # 默认无网
  shell: {enabled: false}
  tools:
    - name: http_get
      budget: {calls_per_run: 10}
```

- 工具执行前过策略引擎（deny-by-default）；越权调用返回结构化错误回喂模型（"你没有 ./etc 的写权限"），模型可改道而不是崩。
- **不可信内容隔离**：工具结果/网页内容默认打 `untrusted` 标记，标记内容进入 prompt 时包上明示边界 + 不允许触发 requires_approval 工具（"Rules of Two" 思路的工程化）。
- 全部决策落审计日志（复用 Action Log）。
- 与 HITL 的关系：HITL 是 capability 的运行时确认（首次使用临时授权，可记忆），permission 是声明式基线。

### 7.3 Time-Travel 调试器——反事实重放 ⭐ 王牌之三

**痛点**：多步 agent 出错后"replay 一次就复现"不成立（非确定性），业界没有开源的反事实调试实现。

**设计**：checkpointer 已经每步存档（Phase 1 修好后免费获得），补上：

```bash
wuwei debug session.jsonl          # TUI：浏览每步的完整上下文/工具调用/成本
```

- **步骤级重放**：从任意步 fork 新 run，可改任意输入——改某个 tool_call 的参数、换模型、改 system prompt，然后只重跑后续步骤（后续 checkpoint 可复用）。
- **轨迹 diff**：两次 run 并排对比（工具序列差异、分歧点定位）。
- 与 evals 闭环：eval 失败 → 一键进入 debug → 修好 → 该案例固化为回归用例。

### 7.4 Cost Governor——预算断路器与模型阶梯

**痛点**：token 成本超预估 10–20 倍、多 agent 15 倍消耗、无法按任务归因。

```python
agent.run(task, budget=Budget(max_cost_usd=0.5, max_tokens=200_000, max_steps=40,
                              on_exhausted="degrade"))   # abort | degrade | ask
```

- runner 每步结算 usage（Phase 0 修好后数据是准的），预算中间件在软阈值发 `budget_warning`、硬阈值执行策略（降级到便宜模型 / 直接中断并输出已完成部分）。
- **模型阶梯路由**：`models=["glm-4-flash", "glm-4-plus", "claude-sonnet"]`——简单步（纯文本响应）用便宜模型，检测到需要工具/长推理时升级；fallback 链（已有）纳入同一机制统一治理。
- 每步事件携带 `cost_so_far`，按 agent/子代理/工具归因到 run 级报表——补齐"成本可归因"这块空白。

### 7.5 Memory Curator——记忆写入策展层

**痛点**：社区共识"主流方案把记忆当检索问题做，真正难的是写入侧策展"；记忆投毒被微软单列为安全威胁。

**设计**：在 MemoryStore 之上加 Curator 中间件：
- **写入闸门**：候选记忆先过分类（偏好/事实/事件/垃圾）+ 去重合并（与现有记忆冲突时不是追加而是合并/更新，保留 revision 历史）。
- **来源信任分级**：用户直接说的话 > agent 推断 > 工具结果；`untrusted` 来源的记忆进隔离区，检索时降权——对记忆投毒的第一道防线。
- **遗忘曲线调度**：现有衰减公式（这是 wuwei 的特色）自动运行 + 情景记忆定期固化成语义记忆（"上周用户说过三次喜欢简洁回复" → "用户偏好简洁回复"）。

### 7.6 Code Mode——代码即动作

- `Agent(code_mode=True)`：给模型一个 `run_code` 工具，解释器内预注入 `tools.*` 命名空间（`tools.file_read(...)`, `tools.http_get(...)`），模型写一段 Python 完成"循环调用 10 个工具 + 中间数据处理"，沙箱一次执行。
- 收益（smolagents 验证）：轮次与 token 显著下降、fan-out 不再受 ~13 个并行 tool call 的准确率上限约束。
- 前置：6.6 沙箱必须先落地；默认关闭，文档标注安全边界。

### 7.7 协议出口：让 wuwei agent 成为网络公民

- **MCP Server 模式**：`wuwei serve --mcp` 把任意 wuwei agent 暴露为 MCP server（agent-as-tool 给别的生态用）。
- **A2A**：暴露 Agent Card + task 生命周期端点（A2A v1.0，Linux 基金会治理，2026 已进生产）。
- **AG-UI**：EventBus → AG-UI 事件映射（1.0 已冻结），配一个 React 参考前端（聊天 + 工具卡片 + 审批按钮 + 中断恢复 UI）——HITL 的体验层补全。
- 这三件事让 wuwei 从"库"变成"可被组合的节点"，是生态冷启动的最短路径。

### 7.8 其它前瞻（列入 backlog，不承诺版本）

- **Steering**：运行中向 agent 注入异步消息改变行为（Claude Agent SDK 已验证需求）——EventBus 天然支持，加 `agent.steer(session, "先跳过测试直接部署")`。
- **Sleep-time compute**：空闲时段由 Letta 式后台任务整理记忆/预计算索引（复用 Memory Curator 与后台子代理）。
- **多模态**：消息体支持 image/audio（OpenAI/Anthropic 协议已有字段，适配器补齐）；computer-use 驱动器（playwright/操作系统）作为沙箱驱动器的一种。
- **自愈循环**：同一工具连续失败 N 次自动触发"反思中间件"（回灌失败历史要求换策略），把长任务半衰期痛点做成可配置模式。
- **`wuwei new` 脚手架**：一条命令生成带 evals/权限/mcp 配置的项目模板。

---

## 8. 工程化与社区（贯穿各阶段）

| 项 | 内容 |
|---|---|
| CI | GitHub Actions：3 个 Python 版本 × (pytest + ruff + mypy 宽松)，核心包 coverage 门禁 |
| 发布 | extras 填实或删除；monorepo 化评估（`wuwei-core` / `wuwei-<provider>` 分包，对齐 REDESIGN 原规划）；语义化版本 + 固定节奏 changelog |
| 插件分发 | 支持 entry-points（`pip install wuwei-plugin-xxx` 即装）；插件市场页（静态索引即可起步） |
| 文档 | mkdocs 重构：补 Plugin/MCP/multi-agent/observability 四篇；每个特性页配可运行片段；中英双语（中文为主，差异化） |
| 测试策略 | 契约测试（适配器）+ fake-LLM 集成测试（runner）+ 少量真实 API 冒烟（nightly）三层 |
| 基准 | 公开一个 transparent 基准：wuwei 在 BFCL 子集 / 自建轨迹集上与 LangGraph、OpenAI SDK 的轮次/token 对比（Code Mode vs tool-call 也自测） |

---

## 9. 路线图总表

| 阶段 | 版本 | 周期 | 核心交付 | 前置 |
|---|---|---|---|---|
| P0 修地基 | v3.1 | 2–3 周 | Provider 统一基类 + 3 家新适配器；单循环合并；真并行；HITL 回喂；CI + 契约测试；examples/docs 同步 | — |
| P1 内核 | v3.2 | 6 周 | interrupt/resume + checkpointer 后端；reducer 状态模型；run_plan；response_model；SubAgent 合并 + Action Log | P0 |
| P2 Harness/生态 | v4.0 | 8 周 | VFS；Compaction 升级；Skills 开放标准兼容；MCP 官方 SDK + tool_search；VectorStore 抽象；沙箱驱动器；EventBus + OTel；evals CLI | P1 |
| P3 创新 | v4.x–v5.0 | 滚动 | Context Ledger；Capability 权限；Time-Travel 调试器；Cost Governor；Memory Curator；Code Mode；A2A/AG-UI 出口 | P2（沙箱/EventBus/Checkpointer） |

优先级原则：**先修说出去就兑现的（P0），再修决定上限的（P1），再修决定生态的（P2），最后做决定身份的（P3）**。P3 中 Context Ledger 与 Cost Governor 依赖最少、痛点最锐，可提前到 P2 并行启动。

### 北极星指标（每个阶段验收用）

1. **真实性**：README 每条宣称都有可运行示例与测试（P0）。
2. **韧性**：kill -9 任意时刻，resume 后任务完成率（P1 起跟踪）。
3. **效率**：同等任务相对裸 OpenAI SDK 的 token 开销倍数 ≤ 1.5×（Code Mode 后目标 ≤ 1.0×）。
4. **安全**：注入测试集（untrusted 内容尝试触发写操作）拦截率 100%。
5. **生态**：Claude Code 技能目录直接可用数；MCP server 开箱兼容率。

---

## 附录：主要调研来源

- 代码审查：wuwei 全量源码（本仓库），关键证据见正文 `文件:行号`。
- 框架格局：LangChain 1.0 与 middleware（langchain.com/blog/agent-middleware）、deepagents（langchain.com/deep-agents）、LangGraph v1 release notes、OpenAI Agents SDK 与 AgentKit（openai.com/index/introducing-agentkit）、Claude Agent SDK（code.claude.com/docs）、Anthropic 工程博客（Effective harnesses for long-running agents / Effective context engineering / Agent Skills）、MS Agent Framework（github.com/microsoft/agent-framework）、Pydantic AI durable execution（pydantic.dev/docs）、smolagents（huggingface.co/docs/smolagents）、Strands 1.0（aws.amazon.com/blogs）、Mastra Durable Agents（mastra.ai/blog）、Dify New Agent（dify.ai/blog）、Eino（cloudwego.io）、AgentScope（agentscope.io）、MCP 2026-07-28 RC（blog.modelcontextprotocol.io）、A2A 一周年（linuxfoundation.org/press）、AG-UI 1.0（copilotkit.ai/blog/ag-ui-1.0）。
- 痛点调研：Chroma Context Rot（trychroma.com/research/context-rot）、METR/Ord 长任务半衰期（arXiv:2505.05115）、LangChain State of Agent Engineering 2025（langchain.com/state-of-agent-engineering）、Cleanlab AI Agents in Production 2025（cleanlab.ai）、Cognition Don't Build Multi-Agents（cognition.com/blog/dont-build-multi-agents）、Anthropic 多 agent 研究系统（anthropic.com/engineering/built-multi-agent-research-system）、Temporal for AI（temporal.io/solutions/ai）、HumanLayer 12-Factor Agents（humanlayer.dev/blog/12-factor-agents）、OctoMind 弃用 LangChain（octomind.dev/blog）、Simon Willison prompt injection 系列（simonwillison.net）、postmark-mcp 事件（snyk.io/blog）、MCP OAuth 困难（matthewswong.com/en/blog/mcp-oauth-authentication-hard）、Kiro Powers 工具膨胀（kiro.dev/blog/introducing-powers）、LLM-as-judge 偏差（ACL 2024: aclanthology.org/2024.acl-long.511.pdf；arXiv:2410.21819）、Bitter Lesson of Tool Calling（arXiv:2608.06370）。

---

## 实施状态（2026-10-02 更新）

### v3.1（Phase 0 修地基）—— ✅ 全部完成
Provider 收敛为 openai/responses/anthropic 三通道；单一执行循环；真并行；HITL 回喂；
session 复用；中文 token 估算；MCP 环境变量白名单；markitdown 懒导入；默认插件最小化；
Storage/Tracing/YAML 修复；CI + 契约测试；examples/文档/版本对齐。

### v3.2（Phase 1/2/3）—— ✅ 全部完成
- **P1 内核**：interrupt/resume 持久化（SQLite 跨实例恢复验证通过）、轻量 reducer
  （channels 已删除）、PlanExecutor + `Agent.run_plan`、`Agent.run_structured`
  （校验失败自动重试）、SubAgent/AsyncSubAgent 合并、共享 Action Log。
- **P2 生态**：wuwei/fs 虚拟文件系统、MCP 官方 SDK（mcp>=2.2）+ tool_search、
  wuwei/vectorstore（InMemory/Chroma/Qdrant/pgvector/Milvus）、沙箱驱动器
  （LocalPython/Docker/E2B）、EventBus + ConsoleReporter、Skills 对齐 agentskills.io
  （allowed-tools 强制执行）、wuwei/evals（CLI + junit）、Compaction 摘要通道修正。
- **P3 王牌**：Context Ledger、Capability 权限模型（wuwei/security）、Cost Governor
  （wuwei/middleware/budget.py）、Time-Travel 调试器（wuwei/debug + `Agent.fork`
  + `wuwei-debug` CLI）、Memory Curator、Code Mode（wuwei/codemode.py）、
  协议出口（wuwei/serve：MCP Server / A2A / AG-UI）。
- 另有（用户追加）：**wuwei.decision** —— Jev（System One 决策模型）客户端。

验证：~320 个离线测试全绿；DeepSeek（deepseek-flash）真实调用验证
非流式 / 工具循环 / 流式 / 结构化输出四条链路。
