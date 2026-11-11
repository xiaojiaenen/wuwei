# Wuwei 3.3.3 复审：低效实现、重复逻辑、过度设计与未解决问题

日期：2026-11-11　基线：`8c23e6c` / PyPI 3.3.3　范围：只读审查 + 离线复现，未改运行时代码、未发版、未调用真实模型。

复现脚本：[`reproduce_efficiency_gaps.py`](reproduce_efficiency_gaps.py)（fake LLM，不联网）
运行：`uv run python docs/audits/2026-11-11/reproduce_efficiency_gaps.py`

---

## 一、总体判断

1. **项目已经不是"功能不够"，而是"同一能力有多套实现 + 内核持久化写得太重"。**
   3.3.x 把正确性契约补得相当扎实（终止语义、unknown 对账、适配器一致性、ActionLog 收敛），
   这些不该再动；剩下的是工程成本问题。
2. **最大的可量化低效在检查点**：40 步的普通工具循环写了 **201 次全量快照 / 13.3 MB**，
   而最终状态只有 130 KB。其中 36% 是同一份会话的重复序列化，25% 是 ActionLog。
   这不是"以后再说"的优化，长任务会直接吃满 SQLite 与 I/O。
3. **有两处"文档承诺 vs 实际行为"的缺口，属缺陷级**：
   Ledger 声称全量保真，但归档的是被 100k 截断后的内容（实测丢 49,973 字符）；
   `ArtifactStore` 在 Windows 直接抛 `PermissionError`，产物/媒体链路不可用。
4. **还有一处同类 400 风险未修完**：`prompted` 结构化输出把 system 追加到消息尾部，
   而"自定义 base_url 端点"在 `auto` 模式下默认就走这条路——正是 3.3.3 修过的那类错误，
   只是换了入口。
5. **测试基线是红的且被当成常态**：本机 31 failed / 5 errors，逐轮以"与基线一致"结案；
   而唯一覆盖 503 行平台网关的 65 项检查，在 CI 里**根本不执行**（不是 pytest 用例）。
6. **不建议**：不要为"更干净"拆出第二套执行引擎，不要动 resume/unknown/claim 语义。
   真正的杠杆是"删重复 + 减写入 + 把红的变绿"，不是再抽象一层。

---

## 二、证据

### A. 低效实现

| # | 问题 | 证据 | 量化 |
|---|---|---|---|
| A1 | 每步全量快照 | `runtime/agent_runner.py:374-402`，`_save_checkpoint` 每次写 `session.to_dict()` + `State(messages=...)` | 5/10/20/40 步 → 26/51/101/201 次保存，280 KB/953 KB/3.5 MB/**13.3 MB** |
| A2 | 每步 5 次保存，其中 `tool_running` 重复两次 | `agent_runner.py:891`（preparing）与 `:920-923`（running）在无中间件路径上连续触发 | kinds: `pre_tool 40 / tool_running 80 / tool_result 40 / step 40 / final 1` |
| A3 | 同一份会话被序列化两次 | `:385` `metadata["session"]=session.to_dict()`（含 context）＋ `:401` `State(messages=...)` | 末次快照 129.8 KB = session 77.1 + State.messages 46.3 + run_state 6.2 |
| A4 | ActionLog 全量进检查点 | `:630`、`:1006` 每次 llm/tool 追加；`AgentSession.to_dict` 原样带入 | 40 步时 action_log **32.6 KB = session 的 42%**，每步都在重写 |
| A5 | `usage_records` / `dispatched_calls` 无界持久化 | `contracts.py:19` `extra="allow"`，`_charge_usage:411` 每次调用 append | 40 步 run_state 6.2 KB；随调用数线性增长 |
| A6 | 单函数承载整个运行循环 | `_execute_events:423-781`：359 行 / 74 个分支（第二名为 39） | 文件 churn 第一：18 次提交 / 3,808 行改动 |
| A7 | token 估算注释与实现不符 | `middleware/context_compression.py:93-113` 注释写"优先使用 tiktoken"，代码里没有 tiktoken 分支 | 只有 `cjk*1.5 + other*0.3` 硬编码 |

### B. 重复逻辑

| # | 重复 | 位置 | 影响 |
|---|---|---|---|
| B1 | 两套文件工具 | `plugin/builtin/file.py`（read_text_file / write_text_file / replace_text_in_file …）与 `fs/tools.py`+`vfs` 插件（vfs_read_file / vfs_write_file …） | `load_builtins="all"` 时同名语义两套、截断阈值两套（20k vs `DEFAULT_READ_LIMIT`） |
| B2 | 路径边界 4 处实现 | `plugin/builtin/file.py:11`、`git.py:22`、`python.py:17`、`fs/local.py:_resolve`、`security/permissions.py` | 修一次要改五处；边界语义容易漂移 |
| B3 | 结构化输出注入两套 | runner `_attach_structured_schema:42`（并入首条 system）/ gateway `build_request:333-339`（**追加到尾部**） | 见 D5：两条路径行为相反 |
| B4 | 子代理构建三份 | `sub_agent.py:237`、`multi_agent.py:51`、`agent.py:342`（planning 内部再建 Agent） | 预算/权限/产物继承规则各不相同（multi_agent 不传 deps） |
| B5 | 上下文压缩两套 + 一套死代码 | `middleware/context_compression.py`（结构摘要/去重/反抖动）、`memory/context_window.py`（仅 `split_turns` 被用）、`memory/context_compressor.py`（**零引用**） | 146 行死代码；两套阈值语义 |
| B6 | 编排面三套 | `graph/`（StateGraph）、`planning/`（DAG+resume）、`multi_agent.py`（leader/workers） | 三套恢复语义与状态模型，认知成本高 |

### C. 过度设计 / 孤儿子系统

| # | 事实 | 证据 |
|---|---|---|
| C1 | `sandbox/` 914 行、7 文件，**没有任何模块调用**（唯一提及是 `codemode.py` 的注释） | `grep -rn sandbox wuwei` 除自身外零命中；工具执行全程 in-process |
| C2 | 平台网关 503 行（飞书/钉钉/Telegram/微信/Webhook）只在脚本测试里出现 | `grep -rl feishu tests` 仅 `test_coverage_gaps.py` |
| C3 | RAG 三层：`memory.KnowledgeStore` → `vectorstore.VectorStore`（5 后端）→ `KnowledgeStoreCompat` → `RagMiddleware` | `vectorstore/adapter.py` 自述"适配成现有协议" |
| C4 | `persistence/` 691 行（SessionStore/JobService/DBOS）在包内无调用，只作为公开 API 存在 | 与 C1/C2 同类：宽度先于深度 |
| C5 | 根目录 146 KB 自述"历史设计记录"（REDESIGN 84 K / OPTIMIZATION_PLAN 46 K / DEVPLAN 16 K）＋ `docs/audits` 1,160 行 | `DEVPLAN.md:1` 明写"不代表当前实现" |

### D. 未解决问题

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| D1 | 测试基线常红 | 本机 `31 failed, 512 passed, 16 skipped, 5 errors in 20.08s` | 回归信号被噪声淹没；每轮以"与基线一致"结案 |
| D2 | 65 项检查在 CI 中不执行 | `tests/test_coverage_gaps.py` 无 `test_` 函数、模块级执行、失败 `sys.exit(1)` | CI 计 0 项；Windows 上则直接破坏收集（必须 `--ignore`） |
| D3 | ArtifactStore 在 Windows 崩 | `artifacts.py:58` `os.open(self.root, os.O_RDONLY)` → `PermissionError [Errno 13]` | 产物/多模态持久化在 Windows 全链路不可用（5+ 用例失败） |
| D4 | Ledger 全量保真失效（>100k） | 执行器 `executor.py:239` 先按 100k 截断，`ledger.py:155` 才归档 | 工具返回 150,000 → 归档 100,027，**静默丢 49,973**，占位文案仍称"如需全文请 read_context" |
| D5 | prompted 模式 system 在尾部 | `gateway.py:333-339`；兼容端点 `default_profile(custom_endpoint=True).native_schema=False` → `auto` 解析为 `prompted` | 发送角色 `['system','user','system']`，严格端点 400（与 3.3.3 同类，换入口复发） |
| D6 | 无真实环境验收 | `IMPLEMENTATION_STATUS.md` 自述未调用真实模型/远端 MCP/Docker/E2B/生产 DBOS | 契约绿 ≠ 互操作绿（可接受，但要写进使用条件） |

---

## 三、值得重新设计的方案（按收益 ÷ 复杂度排序）

### P1 — 先做，收益明确、改动小

| 项 | 问题 | 预期收益 | 改动范围 | 验证方式 |
|---|---|---|---|---|
| **1. 修 ArtifactStore 跨平台** | D3 | 产物/媒体在 Windows 可用；5+ 现有用例转绿 | `artifacts.py` 约 5 行：目录 fsync 仅 POSIX 执行（`if os.name != "nt"` 或 try/except） | `pytest tests/test_model_contracts.py -q`；新增 Windows 上 put/read/delete 冒烟 |
| **2. 统一结构化输出注入** | D5 + B3 | 消除一类 400；injection 单点 | 把 runner 的合并逻辑提为 `llm/output.py:inject_prompted_schema()`，gateway `build_request` 改用它（约 30 行移动，0 新概念） | 断言 `roles == [system, user]` 且 system 唯一；兼容端点 `run_structured` 端到端（fake adapter 抓请求） |
| **3. 让 Ledger 真的全量** | D4 | 兑现"全量保真"，消除静默数据丢失 | `executor.py` 约 10 行：`after_tool` 之前保留原文（metadata 或先归档后截断）；`ledger` 优先于 100k 截断 | 复现脚本第 2 段：归档长度应 = 150,000；补一条 >100k 用例 |
| **4. 把红灯变绿 / 让脚本测试进套件** | D1 + D2 | 本地"一键全绿"，回归才看得见；503 行网关从零测试变有测试 | `pytest.importorskip`（anthropic/mcp/a2a，约 11 处）；`skipif(sys.platform=="win32")` 仅用于 Windows 语义差异（约 6 处，需逐条确认非缺陷）；`test_coverage_gaps.py` 改写为真 pytest 用例或移出 `testpaths` | `pytest -q` → `0 failed, 0 errors`；故意改坏一处观察是否变红 |

> 这四项合计改动量约 150–250 行，不动执行语义，可在一个 patch 内完成并单独发版。

### P2 — 值得重构，收益明确但需回归

| 项 | 问题 | 预期收益 | 改动范围 | 验证方式 |
|---|---|---|---|---|
| **5. 检查点瘦身** | A1–A5 | 保守估算写出量 **-60%～-75%**、保存次数 5→3/步；SQLite 体积同步下降 | `_save_checkpoint`：元数据只留 session 引用/不再重复存 messages（约 30 行）；ActionLog 移出快照（独立表或仅存 digest+引用）；`_charge_usage` 的 `usage_records` 只留聚合；去掉 `:891`/`:923` 的重复 `tool_running`（需保留 preparing→running 的崩溃区分，改为"状态变化才写"） | 复现脚本第 1 段：40 步写出应 ≤ 4 MB、saves ≤ 121；`test_p1_p2_runtime` / `test_action_log_restore` / resume 契约全绿 |
| **6. 拆 `_execute_events`** | A6 | 可读性与可测性；降低最高 churn 文件的改动风险 | 同文件内抽 `_begin_run`（初始化/恢复）、`_call_model`（含流式聚合与结构化分支）、`_finish_error`；不新增类、不改语义（约 -150 行主函数） | 现有 `test_runtime_contracts` / `test_e2e_all` 全绿；逐段 diff 确认只移动不修改 |
| **7. 合并文件工具与路径边界** | B1 + B2 | 工具数量减半；边界单点维护 | `plugin/builtin/file.py` 改为 `fs.LocalFS` 之上的薄壳（保留旧工具名做别名）；`_resolve_workspace_path` 收敛到 `fs` 一个实现 | 两套工具行为对齐测试（读写/越界/符号链接）；`load_builtins="all"` 工具清单快照 |
| **8. 子代理构建收敛** | B4 | 父子预算/权限/产物继承一致 | 抽 `build_child_agent(parent_ctx, *, tools, middleware, deps, artifacts)`，三处调用接入 | `test_delegation_hardening` / `test_multi_agent_integration` 全绿 |

### P3 — 战略收缩（不建议这轮做，但要先定原则）

| 项 | 建议 |
|---|---|
| C1 `sandbox` 914 行无调用 | 二选一：接一条真实路径（`python`/Code Mode 工具走 `Sandbox.run_code`），或 README 标 `experimental/未接入`。**不要**继续扩 E2B/Docker |
| C2 平台网关零测试 | 随 P1-4 一起进套件；四个适配器至少各一条"构造消息→请求体"用例 |
| C3 RAG 三层 | 短期只保留 `KnowledgeStoreCompat` 作为唯一入口，`InMemoryKnowledgeStore` 标注"零依赖默认" |
| B6 三套编排 | 冻结 `planning/` 与 `multi_agent.py` 的新特性，文档明确"默认 runner + graph"；不再新增第三种调度语义 |
| B5 + C5 死代码/历史文档 | 删 `memory/context_compressor.py`、未用的 `ContextWindow` 类；根目录三份历史设计稿移入 `docs/archive/` |

---

## 四、不建议动的地方

- **resume / unknown / claim / head 语义**（`resume:123`、`reconcile:177`、`checkpoint.claim`）：这是 3.3.x 最值钱的部分，检查点瘦身只能改"存什么"，不能改"怎么判"。
- **流式桥接与消费方消失判定**（`runtime/stream_bridge.py`）：修过一次的坑，别再动结构。
- **不引入 Redis/Temporal/第二套 DAG 引擎**：`persistence/` 已经证明"可选后端 + 自建队列"会变成孤儿代码。要持久化就适配一种成熟后端，否则保持本地 SQLite。
- **不要为了减少 LOC 拆 runner**：单一 runner 是正确决策，问题在它太长，不在它唯一。

---

## 五、建议的执行顺序与验收信号

1. **P1-1～P1-4 一起做**（一个 patch）：验收信号是 `pytest -q` 零红 + 复现脚本第 2/3 段输出符合预期。
2. **再做 P2-5 检查点瘦身**（独立 patch，独立发版）：验收信号是脚本第 1 段 `written ≤ 4 MB @40 步`，且 resume/interrupt/unknown 契约测试全绿。
3. **P2-6/7/8 作为"结构性清理"分批**，每批只做一项，靠现有契约测试兜底。
4. **P3 只做决策与文档，不写代码**：先把"哪些是 experimental"写清楚，比继续加能力更值钱。

---

## 六、本次审查的边界

- 只做了静态阅读、git 历史统计与离线复现；**没有**跑真实模型、远端 MCP、Docker/E2B、生产 DBOS。
- 测试基线来自本机 Windows + Python 3.13 + 基础依赖（无 anthropic/mcp），与 CI（Linux + `.[dev,anthropic,mcp]`）不同；D1 的失败分类需要在 CI 环境复核一次。
- 收益数字（-60%～-75%）基于 400 字符工具结果的合成负载；真实负载会话更长，比例会变化，但"两次序列化 + 日志入快照"的结构性冗余不变。
