# 同类框架设计对标与二次复审

日期：2026-11-11
范围：只读复审。除本文件外未修改任何运行时代码。
方法：先取 LangGraph / OpenAI Agents SDK / PydanticAI / smolagents 的公开设计文档作为参照系，再回到 `wuwei` 代码逐条对照；所有数字来自本目录 `reproduce_efficiency_gaps.py` 的实测输出。

## 结论先行

功能面上，Wuwei 已经追平主流框架：六态运行结果、可恢复审批、检查点、结构化输出、子代理、MCP/A2A/AG-UI 都在。**落后的是架构收敛度**，具体表现为三件事：

1. **有机制，没有取舍**。同类框架把「耐久 vs 延迟」「原生 vs 工具 vs 提示」做成显式参数交给调用方；Wuwei 每处只留一个最保守的默认值，代价被隐藏。
2. **有实现，没有单一真相**。同一语义存在多条路径（两份消息、两次元数据、三套委派、两套压缩），每条都正确，合起来是放大。
3. **有能力，没有收敛面**。60+ 公共导出、17 个子系统、33 页 1030 行文档——文档薄是结果，不是原因。

一个数字概括：**40 步普通循环写出 13.3 MB，最终状态只有 130 KB，而每个字节平均被写了 2 次。**

## 一、对标：主流框架靠什么控制复杂度

| 维度 | 主流做法 | 代表 | Wuwei 现状 |
|---|---|---|---|
| 执行内核 | 一个循环 + 可插拔 action 解析 | smolagents `MultiStepAgent`（Code / ToolCalling 共用） | 一个 runner，但委派有三套实现 |
| 状态持久化 | 通道 + 增量/待写 + **显式耐久度** | LangGraph `durability=exit\|async\|sync`、pending writes、DeltaChannel | 每步全量快照 ×5，无档位、无清理 |
| 中断恢复 | 中断项 + 可序列化 state + 恢复值 | OpenAI `RunResult.interruptions` → `RunState.to_json()` → `approve()/reject()` | pending batch 持久化 + claim 对账（**更严**） |
| 结构化输出 | 工具优先 → 原生 → 提示，三级降级 | PydanticAI：`ToolOutput` 为默认，`PromptedOutput` 是最后手段 | 有 `tool` 模式，但 `auto` 永不选中它 |
| 事件流 | 分层：原始增量 / 语义条目 / 运行生命周期 | OpenAI `raw_response_event` / `run_item_stream_event` / `agent_updated_stream_event` | 12 个扁平事件 |
| 测试 | 确定性模型替身是一等公民 | PydanticAI `TestModel`/`FunctionModel`；OpenAI `testing` 模块 | 库内无替身，测试基线 31 failed |

值得抄的不是 API 形状，是**它们如何把代价摆到明面上**：

- LangGraph 明说 `exit` 最快但崩了就丢整轮、`sync` 最稳但每步付一次写延迟，并承认「长会话检查点近二次增长」，给了保留策略和 `DeltaChannel`（只存增量不存全量）。
- PydanticAI 明说 `PromptedOutput` 最不可靠，但它是没有 tool calling 的模型唯一可行方案——所以它是**兜底**而不是默认。
- OpenAI Agents SDK 明说中断要序列化到数据库、可以跨进程恢复；并且把「拒绝理由、粘性批准（always_approve）、嵌套委派里的中断冒泡到最外层」都定义清楚。

## 二、复审发现

### P0-1 检查点：同一份数据写两遍、消息存两份

**证据**

`agent_runner.py:378-402` 组装的 metadata 里放了 `run_state`、`session.to_dict()`、`pending_tool_calls`、`ancestor_state`；随后 `state = State(messages=..., metadata=metadata)`，再 `save(state, metadata=metadata)`。

而 `graph/checkpoint.py:135-160` 的 SQLite 写入是：

```python
snapshot = json.dumps(state.to_dict(), ...)   # 内含 metadata
meta_json = json.dumps(metadata, ...)          # 同一份 metadata 再来一次
INSERT INTO checkpoints (id, state, metadata) ...
```

`State.to_dict()` 返回 `['messages', 'metadata', 'step']`（已实测），所以**每次 save 元数据被序列化两次**。同时 `session.to_dict()` 里已经有 `context.to_dict()` 的完整消息，`State.messages` 是第二份，只在 unknown 对账路径（`agent_runner.py:246`）被读一次。

写入次数也不必要地多：`tool_running` 在 871 写一次，进入批次循环又在 891 写一次，同一批 pending 写了两遍（40 步实测 80 次）。相邻的 `pre_tool`（685）与 `tool_running`（871）之间只有一个函数调用，可以只保留一个崩溃点，但这是可选优化，不是冗余。

实测（400 字符工具结果）：

| steps | saves | 写出总量 | 末次快照 | 其中 |
|---|---|---|---|---|
| 5 | 26 | 280 KB | 18.8 KB | |
| 10 | 51 | 953 KB | 34.7 KB | |
| 20 | 101 | 3.49 MB | 66.4 KB | |
| 40 | 201 | **13.32 MB** | **129.8 KB** | session 77.1 KB（action_log 32.6 KB）+ State.messages 46.3 KB |

**同类怎么做**：LangGraph 每个 super-step 只提交一次全量快照，节点级只写增量（pending writes），并提供 `durability` 三档与保留策略；对可追加通道用 DeltaChannel 避免重复保存全量。

**判断**：这是当前最高收益的改动，且大部分是**删除重复**而非新增机制。按收益排序：

1. 元数据只存一份：`save` 时 `state` 只放 messages/step，`load` 显式合并 metadata 列（`checkpoint.py:173-178` 现在只 `SELECT state`，需要一起改；旧快照保留回退读取）。**约 -50%**
2. 去掉 `State.messages`：消息的唯一真相是 `session.context`，unknown 路径从 metadata 里取。**再 -30%~-35%**
3. 删掉 871/891 的重复 `tool_running`，把 `pre_tool` 与 `tool_running` 合成一次批次写入。**save 次数 201 → ~120**
4. `action_log` 从 session.metadata 里移出，按增量追加（现在它占末次快照 25%）。

预期：40 步 13.3 MB → **2~2.5 MB**，末次快照 130 KB → **35 KB 以内**。范围约 150~220 行，涉及 `agent_runner.py`、`checkpoint.py`、`session.to_dict` 的兼容读取。

**验证**：把 `reproduce_efficiency_gaps.py` 的测量改成断言——40 步写入 ≤ 2.5 MB、每步 save ≤ 2 次、末次快照 ≤ 40 KB；再加一条「旧格式检查点仍可 load/resume」的回归测试。

### P0-2 结构化输出：`auto` 绕过了唯一稳妥的模式

**证据**

`llm/output.py:23-31`：

```python
mode = ("native" if profile.capabilities.native_schema else "prompted") if self.mode == "auto" else self.mode
```

`tool` 模式存在、实现完整，但 `auto` **永远不会**选它。于是任何没有原生 schema 的端点（自定义 base_url、多数兼容网关）直接落到 prompted——三种模式里最不可靠的一种。

而 `gateway.py:333-339` 的 prompted 实现是把 `SystemMessage(...)` **追加到消息尾部**，实测发送角色为 `['system', 'user', 'system']`。这是多家兼容端点直接 400 的形状，也是 3.3.3 修过的那一类问题，只是换了入口。

**同类怎么做**：PydanticAI 把工具模式设为默认（「几乎支持所有模型」），原生模式仅在模型确实支持时启用，提示模式明确标注为最后手段；并且多输出类型时每种注册为独立工具以降低 schema 复杂度。

**判断**：改两处默认值即可，风险低、收益直接。

- `auto` 的解析顺序改为 `native → tool（若 tool_calling）→ prompted`；
- prompted 的 schema 指令放进**第一条 system**（或最后一条 user），不再新增尾部 system。

范围约 30~40 行，含 4 个新测试（provider × mode 矩阵，断言消息首条是 system 且只有一条）。

### P1-1 上下文压缩：算了，但没落到状态上

**证据**

`middleware/context_compression.py:140` 把压缩结果赋给 `ctx.state.messages`；而 `ctx` 里的 state 是 `_copy_messages()` 的副本（`agent_runner.py:306-307`），runner 只在 `agent_runner.py:542` 用它发一次请求，**从不写回 session**。

结果有两层：压缩对**当次 prompt** 有效，对**持久化状态无效**——checkpoint 里的历史仍然全量增长，这与 P0-1 叠加；每步都要重新判定、可能重复摘要。同类实现（Claude 式 compact、OpenAI 的 compaction 能力）是**替换历史**并留下可观测的摘要条目。

同一文件 `:93-97` 的 docstring 写「优先使用 tiktoken」，实现里只有 CJK 字符启发式。这是典型的「用便宜启发式顶替已有机制」——真正需要的精度其实可以免费拿到：**上一次响应的 `usage.prompt_tokens` 就是真值**，只需对新增消息做增量估算并校准。

**判断**：压缩本身该保留，但要做成「一次显式、可观测、写回状态的动作」：

- 触发时用 provider 返回的 usage 校准估算，去掉 char 系数；
- 压缩后写回 `session.context`（并在 metadata 记 `summary_entry` 供审计/召回）；
- 删掉 `memory/context_compressor.py` 与 `memory/context_window.py` 两套未接入实现（零引用）。

范围约 80~120 行。验证：构造 200 轮会话，断言压缩后 `len(session.context.messages)` 显著下降且 checkpoint 体积不再随轮数线性增长。

### P1-2 事件流：缺少「条目」这一层

**证据**：`llm/types.py:60-73` 定义 12 个扁平事件（run/llm/text/reasoning/tool/approval/done/error），`AgentEvent.data` 是裸 dict。UI 侧要自己累积 delta 才能渲染一条消息，也无法把「第 3 步的那次工具调用及其结果」当成一个对象处理。

**同类怎么做**：OpenAI Agents SDK 分三层——原始增量（`raw_response_event`）、语义条目（`run_item_stream_event`，消息/工具调用/交接各是一个 item）、运行级（agent 变更）。上层只需要条目层就能写出完整 UI，底层留给需要逐字渲染的场景。

**判断**：不必复制三层命名，但需要补一个 **item 级事件**：`message_item` / `tool_call_item` / `tool_result_item`，带 `item_id`、`step`、状态。这样 AG-UI/A2A 适配层和日志可以按条目投影，而不必解析 delta。范围约 60~100 行，向后兼容（只增不改）。

### P1-3 测试替身缺失，31 个失败成了基线

**证据**：库内 grep 不到任何 `TestModel`/`FunctionModel`/`FakeLLM`；每轮测试以 `31 failed, 512 passed, 16 skipped, 5 errors` 收尾并被当作「与基线一致」。而 `tests/test_coverage_gaps.py` 是脚本式、CI 里实际执行 0 个用例，却唯一覆盖着 503 行的平台网关。

**同类怎么做**：PydanticAI 提供 `TestModel`（自动生成合法工具调用）与 `FunctionModel`（自定义响应函数），并可用 `ALLOW_MODEL_REQUESTS=False` 全局禁止真实请求；OpenAI SDK 有独立 `testing` 模块。

**判断**：补一个 `wuwei.testing`（`ScriptedLLM` + `assert_tool_calls`），约 80 行。它同时解决两个问题：失败用例可以用替身重写而不是长期挂着；文档里的每段示例都可以被 doctest 执行。这是**让文档不再腐烂**的前提。

### P2-1 三套委派、两套文件工具、914 行零调用的沙箱

`sub_agent.py` / `async_sub_agent.py` / `multi_agent.py` 各自实现了「构造子代理 → 注入工具 schema → 继承治理 → 汇总结果」；`fs/tools.py` 与 `plugin/builtin/file.py` 是两套路径边界实现（另有 2 处路径校验）；`sandbox/` 914 行没有任何调用点。

**判断**：这是「能力铺开快于收敛」的存量，不该再扩。方向是**一个原语 + 两种呈现**：把「子代理即工具」做成唯一实现，同步/后台只是执行方式差异，handoff 是它的特例。沙箱二选一：接一条真实执行路径，或在文档与 `__init__` 标注 experimental 并从 README 能力表移除。不建议为「更干净」重写执行引擎——这部分风险远大于收益。

## 三、Wuwei 比同类强的地方（明确不要动）

1. **unknown 与对账**。OpenAI SDK 的 `error_handlers` 只能给一个受控 fallback，不保证副作用不重复；Wuwei 的 `tool_states` + `_blocked_run` + `reconcile` 是更严格的承诺。这是差异化，不是负担。
2. **恢复不重跑节点**。LangGraph 的 `interrupt()` 靠抛异常实现，恢复时**整个节点从头重跑**，因此要求 `interrupt()` 之前的副作用幂等——这是它的结构性负担。Wuwei 持久化 pending batch 后从批次继续，方向更对。
3. **六态 status**。`completed/interrupted/error/incomplete/cancelled/unknown` 比「成功或抛异常」更接近真实运维；同类多数只有 2~3 态。

结论：**正确性内核已经领先，不要再往里加语义。**改动能省的地方全在 I/O 与默认值。

## 四、取舍表

| 对象 | 动作 | 理由 |
|---|---|---|
| 检查点双写 / 重复消息 | 改造 | 纯冗余，减半写入 |
| 重复 `tool_running` | 删除 | 同一批次写两次 |
| `auto` 输出模式 | 改默认 | 现默认绕过唯一稳妥路径 |
| prompted 注入位置 | 修复 | 尾部 system 触发 400 |
| 压缩写回状态 | 改造 | 现只影响单次 prompt |
| `memory/context_*` 两套 | 删除 | 零引用 |
| 事件 item 层 | 新增 | UI/协议层需要 |
| `wuwei.testing` | 新增 | 让失败可修、文档可执行 |
| 三套委派 / 两套文件工具 | 收敛 | 长期债 |
| `sandbox` | 接入或标注 | 914 行零调用 |
| claim / unknown / 六态 | **保留** | 差异化能力 |

## 五、切分与验收

**PR1（低风险，~150 行）**：输出模式降级链 + prompted 注入位置 + 删两套死代码 + 重复 `tool_running`。验收：provider×mode 矩阵测试全绿；40 步 save 次数 201 → ≤120。

**PR2（~250 行）**：检查点元数据单写 + 去掉 `State.messages` + `action_log` 增量化。验收：40 步写入 ≤2.5 MB、末次快照 ≤40 KB；旧格式检查点可 load/resume。

**PR3（~200 行）**：压缩写回状态 + usage 校准 + `wuwei.testing` + 事件 item 层。验收：200 轮会话 checkpoint 不线性增长；31 个失败用例清零或转 skip 并说明原因。

**决策项（不在 PR 内）**：委派收敛、文件工具合并、沙箱去留。这三项需要先定产品主张——是做「可恢复的执行框架」还是「全家桶 SDK」。当前代码同时押了两边。

## 六、对上一轮的修正

- 上一轮把检查点瘦身归为 P2。**升级为 P0**：因为找到了「state 与 metadata 各写一份」这个纯冗余，收益比估算更大、风险比估算更小。
- 上一轮建议「优化文档」。**修正判断**：1030 行 / 33 页 / 零图表的文档，是 60+ 导出、17 个子系统、6 种状态、4 种输出模式的投影。先收敛公共面（本文 PR1~PR3），再按「一页一个可运行片段」重写；否则只是把复杂度翻译成中文。
- 上一轮的收益百分比基于 400 字符工具结果的合成负载，这一限制仍然成立；但「每字节写两次」与「消息存两份」是结构性的，与负载长度无关。

## 参考

- LangGraph：Durability modes（`exit`/`async`/`sync`）、Checkpointers（pending writes、DeltaChannel、保留策略）、`interrupt()` 与 `Command(resume=)`
- OpenAI Agents SDK：Running agents（`max_turns`、`RunResult`、`error_handlers`）、Human-in-the-loop（`interruptions`、`RunState.to_json/from_json`、`approve/reject`）、Streaming events（三层事件）
- PydanticAI：Output（`ToolOutput`/`NativeOutput`/`PromptedOutput`、重试预算）、Testing（`TestModel`/`FunctionModel`）
- smolagents：ReAct 概念指南（`MultiStepAgent`、typed memory steps、`step_callbacks`）
