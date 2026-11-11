# Wuwei 优化方案 v4

> 版本：2026-11-11 · 适用代码：3.3.3（`pyproject.toml:7`）
> 输入：两轮复审（`REVIEW_JUDGEMENT.md`、`PEER_DESIGN_REVIEW.md`）+ `reproduce_efficiency_gaps.py` 实测
> 关系：**取代根目录 `OPTIMIZATION_PLAN.md`（v1.0 / 2026-10-02）**。旧方案中的 P0 大多已修，剩余项已并入本文。
> 性质：工程改造方案，不含产品路线图。

---

## 0. 一页摘要

### 诊断

Wuwei 的正确性内核（六态状态、unknown 对账、claim 防重领、恢复不重跑节点）**已经领先同类**。问题集中在三处，且都能用「删重复 + 显式取舍」解决，不需要重写执行引擎：

1. **同一份数据写多遍**：每次检查点把 metadata 序列化两次、消息存两份、同一批 pending 写两次。
2. **有实现但没有默认值**：`tool` 输出模式存在却永不启用；`prompted` 有两处注入、行为相反；压缩有实现却不写回状态。
3. **能力面宽于验证面**：三套委派、两套文件工具、914 行零调用沙箱、33 页零图表文档、36 个非通过测试（其中 13 个是环境归因缺失）。

### 目标（可测，不是形容词）

| 指标 | 当前实测 | 目标 | 归属 |
|---|---|---|---|
| 40 步写入总量 | 13.32 MB | ≤ 2.5 MB | P0 + P1 |
| 40 步 save 次数 | 201 | ≤ 120 | P0 + P1 |
| 末次快照 | 129.8 KB | ≤ 40 KB | P0 |
| 每步 `tool_running` 写入 | 2 次 | 1 次 | P0 |
| 有 tool_calling 时的输出模式 | 100% prompted | 0% prompted | P0 |
| Windows 测试 | 36 非通过 | 0 failed（环境类显式 skip） | P0 + P3 |
| 文档 | 33 页 / 1030 行 / 0 图 | 每概念页可运行 + 图 + 生成式参考 | P5 |

### 批次

| 批次 | 主题 | 项数 | 预估改动 | 风险 |
|---|---|---|---|---|
| P0 | 正确性与纯冗余 | 10 | ~330 行 | 低（多为删除与默认值） |
| P1 | 持久化工程化 | 4 | ~220 行 | 中（格式变更，需双向读） |
| P2 | 上下文与输出语义 | 4 | ~200 行 | 中 |
| P3 | 测试与可观测 | 4 | ~280 行 | 低 |
| P4 | 结构收敛 | 5 | ~500 行 | 高（需先定产品主张） |
| P5 | 文档与仓库 | 4 | ~600 行（新增） | 低 |

---

## 1. 基线测量（改造前后都用它验收）

```bash
# 现状（已完成，作为对照）
uv run python docs/audits/2026-11-11/reproduce_efficiency_gaps.py

# 改造后：同一脚本扩展为断言式
uv run python scripts/bench_checkpoints.py --assert
```

当前输出（400 字符工具结果）：

| steps | saves | 写出 | 末次快照 | 说明 |
|---|---|---|---|---|
| 5 | 26 | 280.2 KB | 18.8 KB | |
| 10 | 51 | 953.1 KB | 34.7 KB | |
| 20 | 101 | 3488.1 KB | 66.4 KB | |
| 40 | 201 | 13317.7 KB | 129.8 KB | 构成：session 77.1（action_log 32.6）+ State.messages 46.3 + run_state 6.2 |

40 步的 save 构成：`pre_tool 40 + tool_running 80 + tool_result 40 + step 40 + final 1 = 201`。

**注意**：这些数字依赖 400 字符负载；但「元数据写两次、消息存两份、pending 写两次」是结构性冗余，与负载长度无关。改造后基准脚本必须同时跑 400 字符与 8 KB 两档，防止只优化了小负载。

---

## 2. 设计原则（所有改动必须遵守）

1. **单一真相**。同一份数据只在一个地方落盘、只在一个字段被读取。允许有派生视图，不允许有两个写入口。
2. **代价显式**。耐久 vs 延迟、原生 vs 工具 vs 提示、压缩 vs 保留原文——凡是取舍，做成参数，并给默认值和安全解释。
3. **先删后加**。新增机制前先证明现有路径不能复用；本次 P0/P1 中 60% 的收益来自删除。
4. **不改正确性语义**。`unknown`、`claim`、六态 status、恢复不重跑批次——一行都不碰。
5. **读兼容、写不兼容**。格式变更允许新格式单向写入，但必须能读旧格式；不允许双写两套格式。

### 非目标（明确不做）

- 不引入 LangGraph 式 channel + reducer 全套；现有 `StateGraph` 够用。
- 不重写 `AgentRunner` 为第二套引擎；只做函数级拆分。
- 不为「对齐生态」新增协议（A2A/AG-UI/MCP 保持现状）。
- 不做多语言、多进程调度、分布式队列。

---

## 3. 问题全景

等级：**S0** 行为错误或纯浪费且低风险；**S1** 结构性浪费或语义不一致；**S2** 收敛债；**S3** 仓库/文档卫生。

### A. 持久化与状态

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| A1 | metadata 每次 save 被序列化两次（state 内 + metadata 列） | `agent_runner.py:401-402`；`checkpoint.py:135-160` | S0 |
| A2 | 消息存两份：`State.messages` 与 `session.context` | `agent_runner.py:401`；`session.py:76` | S0 |
| A3 | 同一 pending 连续写两次 `tool_running` | 实测 step 0：`tool_running ['c1']` ×2（891 / 923） | S0 |
| A4 | 无 durability 档位，全部同步写 | `grep durability` 无结果 | S1 |
| A5 | 无检查点保留/清理策略 | `checkpoint.py` 无 prune | S1 |
| A6 | `action_log` 存在 `session.metadata` 内，占末次快照 25% | 实测 32.6 KB / 129.8 KB | S1 |
| A7 | 无「内容未变则跳过」的写入去重 | A3 是它的一个实例 | S1 |

### B. 模型与结构化输出

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| B1 | `auto` 永不选 `tool`，非原生端点直接落 prompted | `output.py:27` | S0 |
| B2 | prompted 注入追加到消息尾部，形成 `system,user,system` | `gateway.py:333-339`；实测复现 | S0 |
| B3 | 3.3.3 只修了 runner 侧注入，gateway 侧未修 | `agent_runner.py:42-67` vs `gateway.py:333` | S0 |
| B4 | Anthropic 原生结构化字段 `output_config` 未验证；相关契约测试全红 | `output.py:88`；`test_v31_core` 3 项 | S1 |
| B5 | 结构化校验失败的反馈以 user 消息追加 | `agent_runner.py:678-680` | S2 |

### C. 上下文与记忆

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| C1 | 压缩只写 `ctx.state.messages` 副本，不回写 session | `context_compression.py:140`；`agent_runner.py:542` | S1 |
| C2 | `_estimate_tokens` docstring 称 tiktoken，实现是字符系数 | `context_compression.py:93-116` | S1 |
| C3 | 两套压缩实现零引用 | `memory/context_compressor.py`、`memory/context_window.py` | S0 |
| C4 | 异步子代理用「最近 6 条 × 200 字」代替摘要，静默丢任务 | `async_sub_agent.py:518-526` | S1 |
| C5 | Memory/RAG 中间件只插不删 | `middleware/memory.py`、`middleware/rag.py` | S2 |

### D. 工具与执行

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| D1 | 产物目录 `os.open(..., O_RDONLY)` + fsync，Windows 直接 `PermissionError` | `artifacts.py:58-63` | S0 |
| D2 | 执行器先截断到 100k，Ledger 才归档，静默丢内容 | 实测 150,000 → 归档 100,027，提示语仍称可 read_context | S0 |
| D3 | 进程组终止语义在 Windows 缺失（4 项契约测试红） | `test_quality_contracts` 4 项 | S1 |
| D4 | 两套文件工具 + 4 处路径边界 | `fs/tools.py` vs `plugin/builtin/file.py` | S2 |
| D5 | 沙箱 914 行零调用；Code Mode 同样未接入 | 已复核 grep 无引用 | S2/S3 |

### E. 委派与编排

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| E1 | 三套委派各自实现 schema 注入与治理继承 | `sub_agent.py`、`async_sub_agent.py`、`multi_agent.py` | S2 |
| E2 | `_execute_events` 359 行 / 74 分支 | `agent_runner.py:423-782` | S2 |

### F. 测试与可观测

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| F1 | 36 个非通过中 13 个是缺可选依赖，没有 marker 归因 | `test_mcp` 5 ERROR、`test_serve` 4、anthropic 4 | S0 |
| F2 | 库内无测试替身，失败长期挂账 | 无 `TestModel`/`FakeLLM` | S1 |
| F3 | 事件 12 类扁平，缺 item 层 | `llm/types.py:60-73` | S2 |
| F4 | `test_coverage_gaps.py` 脚本式，pytest 收集 0 个用例，却唯一覆盖 503 行网关 | `grep -c "def test_"` 65 但不可被 pytest 收集 | S1 |

### G. 文档与仓库

| ID | 问题 | 证据 | 等级 |
|---|---|---|---|
| G1 | 33 页 1030 行，零 mermaid；概念页无参数表、无预期输出 | 全量统计 | S3 |
| G2 | mkdocstrings 已配置但 `reference.md` 只有模块表 | `mkdocs.yml` + `docs/api/reference.md` | S3 |
| G3 | 根目录 4000 行内部文档（REDESIGN 2385 等） | `wc -l *.md` | S3 |
| G4 | 公共导出 60+，无 core/experimental 分层 | `wuwei/__init__.py` | S2 |

---

## 4. 详细设计与改法

### P0-1 检查点：元数据单写（A1）

**现象**：`_save_checkpoint` 把同一个 `metadata` 放进 `State`，又作为参数传一次：

```python
# agent_runner.py:401-402
state = State(messages=self._copy_messages(), metadata=metadata, step=meta["step_index"])
return await self.checkpointer.save(state, metadata=metadata)
```

后端两个都写：

```python
# checkpoint.py:135-160（SQLite）
snapshot = json.dumps(state.to_dict(), ensure_ascii=False)   # 含 metadata
meta_json = json.dumps(metadata, ensure_ascii=False)          # 同一份
```

**根因**：`load` 只 `SELECT state`（`checkpoint.py:173-178`），所以元数据必须藏在 state 里才能被读回；metadata 列是给 `list_checkpoints` 查询用的。两边都不敢删，于是写两遍。

**目标设计**：一份存储，字段明确。

```python
# checkpoint v2
# state 列:    {"schema_version": 2, "step": n, "messages": [...]}
# metadata 列: {"schema_version": 2, "run_state": {...}, "session": {...},
#               "pending_tool_calls": [...], "ancestor_state": {...}}
```

```python
# BaseCheckpointer 统一序列化
@staticmethod
def _serialize_state(state: State) -> dict:
    return {
        "schema_version": 2,
        "step": state.step,
        "messages": [m.model_dump(mode="json") for m in state.messages],
        # 不再写 metadata
    }

async def load(self, checkpoint_id) -> State:
    row = SELECT state, metadata FROM checkpoints WHERE id=?
    state_data = json.loads(row["state"])
    meta = json.loads(row["metadata"] or "{}")
    if state_data.get("schema_version", 1) == 1:
        meta = meta or state_data.get("metadata") or {}   # 旧格式回退
    return State.from_dict({**state_data, "metadata": meta})
```

**改动点**

| 文件 | 改动 | 行数 |
|---|---|---|
| `graph/checkpoint.py` | `_serialize_state` 去 metadata；Memory/SQLite `load` 合并两列；SQLite 加 `SELECT state, metadata` | ~35 |
| `runtime/agent_runner.py` | `state = State(messages=..., step=...)`（不传 metadata） | 1 |
| `graph/checkpoint.py` | `replace_current` 同步改 | ~8 |

**风险**：旧检查点必须能读。回退分支保留，并加一条「v1 快照 load → resume」的测试。

**验收**：40 步写入 ≤ 8 MB（单此项）；`load` 返回的 `state.metadata` 与改造前逐键相等（用现有快照做对照测试）。

---

### P0-2 消息只存一份（A2）

**现象**：`session.to_dict()` 里已有 `context.to_dict()` 的完整消息（`session.py:76`），`State.messages` 是第二份。两套读法各自成立：`resume` 用 `state.metadata["session"]` 重建会话（`agent.py:241-244`），而 restore 与 reconcile 用 `saved.messages`（`agent_runner.py:453`、`:213-218`）。因为都「能用」，所以没人发现重复。

**目标设计**：消息的唯一真相是 `State.messages`（restore/reconcile 已经这么用）；session 快照只存**非消息**字段，恢复时把消息填回去。

```python
# session.py
def to_dict(self, *, include_messages: bool = True) -> dict:
    ...
    context = self.context.to_dict()
    if not include_messages:
        context.pop("messages", None)
    return {..., "context": context}

@classmethod
def from_dict(cls, data: dict, *, messages: list | None = None) -> "AgentSession":
    session = cls(...)
    ctx = data.get("context", {})
    ctx["messages"] = messages if messages is not None else ctx.get("messages", [])
    session.context = Context.from_dict(ctx)
```

```python
# agent.py:241-244（resume）
raw_session = state.metadata["session"]
current = AgentSession.from_dict(raw_session, messages=state.messages)
```

**改动点**：`agent/session.py`（~20 行）、`agent/agent.py:241-244` 与 `:510`（~6 行）、`runtime/agent_runner.py:385` 改 `to_dict(include_messages=False)`、`memory/storage.py` 与 `persistence/sessions.py` 保持默认 `True`（会话存储要完整消息）。

**风险**：中。所有 `AgentSession.from_dict` 调用点都要过一遍（共 7 处：`agent.py:244/510`、`memory/storage.py:73`、`persistence/sessions.py:70/116/123/175`）。旧检查点里 `context.messages` 仍存在，`from_dict` 默认取它，天然兼容。

**验收**：40 步末次快照 ≤ 40 KB；`Agent.load_session` / `resume` / `reconcile` 三条路径的回归测试全绿；对比旧快照恢复后的消息列表完全一致。

---

### P0-3 写入去重（A3）

**现象**：实测同一 step、同一 pending 连续两次 `tool_running`：

```
(0, 'pre_tool',    ['c1'])
(0, 'tool_running', ['c1'])
(0, 'tool_running', ['c1'])   ← 891 与 923 各一次
(0, 'tool_result', [])
```

**根因**：串行路径在「准备执行」时写一次（891），随后中间件未接管、`tc` 被替换或状态推进，又在 923 写一次；两次 payload 相同。批次路径（871）只写一次，是正确的。

**目标设计**：在 `_save_checkpoint` 里做**同 kind + 同指纹**去重，而不是逐处删代码——这样后续新增保存点不会再犯。

```python
def _fingerprint(self, kind, metadata, state) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(kind.encode())
    h.update(str(metadata.get("step")).encode())
    h.update(json.dumps(metadata.get("pending_tool_calls"), sort_keys=True).encode())
    h.update(json.dumps(metadata.get("run_state", {}).get("tool_states"), sort_keys=True).encode())
    h.update(str(len(state.messages)).encode())
    h.update((state.messages[-1].id or state.messages[-1].content[:64]).encode() if state.messages else b"")
    return h.hexdigest()

async def _save_checkpoint(self, *, kind, pending=None, approval_call=None, continuation=None):
    ...
    fp = self._fingerprint(kind, metadata, state)
    if fp == self._last_fingerprint.get(kind):
        return self._last_checkpoint_id          # 内容未变，复用上一次
    self._last_fingerprint[kind] = fp
```

同时把 923 的保存改为**仅在 `remaining` 或 `tool_states` 实际变化时**调用（显式意图，不依赖指纹兜底）。

**改动点**：`agent_runner.py:374-402`（~20 行）、`:923`（~3 行）。

**验收**：40 步 `tool_running` 从 80 → 40；save 总数 201 → ≤ 165（此项）→ ≤ 120（含 P1-1）。

---

### P0-4 结构化输出：让 `auto` 走工具模式（B1）

**现象**：`output.py:27`：

```python
mode = ("native" if profile.capabilities.native_schema else "prompted") ...
```

`tool` 分支写了、`STRUCTURED_TOOL_NAME` 也定义了，但 `auto` 永远到不了。

**目标设计**（对齐 PydanticAI 的降级链：工具优先，原生其次，提示兜底）：

```python
def resolve(self, profile: ModelProfile) -> str:
    if self.mode != "auto":
        mode = self.mode
    elif profile.capabilities.native_schema:
        mode = "native"
    elif profile.capabilities.tool_calling:
        mode = "tool"          # 覆盖绝大多数兼容端点
    else:
        mode = "prompted"      # 没有 tool calling 才允许
    ...
```

**为什么这是收益最高的一行改动**：工具模式的参数由 provider 强约束，且不依赖消息顺序；prompted 是唯一需要在 prompt 里塞 schema 的模式，也是唯一会因为注入位置被 400 的模式。把默认路径从 prompted 移开，等于让 B2 从「主路径故障」降级为「边缘路径隐患」。

**改动点**：`llm/output.py:23-31`（~8 行）、新增 `tests/test_output_modes.py`（4 provider × 4 mode 矩阵，~90 行）。

**验收**：`auto` 在 `tool_calling=True & native_schema=False` 的 profile 上解析为 `tool`；请求里出现 `__wuwei_structured_output` 且**不含** schema system 追加；对不支持 tool 的 profile 仍回落 prompted 并有显式日志。

---

### P0-5 结构化注入只留一处（B2 + B3）

**现象**：3.3.3 已经修好了 runner 侧——`_attach_structured_schema`（`agent_runner.py:42-67`）把约束**并入首条 system**，注释还专门说明了为什么不能 `insert(0, ...)`。但 gateway 侧仍是老写法：

```python
# gateway.py:333-339
if mode == "prompted":
    messages = [*messages, SystemMessage("Return JSON matching this schema: " + json.dumps(...))]
```

实测发送角色：`['system', 'user', 'system']`。**同一个问题修了一半**，因为有两处注入代码。

**目标设计**：gateway 不再自己拼，直接复用同一个函数：

```python
# gateway.build_request
if resolved_mode == "prompted":
    from ..runtime.agent_runner import _attach_structured_schema   # 或上移到 llm/prompting.py
    messages = list(messages)
    _attach_structured_schema(messages, spec.schema_)
```

更干净的落法：把 `_attach_structured_schema` 移到 `llm/prompting.py`，runner 与 gateway 都从那里导入，**物理上只有一份实现**。

**改动点**：新增 `llm/prompting.py`（迁移 26 行）、`agent_runner.py:42-67` 改为导入、`gateway.py:333-339` 改为调用（~6 行）。

**验收**：新增断言「prompted 模式下 system 消息数量 == 1 且位于首位」；对 `['system','user']`、`[]`、`['user']` 三种输入各一条用例；兼容端点（`base_url` + `model_profile` 标记 native_schema=False）回归不再产生尾部 system。

---

### P0-6 产物存储跨平台（D1）

**现象**：`artifacts.py:58-63`：

```python
directory = os.open(self.root, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
```

POSIX 用目录 fd 做 fsync；Windows 上 `os.open` 目录直接 `PermissionError`，产物链路整体不可用。这是 5 个 artifact 契约测试红的直接原因。

**目标设计**：抽出平台守卫，Windows 走「重开文件 + FlushFileBuffers」或直接跳过目录 fsync（Windows 的 `ReplaceFile` 语义已提供原子性）：

```python
def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return          # Windows 无目录 fsync；原子替换由 os.replace 保证
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
```

**改动点**：`artifacts.py`（~12 行）、`plugin/builtin/*` 中同类调用一并收口（grep `os.open(` 后确认无第二处）。

**验收**：`pytest tests/test_model_contracts.py -k artifact` 在 Windows 全绿；Linux 行为不变（目录 fsync 仍执行）。

---

### P0-7 Ledger 归档顺序（D2）

**现象**：执行器 `serialize_tool_output(output, limit=DEFAULT_MAX_OUTPUT_CHARS)` 先截到 100k（`executor.py:234, 324`），Ledger 收到的已经是截断文本。150,000 字符的返回，归档 100,027，**静默丢 49,973**，而占位提示仍写「如需全文请 read_context」。

**目标设计**：把「原始」和「渲染」分开，截断只发生在给模型的视图上。

```python
@dataclass
class RenderedOutput:
    text: str            # 已截断，给模型
    raw: Any             # 完整结果
    truncated: bool
```

- `ToolExecutor.execute_one` 返回 `RenderedOutput`（或让 `ToolOutcome` 携带 `raw`）；
- `LedgerMiddleware.after_tool` 归档 `raw`；
- 若 `raw` 超过阈值（如 1 MB），写入 `ArtifactStore`，账本只存 `ArtifactRef` —— 与产物体系复用，避免把大对象塞进 journal。

**改动点**：`tools/executor.py`（~25 行）、`middleware/ledger.py`（~20 行）、`middleware/stack.py` 传参（~5 行）。

**验收**：新增用例——工具返回 150k / 1.5M 两档，断言「归档长度 == 原始长度」或「归档为 artifact ref 且可读回完整内容」；占位文案与实际能力一致。

---

### P0-8 测试归因与 CI 真实化（F1）

**现象**：本机 36 个非通过里，**13 个不是代码问题**：

| 组 | 数量 | 原因 |
|---|---|---|
| `test_mcp.py` | 5 ERROR | `import mcp` 失败（可选依赖未装） |
| `test_serve.py` | 4 | `fastapi`/`uvicorn` 未装 |
| Anthropic 契约 | 4 | `anthropic` 未装 + 字段待核实（见 B4） |

剩下 23 个才是真问题：artifact 5（D1）、进程组 4（D3）、其余分散在契约断言。

**目标设计**：

1. 引入 `tests/conftest.py` 的统一门禁：`pytest.importorskip("mcp")` 风格改为显式 marker，skip 原因可读。
2. CI 安装全部 extras：`.[dev,anthropic,mcp,serve,dbos,docs]`，让「缺依赖」不再伪装成「失败」。
3. 矩阵加 Windows：`os: [ubuntu-latest, windows-latest]`，允许 Windows 上标记 `xfail` 的项单独列出，不允许静默。
4. 失败清零的定义写进 README：**0 failed；skip 必须有 marker 与原因**。

**改动点**：`tests/conftest.py`（~40 行）、`.github/workflows/ci.yml`（~15 行）、若干测试文件加 marker。

**验收**：本地与 CI 都输出 `0 failed, N skipped`，且 `pytest -q -rs` 的 skip 原因全部是「缺少可选依赖 xxx」。

---

### P0-9 Windows 进程树终止（D3）

**现象**：`sandbox/process.py:11-12` 在 Windows 上返回空 options（不建进程组），超时清理只 `process.kill()` 杀直接子进程。于是 shell 里再起的后代进程成为孤儿，`test_shell_timeout_kills_descendant`、`test_cancel_other_owned_subprocesses[shell/docker]`、`test_cancel_local_process_is_reaped` 四项在 Windows 红。

**目标设计**：

```python
def process_options():
    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

async def kill_tree(process):
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        # 无控制台时 CTRL_BREAK 无效，直接按进程树强制结束
        await asyncio.create_subprocess_exec("taskkill", "/T", "/F", "/PID", str(process.pid))
    with contextlib.suppress(ProcessLookupError):
        process.kill()
```

更严格的做法是 Windows Job Object（`CreateJobObject` + `AssignProcessToJobObject` + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`），但 `taskkill /T` 已能覆盖测试表达的场景，先用它，避免引入 ctypes 依赖。

**改动点**：`sandbox/process.py`（~35 行）。

**验收**：上述 4 项在 Windows 全绿；新增一条「孙进程必须被回收」的用例（`bash -c "sleep 60 & wait"` 类结构）。

---

### P0-10 Anthropic 结构化字段核实（B4）

**现象**：`llm/output.py:85` 对 Anthropic 设置 `request["output_config"] = {"format": {...}}`。这个字段形状需要按官方 API 核实——如果不是真实字段，请求会 400；而这条路径因为缺 `anthropic` 依赖，测试一直红着，等于**从未被真正验证**。

**处理顺序**：

1. 先核实官方结构化输出字段（是否走 `output_format`、beta header 或 tools 的 `input_schema`）；
2. 无法确认前，让 Anthropic 的 `native_schema=False`，由 B1 的降级链自动走 `tool` 模式——**这本身就是一个可发布的正确行为**；
3. 确认后再实现 native，并加一条「用 SDK 的请求模型校验字段合法性」的离线断言（不联网）。

**改动点**：`llm/output.py:80-90`、`llm/profile.py` 的 anthropic profile（~15 行）。

**验收**：Anthropic 路径在 `auto` 下解析为 `tool`（若 native 未确认）；安装 `anthropic` 后相关契约测试从 skip 变为真跑。

---

### P1-1 耐久度档位（A4）

**目标设计**（对齐 LangGraph 的三档，但只做两档起步）：

```python
Agent(..., durability: Literal["sync", "async", "exit"] = "sync")
```

- `sync`：现状，每步边界同步落盘（默认不变，避免语义漂移）。
- `async`：写入 post 到后台任务，在下一次「必须确定」的边界（`interrupt`、`final`、`unknown`、进程退出钩子）前 `await` 收敛。
- `exit`：只在 `final`/`interrupt` 落盘；崩溃丢整轮。

**关键约束**：`interrupt`、`unknown`、`final` **永远同步**，不受档位影响——否则恢复语义被破坏。这一条要写进 docstring 与测试。

**改动点**：`agent/agent.py`（参数透传 ~10 行）、`runtime/agent_runner.py`（`_save_checkpoint` 分派 + `_flush_pending_saves` ~45 行）、`agent/session.py` 无改动。

**验收**：三档各一条集成测试——`async` 下强制在两次保存之间抛异常，断言已落盘的检查点可恢复；`exit` 下断言中间无检查点。

---

### P1-2 保留与清理（A5）

**目标设计**：

```python
await checkpointer.prune(run_id=..., keep_last=20)
await checkpointer.prune(older_than_days=7)      # 全局
```

SQLite 加索引 `(run_id, created_at)`；`Agent(..., checkpoint_retention=20)` 在 `final` 后触发。默认**不自动删**（避免用户数据丢失），只在文档给出推荐配置。

**改动点**：`graph/checkpoint.py`（~50 行，两个后端各一份）、`docs/guides/durable-execution.md`（补一段运维建议）。

**验收**：插入 100 个检查点 → `prune(keep_last=10)` → 剩 10 且 `run_heads` 仍指向最新；`list_checkpoints` 分页不受影响。

---

### P1-3 `action_log` 移出热路径（A6）

**现象**：末次快照 129.8 KB 中 `action_log` 占 32.6 KB（25%），并且**每一次** save 都带着它。

**分两步，先做低风险的**：

1. **中间态不写日志**：`pre_tool` / `tool_running` / `tool_result` 三类检查点移除 `action_log`，只在 `step` / `final` / `interrupt` 保留。动作日志是审计用的，不需要每个瞬时态都快照。约 -60% 的日志写入量，10 行以内。
2. **后续（可选）**：`ActionLog` 自带持久化（append-only 表或 JSONL），checkpoint 只存 `log_id`。这一步涉及新表与迁移，放到 P2 之后。

**验收（第 1 步）**：40 步下 `action_log` 参与写入的次数从 201 → 41；末次快照仍包含完整日志（`final` 那次带全量）。

---

### P1-4 写入合并（A7）

在 P0-3 的指纹去重之上，补两条规则：

- **同 step 内 `pre_tool` 与 `tool_running` 合并**：二者之间只有一次函数调用，崩溃窗口可忽略。合并后每步固定 4 次写（`step` / `tool_running` / `tool_result` / 视情况 `final`）。
- **连续相同 `tool_result` 合并**：批次执行完一次性写，而不是逐条。

**验收**：40 步 save 201 → ≤ 120（与 P0-3 合计）。

---

### P2-1 压缩写回 + 真实 token 计量（C1 + C2）

**目标设计**：

1. **计量**：用上一次响应的 `usage.prompt_tokens` 做锚点，只对新增消息做估算：

```python
def estimate(self, messages) -> int:
    if self._last_prompt_tokens and self._anchor_index <= len(messages):
        return self._last_prompt_tokens + self._estimate_since(self._anchor_index, messages)
    return self._estimate_all(messages)     # 冷启动才全量估算
```

`_estimate_since` 仍可用字符法，但它现在只作用于**增量**，误差不累积。删掉「优先 tiktoken」的错误 docstring。

2. **写回**：压缩后调用 `session.context.replace_messages(compressed)`，并在 `session.metadata["compaction"]` 记录 `{at_step, before_tokens, after_tokens, summary_id}`。

3. **可观测**：压缩发生时发一个事件（复用 P3 的 item 层或现有 `llm_start` 前插入），让调用方能看见「上下文被改了」。

**改动点**：`middleware/context_compression.py`（~70 行）、`agent_runner.py` 提供 `usage` 回填（~10 行）。

**验收**：200 轮会话，压缩后 `len(session.context.messages)` 显著下降且 **checkpoint 体积不再随轮数线性增长**；两次压缩之间不重复摘要（用计数器断言）。

**删除**：`memory/context_compressor.py`、`memory/context_window.py` 整体移除，连带 `memory/__init__.py` 的导出检查。

---

### P2-2 异步子代理的上下文策略（C4）

**现象**：`async_sub_agent.py:518-526` —— 取最近 6 条、每条 200 字，注释明确写「非摘要，避免二次 LLM 调用」。问题不是省钱，而是**可能把任务描述本身截掉**，子代理拿到的是残缺指令。

**目标设计**：显式契约，三种策略由调用方选：

```python
AsyncSubAgent(..., context_policy="brief")   # 默认
# "brief":   {task(全文), constraints, artifact_refs, 最近 3 条原文}
# "summary": 走压缩中间件生成摘要（多一次调用）
# "full":    全量继承（受 token 预算保护）
```

`brief` 的关键是**任务全文必须保留**，只有历史被裁剪；并且给出 `artifact_refs` 让子代理按需取回，而不是丢内容。

**改动点**：`agent/async_sub_agent.py`（~60 行，替换 `_compress_context`）。

**验收**：构造「任务在第 1 条、后续 50 条无关历史」的用例，断言子代理收到的 brief 包含完整任务；`full` 策略在超预算时显式报错而非静默截断。

---

### P2-3 记忆/检索中间件去重（C5）

- 同一 `(source, query_hash)` 的系统消息不重复插入；
- 插入位置统一为「首条 system 之后」，不再 `insert(0, ...)` 抢占 system 位；
- 提供 `max_injected` 上限，超出淘汰最旧。

**改动点**：`middleware/memory.py`、`middleware/rag.py`（各 ~25 行）。

**验收**：10 轮同 query 后，system 消息数量恒定。

---

### P2-4 结构化校验反馈的角色（B5）

`agent_runner.py:678-680` 把「输出 JSON 校验失败，请纠正」作为 **user** 消息追加。这会污染对话历史（用户从未说过这句话），并在下一次 `run` 时把错误提示当作用户输入持久化。

**改法**：改为在本轮消息视图内追加一条 `system` 提示（或 tool 角色失败结果），不写入会话历史。与 B2 的注入收口共用同一个 `llm/prompting.py`。

**验收**：校验失败重试后，会话历史中不存在该提示；重试次数与 `output_max_retries` 一致。

---

### P3-1 测试替身（F2）

新增 `wuwei/testing.py`（或 `wuwei.testing` 子模块，不进入 `__all__` 主命名空间）：

```python
from wuwei.testing import ScriptedLLM, FunctionModel, assert_tool_calls, collect_events

llm = ScriptedLLM([
    tool_call("search", {"q": "x"}),
    text("完成"),
])
agent = Agent(llm=llm, tools=registry, load_builtins=False)
result = await agent.run("go")
assert_tool_calls(llm, [("search", {"q": "x"})])
```

- `ScriptedLLM`：按脚本返回，脚本耗尽即报错（防止隐式循环）；
- `FunctionModel`：`fn(messages, tools) -> LLMResponse`，便于断言 prompt 形状；
- `collect_events`：把事件流收成列表，供 golden 测试。

**为什么优先做**：它是 P3-2（事件 golden）、P5（文档可执行）与 31 个失败用例重写的前置。~100 行，收益覆盖后续所有测试工作。

---

### P3-2 事件 item 层（F3）

在现有 12 类事件之上**追加**（不改已有）：

```python
item_started   {item: {id, type: "message"|"tool_call"|"tool_result", step, name?}}
item_completed {item: {...}, payload: {...}}
```

上层（AG-UI 适配、日志、前端）按 item 投影，不必自己累积 delta；底层 `text_delta` 保留给逐字渲染。

**验收**：用 `collect_events` 对一次「工具调用 + 回答」做 golden 断言：item 序列为 `message → tool_call → tool_result → message`，id 与 session 中的消息 id 一致。

---

### P3-3 事件契约测试与快照（F4 配套）

把所有事件路径（run/stream/resume/interrupt/unknown/error/cancel）各录一份 golden JSON，改执行引擎前先跑一遍。这是 E2 拆分 `_execute_events` 的安全网。

---

### P3-4 脚本测试并入 pytest（F4）

`tests/test_coverage_gaps.py` 有 65 个 `test_` 函数，但是脚本式（`if __name__ == "__main__"` 驱动），pytest 收集 0 个；CI 里它**跑了但什么都没验**，而它唯一覆盖 `wuwei/serve/gateway` 503 行。

**改法**：机械转换——每个 `def test_x(): ...` 加 `assert` 断言现有返回值（当前只 print），补 `pytest.importorskip` 门禁。工作量约 2 小时，覆盖率立刻可见。

---

### P4-1 委派收敛（E1）

**目标**：一个原语，两种呈现。

```
AgentTool(agent, mode="sync"|"background", context_policy, inherit_governance=True)
   ├─ SubAgent        = AgentTool(mode="sync")
   ├─ AsyncSubAgent   = AgentTool(mode="background")
   └─ Handoff         = AgentTool + 路由（MultiAgentGraph 只保留路由与状态归并）
```

三者共同的「构造子 → 注入 schema → 继承治理 → 汇总结果」抽到 `agent/delegation.py`（~200 行），旧的三个类改为薄封装并保留签名（deprecation warning）。

**前置条件**：P3-1（替身）与 P3-3（事件 golden）先落地，否则无法证明「行为等价」。

**验收**：委派相关测试在旧入口与新原语下产生**相同的事件序列**（golden 对比）；代码总行数下降 ≥ 300。

---

### P4-2 拆分 `_execute_events`（E2）

359 行 / 74 分支，是改动风险最集中的地方。按职责拆成 6 个函数，每个 ≤ 80 行：

| 新函数 | 职责 | 来源行 |
|---|---|---|
| `_prepare_step` | 组装 ctx、压缩、结构化注入 | 423-520 |
| `_call_model` | 调用、usage 记账、错误分类 | 520-600 |
| `_handle_output_retry` | 校验失败重试 | 660-684 |
| `_run_tool_phase` | 批次/串行调度（已存在，细化） | 840-1030 |
| `_apply_interrupt` | 中断落盘与 finish | 686-715 |
| `_finalize` | 终态、清理、后台任务 | 783-830 |

**约束**：纯函数级搬移，不改控制流；每搬一个函数跑一次 golden 事件对比。

---

### P4-3 文件工具二合一（D4）

`fs/tools.py` 与 `plugin/builtin/file.py` 是两套实现，路径边界校验散布四处。目标：

```
fs/guard.py     resolve_within(root, path) -> Path   # 唯一实现
fs/tools.py     基于 FS 抽象的工具集（Local/InMem/VFS 共用）
plugin/builtin/file.py  改为薄适配，注册 fs.tools 的实现
```

`resolve_within` 要处理：`..` 逃逸、符号链接逃逸、Windows 大小写与驱动器、UNC 路径。**统一到一处**之后，安全修复只需改一个地方。

**验收**：现有路径边界测试全绿；新增 6 条逃逸用例（`../`、symlink、大小写、UNC、绝对路径、空字节）。

---

### P4-4 沙箱与 Code Mode 决策（D5）

914 行零调用 + 文档宣称 Docker/E2B。二选一，不要继续悬着：

- **方案 A（推荐）**：标注 experimental，从 README 能力表与 `docs/api/reference.md` 移除；`sandbox/` 保留但加 `__all__` 前缀 `_experimental`，不承诺 API 稳定。
- **方案 B**：接一条真实路径——`python` 插件在 `sandbox="docker"` 时走 `DockerDriver`，补至少 2 个集成测试（无 Docker 则 skip）。

**不做**：既宣称又不接。这是文档可信度问题，不是功能问题。

---

### P4-5 公共 API 分层（G4）

60+ 顶层导出，用户分不清「稳定契约」与「顺手导出的内部类」。分三层：

```python
wuwei            # 核心 15 个：Agent, AgentSession, LLMGateway, ToolRegistry, Tool, ...
wuwei.middleware # 中间件全家
wuwei.testing    # 测试替身
wuwei.experimental # sandbox, codemode, 部分 gateway 适配器
```

`__init__.py` 的 `__all__` 拆分 + `docs/api/reference.md` 按层生成。**这是文档能变薄的前提**。

---

### P5-1 文档重构（G1 + G2）

按 Diátaxis 四象限重排，并且**每页一个可运行片段 + 预期输出**：

| 页面 | 现状 | 目标 |
|---|---|---|
| `index.md` | 17 行链接列表 | 价值主张 + 30 秒可跑通 + 预期输出 + 选路表 |
| `quickstart.md` | 代码块无输出 | 安装 → 环境变量 → 最小运行 → **预期输出** → 下一步 |
| 概念页（工具/会话/审批/规划） | 一段压缩说明 | 一句话定义 → 为什么 → mermaid 图 → 可运行片段 → 参数表 → 陷阱 |
| `api/reference.md` | 28 行模块表 | mkdocstrings 生成（已配置未用），按 core/middleware/experimental 分组 |
| 新增 how-to（8 篇） | 无 | 兼容端点接入、结构化输出、中断恢复、并发、预算、压缩、子代理、MCP |
| 新增 `troubleshooting.md` | 无 | 按 `status` 六态与常见 400 归因 |

**验收**：`mkdocs build --strict` 通过；所有代码块由 `pytest --doctest-glob` 或抽取执行（依赖 P3-1 的替身）；每个概念页至少 1 张 mermaid。

### P5-2 根目录文档归档（G3）

`REDESIGN.md`(2385) / `DEVPLAN.md`(493) / `OPTIMIZATION_PLAN.md`(540) 移入 `docs/internal/`（`exclude_docs` 已排除）或删除；根目录只留 `README.md`、`CHANGELOG.md`、`MIGRATION.md`、`USAGE.md`。

### P5-3 变更日志与废弃策略

每个 PR 对应一条 CHANGELOG；破坏性变更在 `MIGRATION.md` 给「旧写法 → 新写法」对照。检查点格式变更必须写清「可读旧格式，写出新格式」。

### P5-4 版本节奏

P0 单独发 3.3.4（纯修复，无 API 变更）；P1+P2 发 3.4.0（新增 `durability`、压缩行为变更）；P3+P4 发 3.5.0（新增 item 事件、`testing`、委派收敛带 deprecation）；P5 随后。

---

## 5. 依赖与顺序

```
P0-8 测试归因 ──┐
P3-1 测试替身 ──┼─→ P3-2 事件 item ─→ P3-3 golden ─→ P4-2 拆分 runner
                │                                   └─→ P4-1 委派收敛
P0-1 ─→ P0-2 ─→ P0-3 ─→ P1-4 写入合并 ─→ P1-1 durability ─→ P1-2 prune
P0-4 ─→ P0-5（注入收口）─→ P2-4（校验反馈）
P0-6 / P0-7 独立
P2-1 压缩 ─→ P2-2 子代理策略
P4-5 API 分层 ─→ P5-1 文档重构
```

硬依赖只有两条：**P3-1 之前不要动 P4-2/P4-1**（没有 golden 就无法证明等价）；**P0-1 之前不要动 P1-1**（在双写基础上加档位只会放大问题）。

---

## 6. 每批完成定义（DoD）

**P0**

- [ ] `bench_checkpoints.py --assert` 通过：40 步写入 ≤ 8 MB、末次快照 ≤ 60 KB、save ≤ 165
- [ ] `tool_running` 每步 1 次
- [ ] `auto` 在 tool_calling profile 上解析为 `tool`；prompted 只有一条 system 且位于首位
- [ ] Windows `artifact` 与 `sandbox cwd` 用例全绿
- [ ] 150k 工具返回的归档长度 == 原始长度（或为可读回的 artifact ref）
- [ ] `pytest -q` 输出 `0 failed`，skip 全部有 marker 与原因

**P1**

- [ ] 40 步写入 ≤ 2.5 MB、save ≤ 120、末次快照 ≤ 40 KB
- [ ] 旧格式（v1）检查点可 load / resume / reconcile
- [ ] `durability` 三档各有集成测试；`interrupt`/`unknown`/`final` 在 `async`/`exit` 下仍同步
- [ ] `prune` 后 `run_heads` 指向最新

**P2**

- [ ] 200 轮会话 checkpoint 体积有界
- [ ] 压缩写回 session 且可观测
- [ ] 子代理 brief 保留完整任务
- [ ] 记忆注入不重复

**P3**

- [ ] `wuwei.testing` 可用，文档示例由它驱动
- [ ] item 事件 golden 覆盖 7 条路径
- [ ] `test_coverage_gaps.py` 被 pytest 收集

**P4**

- [ ] 委派新旧实现事件序列一致，代码净减 ≥ 300 行
- [ ] `_execute_events` 拆分后每个函数 ≤ 80 行、分支 ≤ 15
- [ ] 文件工具只剩一套路径守卫，逃逸用例全绿
- [ ] sandbox/codemode 有明确结论（experimental 或接入）

**P5**

- [ ] `mkdocs build --strict` 通过，0 死链
- [ ] 每个概念页有 mermaid + 可运行片段 + 预期输出
- [ ] 根目录只剩 4 个 md

---

## 7. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 检查点 v2 破坏旧快照恢复 | 中 | 高（丢会话） | 双向读 + 三类路径回归 + 灰度一个版本 |
| 去 `State.messages` 漏改恢复入口 | 中 | 高 | 6 个 `from_dict` 调用点逐一改，加「消息列表逐条相等」断言 |
| `auto→tool` 改变模型行为 | 中 | 中（输出格式变） | 保留 `mode="prompted"` 显式回退；CHANGELOG 说明 |
| 压缩写回导致信息丢失 | 中 | 中 | 摘要保留 artifact 引用，`pin` 可召回 |
| durability=async 下崩溃丢最后一步 | 低 | 中 | 文档明确；interrupt/final 强制 sync |
| 拆分 runner 引入回归 | 高（若不先建 golden） | 高 | 硬依赖 P3-1/P3-3 |
| 委派收敛破坏子代理语义 | 中 | 高 | 新旧入口并存一个版本 + deprecation |

---

## 8. 不做清单（避免"优化"变成"扩张"）

- 不引入 channel/reducer、不做 time travel（当前无需求证据）。
- 不把 `AgentRunner` 改成插件化执行引擎。
- 不为对齐 LangGraph 而要求「节点可重跑」——Wuwei 的 pending batch 恢复更安全。
- 不新增协议适配器；现有 MCP/A2A/AG-UI 先做到有测试。
- 不在 P0/P1 期间顺手改命名或重排目录（会和 diff 冲突）。

---

## 附录 A：证据索引

| 项 | 位置 |
|---|---|
| 双写 | `runtime/agent_runner.py:401-402`、`graph/checkpoint.py:135-160` |
| 消息两份 | `runtime/agent_runner.py:401`、`agent/session.py:76` |
| 重复 tool_running | `runtime/agent_runner.py:891`、`:923`；复现：`docs/audits/2026-11-11/trace_checkpoint_saves.py`（同 step 同 pending ×2） |
| auto 模式 | `llm/output.py:23-31` |
| prompted 注入 | `llm/gateway.py:333-339`；已修版本 `runtime/agent_runner.py:42-67` |
| 压缩不回写 | `middleware/context_compression.py:140`、`runtime/agent_runner.py:542` |
| token 估算 | `middleware/context_compression.py:93-116` |
| 产物 Windows | `artifacts.py:58-63` |
| 截断顺序 | `tools/executor.py:42-57, 234, 324` |
| 子代理截断 | `agent/async_sub_agent.py:518-526` |
| 主循环规模 | `runtime/agent_runner.py:423-782` |
| 沙箱零调用 | `wuwei/sandbox/`（914 行，grep 无 import） |
| 脚本测试 | `tests/test_coverage_gaps.py` |
| 文档统计 | `docs/**/*.md`（33 页 / 1030 行 / 0 mermaid） |

## 附录 B：新增测试清单

1. `test_checkpoint_v2_roundtrip`：写 v2 → 读 → 字段与 v1 等价
2. `test_checkpoint_v1_compat`：旧快照可 load/resume
3. `test_save_dedup`：同 payload 连续保存只落盘一次
4. `test_output_mode_matrix`：4 provider × 4 mode
5. `test_prompted_system_position`：system 恰好一条且在首位
6. `test_artifact_windows_fsync`：Windows 不抛 PermissionError
7. `test_ledger_preserves_raw`：150k / 1.5M 两档
8. `test_durability_async_crash`：后台写未完成即崩溃，恢复到最后肯定点
9. `test_compaction_bounded`：200 轮后体积有界
10. `test_subagent_brief_keeps_task`：任务不被截断
11. `test_event_item_golden`：7 条路径
12. `test_delegation_equivalence`：新旧委派事件序列一致
13. `test_path_guard_escapes`：6 类逃逸
14. `test_scripted_llm_exhausted`：替身脚本耗尽即报错

## 附录 C：验收脚本接口

```bash
# 断言式基准（P0/P1 的 DoD 依赖）
uv run python scripts/bench_checkpoints.py --assert \
  --steps 40 --payload 400 --max-bytes 2.5MB --max-saves 120 --max-last 40KB

# 大负载回归（防止只优化小负载）
uv run python scripts/bench_checkpoints.py --assert --steps 20 --payload 8000

# 事件 golden
uv run pytest tests/test_event_golden.py -q --update-golden   # 仅显式更新
```

---

**一句话总结**：这次不放宽任何语义，只把「写了几遍」变成「写一遍」，把「有个实现」变成「有个默认值」，把「宣传」变成「有测试」。
