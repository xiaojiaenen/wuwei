# Wuwei Agent 框架完整流程

## 一、组件总览

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                               │
│  from_env() / __init__()         创建入口                    │
│  run() / stream_events()         执行入口                    │
│  create_session()                会话工厂                    │
└────────────┬────────────────────────────────────────────────┘
             │ 委托
             ▼
┌─────────────────────────────────────────────────────────────┐
│                      AgentRunner                            │
│  编排 Hook、LLM、Tool 的调用顺序，执行循环直到完成或超限      │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┼──────────┬──────────────┬──────────────┐
    ▼        ▼          ▼              ▼              ▼
┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ LLM  │ │ Hook │ │  Tool    │ │ Context  │ │  Skill   │
│Gateway│ │Mgr   │ │ Executor │ │          │ │  System  │
└──────┘ └──────┘ └──────────┘ └──────────┘ └──────────┘
```

| 模块 | 文件 | 职责 |
|---|---|---|
| `LLMGateway` | `wuwei/llm/gateway.py` | 统一 LLM 入口，内置重试和超时，支持 OpenAI 兼容协议 |
| `AgentSession` | `wuwei/agent/session.py` | 会话容器：配置 + Context + 运行统计 + summary |
| `Context` | `wuwei/memory/context.py` | 内存消息列表，提供 `keep_last_turns()` 裁剪 |
| `AgentRunner` | `wuwei/runtime/agent_runner.py` | 执行循环，编排 Hook/LLM/Tool 调用顺序 |
| `RuntimeHook` | `wuwei/runtime/hooks.py` | 生命周期钩子：`before_llm` / `after_llm` / `after_ai_message` / `before_tool` / `after_tool` |
| `ToolRegistry` | `wuwei/tools/registry.py` | 工具注册表，支持内置/自定义/装饰器三种方式 |
| `ToolExecutor` | `wuwei/tools/executor.py` | 执行工具调用，错误自动包装为 JSON |
| `Storage` | `wuwei/memory/storage.py` | 持久化协议：`save_meta` / `append_message` / `load` / `delete` |
| `FileStorage` | `wuwei/memory/storage.py` | 默认文件实现：meta.json + jsonl，消息增量追加 |
| `ContextCompressor` | `wuwei/memory/context_compressor.py` | 压缩协议 + LLM 实现，把旧对话压缩成结构化摘要 |
| `SkillProvider` | `wuwei/skill/skill.py` | Skill 提供者协议：`list_skills()` / `load_skill_instruction()` |
| `SkillManager` | `wuwei/skill/skill.py` | 聚合多个 SkillProvider，提供索引查询 |
| `ApprovalProvider` | `wuwei/runtime/hitl.py` | HITL 审批协议 + 控制台实现 |
| `Planner` | `wuwei/planning/planner.py` | Plan-and-execute 规划器，把目标分解为任务 DAG |

---

## 二、Hook 链详解

Hook 按注册顺序依次执行。也就是说 `before_*` 和 `after_*` 都是 `Hook1 → Hook2 → ...`，不会倒序执行。

```
before_llm:       Hook1 → Hook2 → Hook3 → Hook4 → Hook5
after_llm:        Hook1 → Hook2 → Hook3 → Hook4 → Hook5
after_ai_message: Hook1 → Hook2 → Hook3 → Hook4 → Hook5
before_tool:      Hook1 → Hook2 → Hook3 → Hook4 → Hook5
after_tool:       Hook1 → Hook2 → Hook3 → Hook4 → Hook5
```

### 2.1 各 Hook 职责

| Hook | 文件 | before_llm | after_llm | after_ai_message | before_tool | after_tool |
|---|---|---|---|---|---|---|
| **SkillHook** | `wuwei/runtime/skill_hook.py` | 注入 skill 使用指引到 system prompt | - | - | - | - |
| **ContextCompressionHook** | `wuwei/runtime/context_hook.py` | 超阈值：压缩旧轮次到 summary，裁剪内存 context | - | - | - | - |
| **StorageHook** | `wuwei/runtime/storage_hook.py` | step=0：存 meta + 追加 user msg | 非流式追加 AI 回复 | 流式追加完整 AI 回复 | - | 追加 tool msg |
| **HitlHook** | `wuwei/runtime/hitl_hook.py` | - | - | - | 检查审批列表，拦截需确认的工具 | - |
| **ConsoleHook** | `wuwei/runtime/console_hook.py` | 打印 [llm:call] | 打印耗时/token | - | 打印工具名 | 打印工具结果 |

### 2.2 推荐注册顺序

```python
hooks = [
    SkillHook(),                 # 1. 最先注入 skill 指引
    ContextCompressionHook(...), # 2. 压缩/裁剪旧上下文
    StorageHook(storage),        # 3. 持久化消息
    HitlHook(...),               # 4. 审批拦截
    ConsoleHook(),               # 5. 调试日志
]
```

---

## 三、执行循环

### 3.1 每轮步骤

```
┌─────────────────────────────────────────────────────────┐
│ 1. copy context                                          │
│    messages = [m.model_copy(deep=True) for m in context] │
│                                                          │
│ 2. before_llm hooks                                      │
│    messages, tools = await hooks.before_llm(             │
│        session, messages, tools, step, task              │
│    )                                                     │
│    ├─ SkillHook: 注入指引到第一条 system msg              │
│    ├─ ContextCompressionHook: 超过 compress_after_turns  │
│    │   则压缩旧 turn 到 session.summary，裁剪 context     │
│    ├─ StorageHook(step==0): save_meta + append_user_msg │
│    ├─ HitlHook: 透传（before_llm 无逻辑）                │
│    └─ ConsoleHook: 打印 [llm:call] step=...              │
│                                                          │
│ 3. LLM 调用                                              │
│    非流式：response = await llm.generate(...)             │
│    流式：async for chunk in await llm.generate(...):      │
│          累积 content / reasoning / tool_calls            │
│                                                          │
│ 4. 写入 assistant message                                │
│    非流式：                                                │
│      await hooks.after_llm(session, response, step, task) │
│      context.add_ai_message(response.message...)          │
│    流式：                                                  │
│      ai_msg = context.add_ai_message(拼好的完整消息)       │
│      await hooks.after_ai_message(session, ai_msg, ...)   │
│                                                          │
│ 5. 如果 finish_reason == "tool_calls" / 有 tool_calls:    │
│    │                                                     │
│    ├─ 逐个 tool_call:                                    │
│    │   ├─ before_tool hooks                              │
│    │   │   └─ HitlHook: 检查 require_approval_tools      │
│    │   │       需要审批 → ConsoleApprovalProvider 询问    │
│    │   │       用户拒绝 → ToolApprovalRejected 异常       │
│    │   │       用户同意 → 继续                            │
│    │   │                                                │
│    │   ├─ ToolExecutor.execute_one(tool_call)            │
│    │   │   成功 → {"ok": true, "result": ...}            │
│    │   │   失败 → {"ok": false, "error": ...}            │
│    │   │                                                │
│    │   └─ after_tool hooks                               │
│    │       └─ StorageHook: append tool msg 到 jsonl      │
│    │                                                    │
│    └─ context.add_tool_message(...) 全部追加完            │
│       step++ 继续循环                                    │
│                                                          │
│ 6. finish_reason == "stop" → 返回 AgentRunResult / done  │
│    finish_reason == "length" → 同上                      │
│    达到 max_steps → 追加截断消息，返回                    │
└─────────────────────────────────────────────────────────┘
```

> 小重点：流式路径没有完整 `LLMResponse`，所以不能依赖 `after_llm` 持久化 assistant。框架在流式结束、拼出完整 assistant message 后触发 `after_ai_message`。

### 3.2 三种执行路径

| 路径 | 方法 | 返回 |
|---|---|---|
| 非流式 | `agent.run(user_input, stream=False)` | `AgentRunResult`（content, usage, latency_ms, llm_calls） |
| 流式 | `agent.run(user_input, stream=True)` | `AsyncIterator[LLMResponseChunk]` |
| 事件流 | `agent.stream_events(user_input)` | `AsyncIterator[AgentEvent]`（text_delta/tool_start/tool_end/done/error） |

---

## 四、上下文治理

### 4.1 压缩流程

```
对话轮次超过 compress_after_turns（默认 16）
  │
  ▼
ContextCompressionHook._compress_old_turns()
  │
  ├─ 取 turns[上次压缩位置 : 当前压缩位置] 的旧消息
  ├─ LLMContextCompressor.compress(old_messages)
  │    → 生成结构化摘要：目标、事实、约束、进度、风险
  ├─ 写入 session.summary
  ├─ session.context.keep_last_turns(keep_recent_turns)
  │    → 内存保留最近 N 轮，旧消息从内存移除
  └─ 重置 _context_compressed_until_turn = 0
```

### 4.2 滑动窗口

```
SimpleContextWindow.build_messages(session, messages)
  │
  ├─ split_turns(messages) → system_msgs + turns
  ├─ 取最后 max_recent_turns（默认 10）轮
  ├─ 若有 session.summary，插入摘要 system msg
  ├─ 截断过长 tool 输出（max_tool_chars，默认 8000）
  └─ 返回 [system, summary_msg, ...recent_messages]
```

### 4.3 关键设计

- **不会丢数据**：裁剪的是 Context 的内存副本，发给 LLM 的是另一份 copy。完整历史通过 StorageHook 逐条追加到存储
- **不会拆坏结构**：按 turn 裁剪，确保 `assistant(tool_calls) → tool` 始终成对保留
- **摘要作为桥梁**：旧对话压缩成摘要，插入 system prompt，LLM 知其全貌但不占窗口

---

## 五、持久化

### 5.1 Storage 协议

```python
class Storage(Protocol):
    async def save_meta(self, session: AgentSession) -> None:
        """保存会话元数据（不含消息）。"""
    async def append_message(self, session_id: str, message: Message) -> None:
        """追加一条消息。"""
    async def load(self, session_id: str) -> AgentSession | None:
        """加载完整会话（元数据 + 全部消息）。"""
    async def delete(self, session_id: str) -> None:
        """删除会话。"""
```

### 5.2 FileStorage 文件格式

**`{session_id}.meta.json`** — 会话元数据：
```json
{
  "session_id": "...",
  "system_prompt": "...",
  "max_steps": 10,
  "summary": "旧对话压缩摘要...",
  "metadata": {"app_user_id": "u1"},
  "last_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
  "last_latency_ms": 1200,
  "last_llm_calls": 3
}
```

**`{session_id}.jsonl`** — 消息逐行追加：
```jsonl
{"role":"user","content":"帮我查时间"}
{"role":"assistant","content":"好的，我来查","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_now","arguments":{}}}]}
{"role":"tool","content":"{\"ok\": true, \"datetime\": \"2026-04-27 10:30:00\"}","tool_call_id":"c1"}
{"role":"assistant","content":"现在是 2026 年 4 月 27 日 10:30"}
```

### 5.3 StorageHook 写入时机

```
before_llm (step==0):
  save_meta(session)         ← 存会话元数据
  append_message(user_msg)   ← 存用户消息

after_llm:
  append_message(ai_msg)     ← 非流式路径：存 AI 回复
  if session.summary:
    save_meta(session)       ← summary 更新时同步元数据

after_ai_message:
  append_message(ai_msg)     ← 流式路径：存拼接后的完整 AI 回复
  if session.summary:
    save_meta(session)       ← summary 更新时同步元数据

after_tool:
  append_message(tool_msg)   ← 存工具结果
```

### 5.4 恢复流程

```python
# 启动时恢复
storage = FileStorage(".sessions")
session = await storage.load("demo-session")

if session:
    # 继续已有会话，所有历史消息已恢复到 session.context
    await agent.run("继续聊", session=session)
else:
    # 新建会话
    session = agent.create_session(session_id="demo-session")
    await agent.run("你好", session=session)
```

### 5.5 自定义存储

实现 `Storage` 协议的 4 个方法即可接入任意后端：

```python
class MySQLStorage:
    async def save_meta(self, session): ...
    async def append_message(self, session_id, message): ...
    async def load(self, session_id): ...
    async def delete(self, session_id): ...
```

---

## 六、Skill 体系

### 6.1 组件关系

```
FileSystemSkillProvider          SkillManager
  扫描 skills/ 目录              聚合所有 Provider
  解析 SKILL.md                 提供索引查询
        │                            │
        └──────────┬─────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   SkillHook    skill 工具    控制台/HITL
   注入 system   list_skills   可像普通工具
   prompt 指引  load_skill     一样审批/记录
               run_skill_
               python_script
```

### 6.2 SKILL.md 格式

```markdown
---
name: code-review
description: 代码审查技能，检查代码中的常见问题并给出改进建议
---

## 代码审查

对用户提供的代码进行审查，检查以下方面：
- 命名规范
- 错误处理
- 潜在的性能问题
- 安全漏洞

按照"问题 → 严重程度 → 建议"的格式输出审查结果。
```

### 6.3 工作流程

```
1. FileSystemSkillProvider 扫描 SKILL.md
     ↓
2. SkillManager 建立索引
     ↓
3. SkillHook.before_llm: 注入指引到 system prompt
   "Skill 是可选的专门能力，不是默认步骤..."
     ↓
4. LLM 根据指引判断任务是否匹配 skill
     ↓
5. 匹配 → 调用 list_skills() 查看可用 skill 列表
     ↓
6. 选中 → 调用 load_skill("code-review")
     返回完整 instruction + load_token
     ↓
7. LLM 读取 instruction，按指引执行审查
     ↓
8. 如需脚本 → 调用 run_skill_python_script(name, path, token)
     校验 load_token，subprocess 执行
```

### 6.4 Skill 工具（3 个）

| 工具 | 描述 |
|---|---|
| `list_skills` | 列出所有可用 skill 的名称、描述、是否有 Python 脚本 |
| `load_skill` | 加载指定 skill 的完整正文，返回 instruction + load_token |
| `run_skill_python_script` | 执行 skill 下 scripts/ 中的 Python 脚本，需 load_token |

### 6.5 安全机制

- 脚本路径限制：必须在 skill 目录下的 `scripts/` 子目录内
- load_token 校验：必须先 `load_skill` 成功才能执行脚本
- 超时控制：默认 10 秒
- 输出截断：默认 4000 字符

---

## 七、HITL 人类审批

### 7.1 组件

```
HitlHook                          ApprovalProvider (Protocol)
  注册到 Hook 链                    实现审批交互
  before_tool 拦截                └─ ConsoleApprovalProvider
        │                             终端交互 [y/N]
        ▼
  ApprovalPolicy
    require_approval_tools: {"save_note"}  需要审批的工具集合
    auto_approve_tools: set()              自动放行的工具集合
```

### 7.2 审批流程

```
before_tool(tool_call)
  │
  ├─ tool_call.function.name 在 auto_approve_tools → 放行
  ├─ tool_call.function.name 在 require_approval_tools → 审批
  │   ├─ ApprovalProvider.request_approval(tool_call)
  │   │   → 控制台显示工具名、参数，等待用户输入
  │   ├─ 用户输入 y/yes → approval_decision(approved=True)
  │   │   → 放行，工具正常执行
  │   └─ 用户输入 n/no → approval_decision(approved=False)
  │       → 抛出 ToolApprovalRejected，返回错误 JSON
  └─ 不在任何集合 → 默认放行
```

### 7.3 自定义审批

```python
class WebApprovalProvider:
    async def request_approval(self, tool_call) -> ApprovalDecision:
        # 通过 WebSocket/HTTP 发送审批请求
        # 等待远程用户点击按钮
        ...
```

---

## 八、Plan-and-Execute

### 8.1 PlanAgent

```
PlanAgent.run(goal)
  │
  ├─ 规划阶段
  │   Planner.plan_task(goal)
  │     → LLM 分解目标为 Task DAG
  │     → 返回 TaskList
  │
  └─ 执行阶段
      PlannerExecutorRunner.execute_events(task_list)
        │
        ├─ 计算依赖关系（task.next → dependencies）
        ├─ 循环：
        │   ├─ 找出所有依赖已完成的就绪 task
        │   └─ 为每个 task 创建子 AgentSession + AgentRunner
        │       ├─ prompt = 根目标 + task 描述 + 上游结果
        │       └─ 完整 hook 链生效（压缩/持久化/审批/...）
        │
        └─ 汇总 → PlanRunResult
```

### 8.2 Task DAG

```
Task(id=1, next=[2, 3])     Task 1 完成后 Task 2、3 并行
Task(id=2, next=[4])
Task(id=3, next=[4])        Task 2、3 都完成后 Task 4 开始
Task(id=4, next=[])
```

---

## 九、完整示例

运行入口：`examples/full_flow.py`

```
完整示例文件结构：
examples/
├── full_flow.py           完整流程示例
├── while_true_agent.py    HITL 审批示例
├── .env                   LLM 配置（需自行创建）
└── skills/
    └── code-review/
        ├── SKILL.md       skill 定义
        └── scripts/
            └── count_lines.py   skill 脚本
```

### 9.1 示例组件清单

```
Agent
├── LLMGateway.from_env()           LLM 网关
├── ToolRegistry
│   ├── get_now                     内置 time 工具
│   ├── add                         自定义工具
│   ├── save_note                   自定义工具（需审批）
│   ├── list_skills                 内置 skill 工具
│   ├── load_skill                  内置 skill 工具
│   └── run_skill_python_script     内置 skill 工具
├── SkillManager
│   └── FileSystemSkillProvider     skills/ 目录
└── Hook 链
    ├── SkillHook
    ├── ContextCompressionHook      LLMContextCompressor
    ├── StorageHook                 FileStorage
    ├── HitlHook                    ConsoleApprovalProvider
    └── ConsoleHook
```

### 9.2 使用方式

```bash
# 1. 创建 .env 文件
echo 'OPENAI_API_KEY=sk-xxx' > examples/.env
echo 'OPENAI_BASE_URL=https://api.openai.com/v1' >> examples/.env
echo 'OPENAI_MODEL=gpt-4o' >> examples/.env

# 2. 运行
python examples/full_flow.py
```

### 9.3 交互建议

可以用下面几类输入观察完整链路：

```text
现在几点？                         # 触发内置 time 工具
帮我把 3 和 5 相加                  # 触发自定义 add 工具
请把“今天完成了框架流程文档”记下来   # 触发 save_note + HITL 审批
用 code-review skill 看看 examples/full_flow.py 的问题
```

运行后可以观察：

- `.wuwei_demo_sessions/demo-session.meta.json`：会话元数据、summary、统计信息。
- `.wuwei_demo_sessions/demo-session.jsonl`：user / assistant / tool 消息逐行追加。
- `agent_notes.txt`：`save_note` 审批通过后的笔记输出。

这些都是示例运行产物，已在 `.gitignore` 中排除。

---

## 十、数据流全景

```
用户输入 "帮我审查这段代码，然后记下来"
  │
  ▼
Agent.stream_events(user_input, session)
  │
  ▼
AgentRunner.stream_events()
  │
  ├─ context.add_user_message("帮我审查...")
  │
  ├─ [step 0]
  │   ├─ copy context → [system, user]
  │   ├─ before_llm:
  │   │   ├─ SkillHook: 注入 skill 指引到 system msg
  │   │   ├─ ContextCompressionHook: turn=1，未超不压缩
  │   │   ├─ StorageHook(step=0): save_meta + append user msg
  │   │   └─ ConsoleHook: [llm:call] step=0
  │   │
  │   ├─ LLM.generate → "我需要审查代码，先看看有什么 skill"
  │   │   返回 tool_calls: [list_skills()]
  │   │
  │   ├─ after_ai_message: StorageHook append AI msg
  │   ├─ context.add_ai_message(tool_calls=[list_skills])
  │   ├─ before_tool: HitlHook → list_skills 不在审批列表，放行
  │   ├─ execute_one: → [{name:"code-review", description:"..."}]
  │   ├─ StorageHook: append tool msg
  │   └─ context.add_tool_message(...)
  │       step++ 继续
  │
  ├─ [step 1]
  │   ├─ copy context → [system, user, assistant+tool_calls, tool]
  │   ├─ LLM.generate → "审查代码..."
  │   │   返回 tool_calls: [load_skill("code-review")]
  │   │
  │   ├─ after_ai_message: StorageHook append AI msg
  │   ├─ before_tool: HitlHook → 放行
  │   ├─ load_skill → 返回 instruction + load_token
  │   ├─ StorageHook: append tool msg
  │   └─ ...
  │
  ├─ [step 2]
  │   ├─ LLM 按 code-review instruction 审查代码
  │   │   返回 tool_calls: [save_note("审查结果: ...")]
  │   │
  │   ├─ after_ai_message: StorageHook append AI msg
  │   ├─ before_tool: HitlHook → save_note 在审批列表！
  │   │   ConsoleApprovalProvider: "允许执行 save_note？[y/N]"
  │   │   用户输入 y → 放行
  │   ├─ save_note → {"ok": true}
  │   └─ StorageHook: append tool msg
  │
  ├─ [step 3]
  │   ├─ LLM.generate → finish_reason=stop
  │   ├─ after_ai_message: StorageHook append AI msg
  │   └─ yield done event
  │
  ▼
用户看到："审查完成，结果已保存"
```

---

## 十一、包结构

```
wuwei/
├── agent/                    Agent 入口和会话
│   ├── base.py              BaseAgent / BaseSessionAgent
│   ├── agent.py             Agent 单 Agent 门面
│   ├── plan_agent.py        PlanAgent 计划执行门面
│   └── session.py           AgentSession 会话容器
│
├── llm/                      LLM 调用层
│   ├── gateway.py           LLMGateway 统一入口
│   ├── types.py             Message / ToolCall / AgentEvent / AgentRunResult
│   └── adapters/
│       ├── base.py          BaseAdapter 协议
│       └── openai.py        OpenAI 适配器
│
├── memory/                   上下文与持久化
│   ├── context.py           Context 消息容器
│   ├── context_window.py    SimpleContextWindow 滑动窗口
│   ├── context_compressor.py  ContextCompressor 协议 + LLM 实现
│   └── storage.py           Storage 协议 + FileStorage
│
├── runtime/                  执行器与 Hook
│   ├── agent_runner.py      单 Agent 执行器
│   ├── planner_executor_runner.py  计划执行器
│   ├── hooks.py             RuntimeHook / HookManager
│   ├── skill_hook.py        SkillHook
│   ├── context_hook.py      ContextCompressionHook
│   ├── storage_hook.py      StorageHook
│   ├── hitl_hook.py         HitlHook
│   ├── hitl.py              ApprovalProvider / ApprovalPolicy
│   └── console_hook.py      ConsoleHook
│
├── tools/                    工具系统
│   ├── tool.py              Tool / ToolParameters
│   ├── registry.py          ToolRegistry
│   ├── executor.py          ToolExecutor
│   └── builtin/
│       ├── time_tools.py    get_now
│       ├── file_tools.py    file_to_md
│       └── skill_tools.py   list_skills / load_skill / run_skill_python_script
│
├── skill/                    Skill 体系
│   ├── skill.py             Skill / SkillProvider / SkillManager
│   └── fs_provider.py       FileSystemSkillProvider
│
└── planning/                 计划系统
    ├── planner.py           Planner
    └── task.py              Task / TaskList / PlanRunResult
```
