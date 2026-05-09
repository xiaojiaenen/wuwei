# Wuwei

Wuwei 是一个轻量、可扩展的 Python Agent 框架，目标是把模型调用、会话管理、工具注册与执行、plan-and-execute 拆成边界清晰的模块，方便学习和继续扩展。

## 目录结构

```text
wuwei/
├─ examples/          # 可直接运行的示例
├─ tests/             # pytest 测试
├─ wuwei/             # 框架源码
│  ├─ agent/          # Agent、PlanAgent、Session、基础抽象
│  ├─ runtime/        # AgentRunner、PlannerExecutorRunner、Hooks
│  ├─ planning/       # Planner、Task
│  ├─ memory/         # Context、MemoryStore、KnowledgeStore、Embedder、Storage
│  ├─ llm/            # LLMGateway、Types、Adapters
│  ├─ tools/          # Tool、Registry、Executor、Builtin Tools
│  └─ skill/          # Skill、SkillManager、SkillProvider
├─ pyproject.toml
└─ README.md
```

## 安装

要求 Python `>=3.10`。

```bash
pip install -e .
```

开发依赖：

```bash
pip install -e ".[dev]"
```

如果使用 `uv`：

```bash
uv sync
```

## 核心模块

| 模块 | 说明 |
|------|------|
| `wuwei.agent` | `Agent`：单 agent 门面；`PlanAgent`：plan-and-execute 门面；`AgentSession`：会话 |
| `wuwei.runtime` | `AgentRunner`：执行器；`PlannerExecutorRunner`：规划执行器；各种 Hook |
| `wuwei.planning` | `Planner`：任务规划器；`Task / TaskList`：任务模型 |
| `wuwei.memory` | `Context`：消息容器；`MemoryStore`：长期记忆；`KnowledgeStore`：RAG 知识库；`Embedder`：向量化 |
| `wuwei.llm` | `LLMGateway`：统一调用入口；`Message / ToolCall / LLMResponse`：类型定义 |
| `wuwei.tools` | `ToolRegistry`：工具注册；`ToolExecutor`：工具执行；内置 file/git/python/npm/calc/time/rag 工具 |
| `wuwei.skill` | `Skill / SkillManager`：技能管理；`FileSystemSkillProvider`：文件系统技能加载 |

## 快速开始

### 离线示例（不需要 API Key）

```bash
python examples/tool_executor_minimal.py
```

### 在线示例

```powershell
$env:WUWEI_API_KEY="your_key"
python examples/agent_minimal.py
python examples/agent_session_minimal.py
python examples/plan_agent_minimal.py
```

### 最小 Agent 示例

```python
from wuwei import Agent, LLMGateway, ToolRegistry

llm = LLMGateway.from_env()
tools = ToolRegistry.from_builtin(["time"])
agent = Agent(llm=llm, tools=tools)

result = await agent.run("现在几点了？")
print(result.content)
```

### 流式输出

```python
async for event in agent.stream_events("介绍一下自己"):
    if event.type == "text_delta":
        print(event.data["content"], end="", flush=True)
```

## 工具系统

### 使用内置工具

```python
from wuwei import ToolRegistry

# 注册单个或多个内置工具
registry = ToolRegistry.from_builtin(["time", "file", "git", "python", "calc"])
```

可用的内置工具：

| 名称 | 工具 |
|------|------|
| `time` | `get_now` — 获取当前时间 |
| `file` | `read_file` / `write_file` / `append_file` / `replace_in_file` / `delete_file` / `list_files` |
| `git` | `git_status` / `git_diff` / `git_log` / `git_show` / `git_add` / `git_commit` |
| `python` | `run_python` — 执行 Python 脚本 |
| `npm` | `npm_list_scripts` / `npm_run` / `npm_install` |
| `calc` | `calculate` — 安全数学表达式计算 |
| `rag` | `ingest_document` / `search_knowledge` — RAG 文档导入与检索 |

### 自定义工具

```python
from wuwei import ToolRegistry

registry = ToolRegistry()

# 方式一：装饰器
@registry.tool(name="add", description="两数相加")
def add(a: int, b: int) -> dict:
    return {"result": a + b}

# 方式二：register_callable
def multiply(x: int, y: int) -> int:
    return x * y
registry.register_callable(multiply)
```

## Hook 系统

Hook 是扩展 Agent 行为的主要方式。继承 `RuntimeHook`，重写需要的回调：

```python
from wuwei import RuntimeHook

class MyHook(RuntimeHook):
    async def before_llm(self, session, messages, tools, *, step, task=None):
        # 在 LLM 调用前修改 messages 或 tools
        return messages, tools

    async def after_tool(self, session, tool_call, tool_message, *, step, task=None, tool=None):
        # 工具执行后的副作用
        ...
```

注册到 Agent：

```python
agent = Agent(llm=llm, tools=tools, hooks=[MyHook()])
```

### 内置 Hook

| Hook | 说明 |
|------|------|
| `ConsoleHook` | 打印 LLM 调用、工具调用、任务生命周期到控制台 |
| `StorageHook` | 增量持久化每条消息到 `FileStorage` |
| `ContextCompressionHook` | 超过阈值时用 LLM 压缩旧轮次，配合滑动窗口裁剪上下文 |
| `SkillHook` | 将 skill 使用指引注入 system prompt |
| `HitlHook` | 工具调用前的人类审批拦截 |
| `MemoryRetrievalHook` | 每次 LLM 调用前检索长期记忆并注入 system prompt |
| `MemoryExtractionHook` | 每轮运行结束后用 LLM 抽取值得记住的信息 |
| `RagRetrievalHook` | 每次 LLM 调用前从知识库检索相关文档片段并注入 |

## 长期记忆

让 Agent 跨会话记住重要信息（用户偏好、关键决策、项目约束）。

```python
from wuwei import Agent, InMemoryMemoryStore, MemoryRetrievalHook, MemoryExtractionHook

memory_store = InMemoryMemoryStore()

agent = Agent(
    llm=llm,
    tools=tools,
    hooks=[
        MemoryRetrievalHook(memory_store),        # 检索注入
        MemoryExtractionHook(llm, memory_store),   # 自动抽取
    ],
)
```

**工作流程：**

1. 用户发消息 → `MemoryRetrievalHook` 从记忆库检索相关记忆 → 注入 system prompt
2. Agent 正常对话
3. 对话结束 → `MemoryExtractionHook` 用 LLM 分析对话 → 提取值得记住的信息 → 写入记忆库
4. 下次对话时，之前记住的信息会自动被检索到

**记忆衰减：**

```python
from wuwei.memory import decay_score

# 查看某条记忆的衰减分数
score = decay_score(record)

# 清理衰减分数低于 0.1 的记忆
deleted = await memory_store.cleanup(threshold=0.1)
```

衰减公式：`importance × 0.9^天数 × log₂(2 + access_count)`，重要且被频繁访问的记忆衰减更慢。

## RAG 知识库

让 Agent 从文档中检索相关内容。

```python
from wuwei import InMemoryKnowledgeStore, RagRetrievalHook

knowledge_store = InMemoryKnowledgeStore()

# 导入文档
await knowledge_store.ingest(
    "Wuwei 是一个轻量 Python Agent 框架...",
    source="README.md",
)

agent = Agent(
    llm=llm,
    tools=tools,
    hooks=[RagRetrievalHook(knowledge_store)],
)
```

**也可以通过工具让 Agent 自己导入文档：**

```python
from wuwei import ToolRegistry

tools = ToolRegistry.from_builtin(["rag"], knowledge_store=knowledge_store)
# Agent 就可以使用 ingest_document 和 search_knowledge 工具
```

## Embedding

默认使用关键词匹配，零依赖。接入向量检索只需传入 `Embedder`：

```python
from wuwei.memory import SimpleEmbedder, OpenAIEmbedder, InMemoryMemoryStore

# 零依赖，测试/演示用
embedder = SimpleEmbedder(dim=256)

# 生产环境用 OpenAI
embedder = OpenAIEmbedder(api_key="sk-xxx", model="text-embedding-3-small")

# 传入 store 即可自动启用向量检索
memory_store = InMemoryMemoryStore(embedder)
knowledge_store = InMemoryKnowledgeStore(embedder)
```

实现 `Embedder` 协议即可接入任何 Embedding 服务（Cohere、本地模型等）。

## HITL（人类审批）

拦截敏感工具调用，等待人类确认：

```python
from wuwei import HitlHook
from wuwei.runtime import ApprovalPolicy, ConsoleApprovalProvider

agent = Agent(
    llm=llm,
    tools=tools,
    hooks=[
        HitlHook(
            provider=ConsoleApprovalProvider(),  # 终端交互式审批
            policy=ApprovalPolicy(require_approval_tools={"write_file", "git_commit"}),
        ),
    ],
)
```

## Plan-and-Execute

`PlanAgent` 先用 LLM 将目标分解为任务 DAG，再按依赖顺序依次执行：

```python
from wuwei import PlanAgent

agent = PlanAgent(llm=llm, tools=tools)

# 流式执行
async for event in agent.stream_events("分析项目代码质量并生成报告"):
    if event.type == "task_start":
        print(f"开始: {event.data['task_description']}")
```

## Skill 系统

技能是基于 Markdown 的指令文件，支持附带脚本和参考资料：

```text
skills/
└─ code-review/
   ├─ SKILL.md              # 技能定义（YAML frontmatter + 正文）
   ├─ scripts/
   │  └─ count_lines.py     # 可执行脚本
   └─ references/
      └─ checklist.md       # 参考资料
```

```python
from wuwei import FileSystemSkillProvider, SkillManager

provider = FileSystemSkillProvider(skill_path="skills/")
manager = SkillManager([provider])
```

## AgentRunner 生命周期

```
用户输入
  → loop (最多 max_steps 轮):
      1. HookManager.before_llm(messages, tools)
         上下文裁剪、记忆注入、RAG 注入、skill 注入
      2. LLMGateway.generate(messages, tools)
      3. HookManager.after_llm(response)
      4. 如果有 tool_calls:
           HookManager.before_tool(tool_call)     → 审批检查
           ToolExecutor.execute_one(tool_call)     → 执行工具（带重试/超时）
           HookManager.after_tool(tool_call, result)
      5. 否则: 结束
  → HookManager.on_run_end(session, result)
     记忆抽取、持久化
  → 返回结果
```

## 测试

```bash
pytest                    # 全部测试
pytest tests/test_builtin_tools.py    # 单文件
pytest tests/test_builtin_tools.py -v # 详细输出
```

## 代码质量

```bash
ruff check wuwei/ tests/  # lint
black wuwei/ tests/       # 格式化
```

## 更多文档

- [docs/agent_framework_core_features.md](docs/agent_framework_core_features.md)：上下文压缩、滑动窗口、长期记忆、HITL 等核心能力设计指南
- [docs/long_term_memory_rag_plan.md](docs/long_term_memory_rag_plan.md)：长期记忆 & RAG 开发指南（含完整实现代码）
