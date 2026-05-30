# Wuwei 2.0 开发计划

> 基于 [REDESIGN.md](REDESIGN.md) 的改造方案，制定详细的开发计划。

## 一、方案优化清单

### 1.1 需要修复的问题

| # | 问题 | 修复方案 | 优先级 |
|---|------|----------|--------|
| 1 | 架构图重复 | 删除旧的单体架构图，只保留 Core + Extensions 架构 | P0 |
| 2 | 缺少 AG-UI | 新增 `wuwei-ext-agui` 扩展包，支持 Agent-User 交互协议 | P1 |
| 3 | 缺少 A2A | 新增 `wuwei-ext-a2a` 扩展包，支持 Agent-to-Agent 通信 | P2 |
| 4 | 缺少 YAML 配置 | 新增 `wuwei/config/` 模块，支持 YAML 定义 Agent | P1 |
| 5 | 缺少错误处理 | 新增错误处理、重试、熔断、降级策略 | P1 |
| 6 | 缺少测试策略 | 新增测试计划和测试工具 | P1 |
| 7 | 缺少迁移指南 | 新增 v1 → v2 迁移指南 | P1 |
| 8 | 缺少示例 | 新增完整使用示例 | P1 |
| 9 | 标题过时 | 更新为"八大框架" | P0 |
| 10 | 缺少安全考虑 | 新增安全最佳实践章节 | P2 |
| 11 | 缺少性能考虑 | 新增性能优化策略 | P2 |
| 12 | 缺少文档计划 | 新增文档策略 | P1 |

### 1.2 新增模块

```
wuwei/
├── ...（原有模块）
└── wuwei/config/             # 配置系统（新增）
    ├── yaml_loader.py        # YAML 配置加载
    └── schemas.py            # 配置 Schema
```

---

## 二、详细开发计划

### Phase 0: 准备工作（1 周）

**目标**：搭建项目结构，准备开发环境

| 任务 | 负责人 | 工时 | 产出 |
|------|--------|------|------|
| 创建 monorepo 结构 | — | 2 天 | 目录结构、pyproject.toml |
| 配置 CI/CD（GitHub Actions） | — | 1 天 | .github/workflows/ |
| 配置代码规范（ruff/mypy） | — | 1 天 | pyproject.toml 配置 |
| 编写 README.md | — | 1 天 | 项目说明文档 |
| 编写 CONTRIBUTING.md | — | 1 天 | 贡献指南 |

**产出物**：
```
wuwei/
├── wuwei/                    # 核心包
│   ├── pyproject.toml
│   └── src/wuwei/
├── wuwei-ext-graph/          # 图编排扩展
├── wuwei-ext-middleware/     # 中间件扩展
├── ...（其他扩展包）
├── .github/workflows/       # CI/CD
├── README.md
└── CONTRIBUTING.md
```

---

### Phase 1: 核心抽象层（2 周）

**目标**：实现 Core 包，建立框架基础

#### Week 1: 基础类型和接口

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Runnable 接口 | `invoke()`, `stream()`, `\|` 操作符 | `core/runnable.py` |
| 实现消息体系 | BaseMessage/AIMessage/ToolMessage | `core/message.py` |
| 实现工具基类 | Tool + Pydantic Schema 自动生成 | `tools/base.py` |
| 实现工具注册表 | ToolRegistry + 装饰器注册 | `tools/registry.py` |
| 实现工具执行器 | 并发执行 + 超时 + 重试 | `tools/executor.py` |

#### Week 2: Agent 核心

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Agent 类 | 线性循环 + 流式输出 | `agent/agent.py` |
| 实现会话管理 | AgentSession + Context | `agent/session.py` |
| 实现错误处理 | 重试 + 熔断 + 降级 | `core/errors.py` |
| 实现内置工具 | time/file/git/python/calc | `tools/builtin/` |
| 编写单元测试 | 核心模块测试 | `tests/` |

**验收标准**：
```python
# 最小可用示例
from wuwei import Agent, LLMGateway, Tool

llm = LLMGateway.from_env()
agent = Agent(llm=llm, tools=[...])
result = await agent.run("Hello")
```

---

### Phase 2: LLM 网关（1 周）

**目标**：支持多模型提供商

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现适配器基类 | BaseAdapter + 类型定义 | `llm/adapters/base.py` |
| 实现 OpenAI 适配器 | 默认适配器 | `llm/adapters/openai.py` |
| 实现 Anthropic 适配器 | Claude 系列 | `llm/adapters/anthropic.py` |
| 实现智谱适配器 | GLM-4 系列 | `llm/adapters/zhipu.py` |
| 实现 DashScope 适配器 | 通义千问系列 | `llm/adapters/dashscope.py` |
| 实现文心适配器 | ERNIE 系列 | `llm/adapters/ernie.py` |
| 实现 Ollama 适配器 | 本地模型 | `llm/adapters/ollama.py` |
| 实现 LLMGateway | 统一入口 + 重试 | `llm/gateway.py` |
| 编写集成测试 | 各适配器测试 | `tests/` |

**验收标准**：
```python
# 多模型切换
from wuwei.llm import LLMGateway

llm = LLMGateway(adapter="zhipu", model="glm-4", api_key="...")
agent = Agent(llm=llm, tools=tools)
result = await agent.run("用中文回答")
```

---

### Phase 3: 状态图编排（2 周）

**目标**：实现图编排能力

#### Week 1: 图构建

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 State 类 | 状态定义 + 消息管理 | `graph/state.py` |
| 实现 StateGraph | 节点 + 边 + 条件边 | `graph/graph.py` |
| 实现 CompiledGraph | 编译 + 执行 | `graph/graph.py` |
| 实现 Channel 系统 | 状态通道 | `graph/channels.py` |

#### Week 2: 检查点和预构建图

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现检查点基类 | BaseCheckpointer | `graph/checkpoint.py` |
| 实现内存检查点 | MemoryCheckpointer | `graph/checkpoint.py` |
| 实现 SQLite 检查点 | SQLiteCheckpointer | `graph/checkpoint.py` |
| 实现 ReAct 图 | 预构建 ReAct Agent | `graph/prebuilt/react.py` |
| 实现 PlanExecute 图 | 预构建计划执行 | `graph/prebuilt/plan_execute.py` |
| 编写集成测试 | 图执行测试 | `tests/` |

**验收标准**：
```python
# 状态图编排
from wuwei_ext_graph import StateGraph, State

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_node("tool", execute_tool)
graph.add_edge("llm", "tool")
graph.add_conditional_edges("tool", should_continue, {"llm": "llm", "end": END})
graph.set_entry_point("llm")
app = graph.compile()

state = await app.invoke(State(messages=[...]))
```

---

### Phase 4: 中间件系统（2 周）

**目标**：实现可组合中间件

#### Week 1: 中间件框架

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Middleware 基类 | 生命周期钩子 | `middleware/base.py` |
| 实现 MiddlewareStack | 栈管理 + 执行 | `middleware/stack.py` |
| 实现 MiddlewareContext | 上下文传递 | `middleware/base.py` |

#### Week 2: 预构建中间件

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现文件系统中间件 | 路径隔离 | `middleware/filesystem.py` |
| 实现记忆中间件 | 记忆注入/提取 | `middleware/memory.py` |
| 实现上下文压缩中间件 | 长对话压缩 | `middleware/context.py` |
| 实现子代理中间件 | 任务委派 | `middleware/subagent.py` |
| 实现 HITL 中间件 | 人工审批 | `middleware/hitl.py` |
| 编写集成测试 | 中间件测试 | `tests/` |

**验收标准**：
```python
# 中间件组合
from wuwei_ext_middleware import MiddlewareStack, MemoryMiddleware

stack = MiddlewareStack()
stack.add(MemoryMiddleware(memory_store, llm))
stack.add(ContextCompressionMiddleware(llm, max_turns=30))
stack.add(HitlMiddleware(approval_policy))

agent = Agent(llm=llm, tools=tools, middleware=stack)
```

---

### Phase 5: MCP 模块（2 周）

**目标**：支持 MCP 协议

#### Week 1: MCP 客户端

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 MCP 配置 | .mcp.json 多作用域 | `mcp/config.py` |
| 实现 Stdio 客户端 | 子进程通信 | `mcp/client.py` |
| 实现 HTTP 客户端 | HTTP/SSE 传输 | `mcp/client.py` |
| 实现工具适配器 | MCP 工具 → wuwei Tool | `mcp/tools.py` |

#### Week 2: 会话管理

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现会话管理器 | 有状态/无状态 | `mcp/session.py` |
| 实现工具发现 | 自动发现 MCP 工具 | `mcp/tools.py` |
| 实现错误处理 | 重连 + 熔断 | `mcp/errors.py` |
| 编写集成测试 | MCP 测试 | `tests/` |

**验收标准**：
```python
# MCP 工具
from wuwei_ext_mcp import MCPSessionManager, MCPConfig

config = MCPConfig.load(scopes=["project", "user"])
session = MCPSessionManager(config)
await session.connect_all()

tools = session.get_all_tools()  # 自动发现的 MCP 工具
agent = Agent(llm=llm, tools=tools)
```

---

### Phase 6: 技能系统（1 周）

**目标**：增强技能系统

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Skill 模型 | 结构化元数据 | `skill/skill.py` |
| 实现多源加载器 | 内置/用户/项目/MCP | `skill/loader.py` |
| 实现 SkillViewer 工具 | Agent 读取技能 | `skill/viewer.py` |
| 实现技能注册表 | 注册 + 查询 | `skill/registry.py` |
| 编写示例技能 | 5+ 示例技能 | `examples/skills/` |

**验收标准**：
```python
# 技能系统
from wuwei_ext_skill import SkillLoader, SkillRegistry, SkillViewerTool

loader = SkillLoader(["~/.wuwei/skills/", ".wuwei/skills/"])
skills = await loader.load_all()

registry = SkillRegistry()
for skill in skills:
    registry.register(skill)

viewer = SkillViewerTool(registry)
agent = Agent(llm=llm, tools=[viewer, ...])
```

---

### Phase 7: 多平台网关（2 周）

**目标**：支持多平台接入

#### Week 1: 网关框架 + 国内平台

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现网关基类 | BaseGateway + GatewayMessage | `gateway/base.py` |
| 实现微信网关 | 企业微信机器人 | `gateway/wechat.py` |
| 实现钉钉网关 | 机器人 + 工作通知 | `gateway/dingtalk.py` |
| 实现飞书网关 | 机器人 + 应用消息 | `gateway/feishu.py` |
| 实现企业微信网关 | 自建应用 | `gateway/wecom.py` |

#### Week 2: 海外平台 + Webhook

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Telegram 网关 | Bot API | `gateway/telegram.py` |
| 实现 Discord 网关 | Bot API | `gateway/discord.py` |
| 实现 Slack 网关 | Socket Mode | `gateway/slack.py` |
| 实现 Webhook 网关 | FastAPI 通用 | `gateway/webhook.py` |
| 编写集成测试 | 各平台测试 | `tests/` |

**验收标准**：
```python
# 多平台网关
from wuwei_ext_gateway import WeChatGateway, DingTalkGateway

gateways = [
    WeChatGateway(agent_factory, webhook_url="..."),
    DingTalkGateway(agent_factory, app_key="...", app_secret="..."),
]

await asyncio.gather(*[gw.start() for gw in gateways])
```

---

### Phase 8: 生产化（2 周）

**目标**：生产就绪

#### Week 1: 沙箱 + 可观测性

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Docker 沙箱 | 容器化执行 | `sandbox/docker.py` |
| 实现 E2B 沙箱 | 云沙箱 | `sandbox/e2b.py` |
| 实现 OpenTelemetry 追踪 | 分布式追踪 | `observability/tracing.py` |
| 实现指标收集 | 性能指标 | `observability/metrics.py` |

#### Week 2: 多 Agent + 文档

| 任务 | 描述 | 产出文件 |
|------|------|----------|
| 实现 Swarm 协作 | 多 Agent 团队 | `agent/multi_agent.py` |
| 编写 API 文档 | 完整 API 参考 | `docs/` |
| 编写教程 | 5+ 入门教程 | `docs/tutorials/` |
| 编写迁移指南 | v1 → v2 迁移 | `docs/migration.md` |
| 编写示例 | 10+ 完整示例 | `examples/` |

---

### Phase 9: 发布（1 周）

**目标**：正式发布 v2.0

| 任务 | 描述 | 产出物 |
|------|------|--------|
| 代码审查 | 全量代码审查 | 审查报告 |
| 性能测试 | 基准测试 | 性能报告 |
| 安全审计 | 安全检查 | 安全报告 |
| 文档审查 | 文档完整性 | 审查报告 |
| 发布到 PyPI | 正式发布 | wuwei 2.0.0 |
| 撰写发布公告 | 博客文章 | 发布说明 |

---

## 三、时间线总览

```
Week 1-2:   Phase 0 - 准备工作
Week 3-4:   Phase 1 - 核心抽象层
Week 5:     Phase 2 - LLM 网关
Week 6-7:   Phase 3 - 状态图编排
Week 8-9:   Phase 4 - 中间件系统
Week 10-11: Phase 5 - MCP 模块
Week 12:    Phase 6 - 技能系统
Week 13-14: Phase 7 - 多平台网关
Week 15-16: Phase 8 - 生产化
Week 17:    Phase 9 - 发布

总计：17 周（约 4 个月）
```

---

## 四、里程碑

| 里程碑 | 时间 | 交付物 |
|--------|------|--------|
| **M1: Core 可用** | Week 4 | `pip install wuwei` 可用 |
| **M2: 多模型支持** | Week 5 | 支持 6+ LLM 提供商 |
| **M3: 图编排可用** | Week 7 | `pip install wuwei[graph]` 可用 |
| **M4: 中间件可用** | Week 9 | `pip install wuwei[middleware]` 可用 |
| **M5: MCP 可用** | Week 11 | `pip install wuwei[mcp]` 可用 |
| **M6: 多平台可用** | Week 14 | 支持 8+ 平台 |
| **M7: v2.0 发布** | Week 17 | 正式发布 |

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM API 变更 | 适配器失效 | 抽象适配器层，快速适配 |
| MCP 协议不稳定 | MCP 模块需要重构 | 关注协议演进，保持兼容 |
| 国内平台 API 变更 | 网关失效 | 模块化设计，快速修复 |
| 性能瓶颈 | 用户体验差 | 基准测试 + 性能优化 |
| 安全漏洞 | 数据泄露 | 安全审计 + 最佳实践 |

---

## 六、测试策略

### 6.1 测试层次

```
├── 单元测试（每个模块）
│   ├── test_core.py          # 核心类型测试
│   ├── test_tool.py          # 工具系统测试
│   ├── test_agent.py         # Agent 测试
│   └── ...
│
├── 集成测试（模块间交互）
│   ├── test_llm_agent.py     # LLM + Agent 集成
│   ├── test_middleware.py    # 中间件集成
│   └── ...
│
└── 端到端测试（完整流程）
    ├── test_e2e_single.py    # 单 Agent 端到端
    ├── test_e2e_multi.py     # 多 Agent 端到端
    └── ...
```

### 6.2 测试工具

```python
# tests/fixtures.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_llm():
    """模拟 LLM"""
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(content="Hello")
    return llm

@pytest.fixture
def mock_tool():
    """模拟工具"""
    return Tool(
        name="test_tool",
        description="Test tool",
        handler=AsyncMock(return_value="result"),
    )
```

---

## 七、安全考虑

### 7.1 工具执行安全

- **路径隔离**：所有文件操作限制在工作空间内
- **命令白名单**：只允许执行预定义的命令
- **沙箱执行**：危险操作在 Docker/E2B 沙箱中执行
- **超时控制**：所有操作都有超时限制

### 7.2 LLM 调用安全

- **API Key 保护**：不硬编码，使用环境变量
- **输入验证**：验证所有用户输入
- **输出过滤**：过滤敏感信息
- **速率限制**：防止 API 滥用

### 7.3 MCP 安全

- **环境变量过滤**：子进程不继承敏感环境变量
- **凭证脱敏**：错误信息中脱敏凭证
- **配置验证**：验证 .mcp.json 配置

---

## 八、性能优化

### 8.1 核心优化

- **延迟导入**：扩展包按需加载
- **异步优先**：所有 I/O 操作异步
- **连接池**：HTTP 客户端使用连接池
- **缓存**：Schema、配置等缓存

### 8.2 扩展优化

- **并行工具执行**：多工具并发
- **流式输出**：LLM 流式响应
- **检查点压缩**：减少存储开销
- **记忆衰减**：自动清理过期记忆

---

*本开发计划基于 [REDESIGN.md](REDESIGN.md) 的改造方案，详细规划了 Wuwei 2.0 的开发流程。*
