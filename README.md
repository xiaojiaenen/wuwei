# Wuwei 2.1

> 无为而治的 AI 智能体框架

Wuwei 2.1 是一个轻量、可扩展的 Python Agent 框架，借鉴了 LangChain、LangGraph、Deep Agents、Hermes Agent、AgentScope、Claude Code、agent-framework、rig 八大框架的优点。

## 特性

- 🚀 **轻量核心** — 核心包仅 ~50KB，按需安装扩展
- 🔌 **多提供商** — 支持 OpenAI、Anthropic、智谱、DashScope、Ollama
- 📊 **状态图编排** — 借鉴 LangGraph 的 StateGraph + Channels
- 🧩 **可组合中间件** — 替代固定 Hook，支持动态插拔
- 🔗 **MCP 支持** — Model Context Protocol 客户端
- 🌐 **多平台网关** — 微信、钉钉、飞书、Telegram、Webhook
- 🛡️ **沙箱执行** — Docker/E2B 隔离执行
- 📈 **可观测性** — OpenTelemetry 追踪
- 🤖 **多 Agent 协作** — Swarm 团队协作模式
- 📦 **插件系统** — 自动发现和加载插件
- 📝 **Output Parsers** — JSON/Pydantic/列表解析器
- 🔄 **上下文压缩** — 自动压缩长对话
- 📡 **流式模式** — 支持多种流式输出模式

## 安装

```bash
# 最小安装（核心）
pip install wuwei

# 按需安装扩展
pip install wuwei[graph]        # 状态图
pip install wuwei[middleware]   # 中间件
pip install wuwei[mcp]          # MCP 支持
pip install wuwei[skill]        # 技能系统
pip install wuwei[gateway]      # 多平台网关
pip install wuwei[memory]       # 记忆系统
pip install wuwei[observability] # 可观测性
pip install wuwei[sandbox]      # 沙箱执行
pip install wuwei[all]          # 全部扩展
```

## 快速开始

### 最小示例

```python
import asyncio
from wuwei import Agent, LLMGateway, tool

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 今天晴天，25°C"

# 创建 Agent
llm = LLMGateway.from_env()
agent = Agent(llm=llm, tools=[get_weather])

# 运行
async def main():
    result = await agent.run("北京今天天气怎么样？")
    print(result)

asyncio.run(main())
```

### 多提供商支持

```python
from wuwei import LLMGateway

# OpenAI（默认）
llm = LLMGateway.from_env(provider="openai")

# 智谱 AI
llm = LLMGateway.from_env(provider="zhipu")

# Anthropic
llm = LLMGateway.from_env(provider="anthropic")

# 阿里云 DashScope
llm = LLMGateway.from_env(provider="dashscope")

# Ollama（本地模型）
llm = LLMGateway.from_env(provider="ollama")
```

### 状态图编排

```python
from wuwei.graph import StateGraph, State, END
from wuwei.graph.channels import LastValue, Topic

async def llm_node(state: State, config: dict) -> State:
    # 调用 LLM
    return state

async def tool_node(state: State, config: dict) -> State:
    # 执行工具
    return state

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)
graph.add_edge("llm", "tool")
graph.add_edge("tool", END)
graph.set_entry_point("llm")

app = graph.compile()
state = await app.invoke(State())
```

### 中间件系统

```python
from wuwei.middleware import Middleware, MiddlewareStack, ContextCompressionMiddleware

class LoggingMiddleware(Middleware):
    async def before_llm(self, ctx):
        print(f"LLM 调用 - 消息数: {len(ctx.state.messages)}")
        return ctx

stack = MiddlewareStack()
stack.add(LoggingMiddleware())
stack.add(ContextCompressionMiddleware(llm=llm, trigger_tokens=4000))

agent = Agent(llm=llm, tools=tools, middleware=stack)
```

### Output Parsers

```python
from wuwei.parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# JSON 解析
parser = JsonOutputParser()
result = parser.parse('{"name": "test", "age": 25}')

# Pydantic 验证
parser = PydanticOutputParser(schema=User)
result = parser.parse('{"name": "test", "age": 25}')
# result = User(name="test", age=25)
```

### 插件系统

```python
from wuwei.plugin import PluginLoader, PluginRegistry

# 加载插件
loader = PluginLoader("plugins/")
plugins = loader.load_all()

# 注册插件
registry = PluginRegistry()
for plugin in plugins:
    registry.register(plugin)

# 获取插件钩子和工具
hooks = registry.get_hook("pre_tool_call")
tools = registry.list_tools()
```

### MCP 支持

```python
from wuwei.mcp import MCPConfig, MCPSessionManager

config = MCPConfig.load()
session = MCPSessionManager(config)
await session.connect_all()

tools = session.get_all_tools()
agent = Agent(llm=llm, tools=tools)
```

### 多 Agent 协作

```python
from wuwei.agent import Swarm, TeamMember

leader = Agent(llm=llm, tools=[...])
members = [
    TeamMember(name="researcher", agent=researcher, role="研究员"),
    TeamMember(name="writer", agent=writer, role="写手"),
]

swarm = Swarm(leader=leader, members=members)
result = await swarm.run("写一篇关于 AI 的文章")
```

## 项目结构

```
wuwei/
├── core/          # 核心抽象层（Runnable/消息/错误）
├── llm/           # LLM 网关（6 个适配器）
├── tools/         # 工具系统（11 组内置工具）
├── agent/         # Agent（单 Agent + 多 Agent Swarm）
├── graph/         # 状态图编排（StateGraph/检查点/Channels）
├── middleware/     # 中间件系统（日志/HITL/上下文压缩）
├── mcp/           # MCP 协议支持
├── skill/         # 技能系统
├── gateway/       # 多平台网关（微信/钉钉/飞书/Telegram/Webhook）
├── sandbox/       # 沙箱执行
├── observability/ # 可观测性（OpenTelemetry）
├── config/        # YAML 配置
├── plugin/        # 插件系统
├── parsers/       # Output Parsers
├── streaming/     # 流式模式
└── tests/         # 测试（72 个）
```

## 内置工具

| 工具组 | 工具 |
|--------|------|
| `time` | `get_now` — 获取当前时间 |
| `file` | `read_file` / `write_file` / `append_file` / `replace_in_file` / `delete_file` / `list_files` |
| `git` | `git_status` / `git_diff` / `git_log` / `git_show` / `git_add` / `git_commit` |
| `python` | `run_python` — 执行 Python 脚本 |
| `npm` | `npm_list_scripts` / `npm_run` / `npm_install` |
| `calc` | `calculate` — 安全数学表达式计算 |
| `rag` | `ingest_document` / `search_knowledge` — RAG 文档导入与检索 |
| `json` | `json_parse` / `json_extract` — JSON 处理 |
| `http` | `http_get` / `http_post` — HTTP 请求 |
| `text` | `text_replace` / `text_split` / `text_join` / `text_upper` / `text_lower` / `text_trim` |
| `skill` | `list_skills` / `load_skill` / `run_skill_script` — 技能系统 |

## 文档

- [迁移指南](MIGRATION.md) — 从 v1 迁移到 v2
- [用户使用文档](USAGE.md) — 完整使用指南
- [改造方案](REDESIGN.md) — 技术设计文档
- [开发计划](DEVPLAN.md) — 开发进度

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_core.py -v

# 运行框架完整性测试
pytest tests/test_framework.py -v
```

## 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

Apache-2.0
