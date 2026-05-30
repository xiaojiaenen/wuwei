# Wuwei 2.0 迁移指南

> 从 Wuwei 1.x 迁移到 2.0 的完整指南。

## 概述

Wuwei 2.0 是一次**破坏性重构**，不兼容旧版 API。主要变化：

1. **新增 core 模块**：Runnable 接口、消息体系、错误类型
2. **LLM 网关升级**：支持多提供商（OpenAI/Anthropic/智谱/DashScope/Ollama）
3. **新增状态图编排**：借鉴 LangGraph 的 StateGraph
4. **Hook → Middleware**：可组合中间件栈替代固定回调
5. **新增 MCP 支持**：Model Context Protocol 客户端
6. **增强技能系统**：结构化元数据 + SkillViewer 工具
7. **新增多平台网关**：微信/钉钉/飞书等

## 安装

```bash
# 卸载旧版
pip uninstall wuwei

# 安装新版
pip install wuwei

# 按需安装扩展
pip install wuwei[graph]        # 状态图
pip install wuwei[middleware]   # 中间件
pip install wuwei[mcp]          # MCP 支持
pip install wuwei[all]          # 全部扩展
```

## API 变化

### 1. Agent 类

**旧版**：
```python
from wuwei import Agent, LLMGateway

llm = LLMGateway.from_env()
agent = Agent(llm=llm, tools=[...])
result = await agent.run("Hello")
```

**新版**：
```python
from wuwei import Agent, LLMGateway

# 多提供商支持
llm = LLMGateway.from_env(provider="zhipu")  # 或 "anthropic", "dashscope", "ollama"
agent = Agent(llm=llm, tools=[...])
result = await agent.run("Hello")

# 新增 Runnable 接口
from wuwei.core import Runnable

class MyRunnable(Runnable):
    async def invoke(self, input, config=None):
        return f"处理: {input}"
```

### 2. LLM Gateway

**旧版**：
```python
from wuwei import LLMGateway

llm = LLMGateway.from_env()  # 只支持 OpenAI
```

**新版**：
```python
from wuwei import LLMGateway

# 自动检测 provider
llm = LLMGateway.from_env()

# 显式指定 provider
llm = LLMGateway.from_env(provider="anthropic")

# 配置文件方式
llm = LLMGateway({
    "provider": "zhipu",
    "api_key": "xxx",
    "model": "glm-4",
})
```

### 3. 工具系统

**旧版**：
```python
from wuwei import Tool

tool = Tool(
    name="search",
    description="搜索",
    parameters={"type": "object", "properties": {...}},
    handler=search_func,
)
```

**新版**：
```python
from wuwei import Tool, tool

# 方式1：从函数自动生成 Schema
@tool
def search(query: str, max_results: int = 5) -> str:
    """搜索文档"""
    return "结果"

# 方式2：手动定义
tool = Tool(
    name="search",
    description="搜索",
    parameters={"type": "object", "properties": {...}},
    handler=search_func,
)

# 方式3：使用 Tool.from_function
tool = Tool.from_function(search_func)
```

### 4. 状态图编排（新增）

```python
from wuwei.graph import StateGraph, State, END

# 定义节点
async def llm_node(state: State, config: dict) -> State:
    # 调用 LLM
    return state

async def tool_node(state: State, config: dict) -> State:
    # 执行工具
    return state

# 构建图
graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)
graph.add_edge("llm", "tool")
graph.add_conditional_edges("tool", should_continue, {"llm": "llm", "end": END})
graph.set_entry_point("llm")

# 编译并执行
app = graph.compile()
state = await app.invoke(State())
```

### 5. 中间件系统

**旧版（Hook）**：
```python
from wuwei import Agent
from wuwei.runtime import ConsoleHook, StorageHook

agent = Agent(
    llm=llm,
    tools=tools,
    hooks=[ConsoleHook(), StorageHook()],
)
```

**新版（Middleware）**：
```python
from wuwei import Agent
from wuwei.middleware import MiddlewareStack, LoggingMiddleware
from wuwei.middleware.hitl import HitlMiddleware

stack = MiddlewareStack()
stack.add(LoggingMiddleware())
stack.add(HitlMiddleware(approval_provider=my_approval))

agent = Agent(llm=llm, tools=tools, middleware=stack)
```

### 6. MCP 支持（新增）

```python
from wuwei.mcp import MCPConfig, MCPSessionManager

# 加载配置
config = MCPConfig.load(scopes=["project", "user"])

# 连接服务器
session = MCPSessionManager(config)
await session.connect_all()

# 获取工具
tools = session.get_all_tools()

# 使用工具
agent = Agent(llm=llm, tools=tools)
```

### 7. 技能系统

**旧版**：
```python
from wuwei import SkillManager, FileSystemSkillProvider

provider = FileSystemSkillProvider("skills/")
manager = SkillManager([provider])
```

**新版**：
```python
from wuwei import SkillManager, FileSystemSkillProvider
from wuwei.skill import SkillViewerTool

provider = FileSystemSkillProvider("skills/")
manager = SkillManager([provider])

# 新增：SkillViewer 工具
viewer = SkillViewerTool(manager)
agent = Agent(llm=llm, tools=[viewer, ...])
```

### 8. 多平台网关（新增）

```python
from wuwei.gateway import WebhookGateway

async def agent_factory():
    return Agent(llm=llm, tools=tools)

# Webhook 网关
gateway = WebhookGateway(
    agent_factory=agent_factory,
    host="0.0.0.0",
    port=8080,
)

# 微信网关
from wuwei.gateway.adapters import WeChatGateway

gateway = WeChatGateway(
    agent_factory=agent_factory,
    webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
)
```

## 环境变量变化

| 旧版 | 新版 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | `OPENAI_API_KEY` | OpenAI（默认） |
| — | `ANTHROPIC_API_KEY` | Anthropic |
| — | `ZHIPU_API_KEY` | 智谱 AI |
| — | `DASHSCOPE_API_KEY` | 阿里云 DashScope |
| — | `OLLAMA_BASE_URL` | Ollama（默认 localhost:11434） |

## 常见问题

### Q: 旧代码还能运行吗？

A: 不能。Wuwei 2.0 是破坏性重构，需要按照本指南修改代码。

### Q: 为什么要破坏性重构？

A: 为了：
1. 支持多提供商 LLM
2. 引入状态图编排
3. 改进扩展性（Hook → Middleware）
4. 支持 MCP 协议
5. 保持轻量（Core + Extensions 架构）

### Q: 迁移需要多长时间？

A: 取决于项目复杂度：
- 简单项目（单 Agent + 几个工具）：1-2 小时
- 中等项目（多 Agent + 状态管理）：1-2 天
- 复杂项目（完整框架集成）：3-5 天

### Q: 遇到问题怎么办？

A: 
1. 查看本文档
2. 查看 [用户使用文档](USAGE.md)
3. 查看 [示例代码](examples/)
4. 提交 Issue 到 GitHub

## 迁移检查清单

- [ ] 卸载旧版 wuwei
- [ ] 安装新版 wuwei
- [ ] 更新 import 语句
- [ ] 修改 LLM Gateway 初始化
- [ ] 修改工具定义（使用 @tool 装饰器）
- [ ] 修改 Hook 为 Middleware
- [ ] 测试所有功能
- [ ] 更新依赖版本
