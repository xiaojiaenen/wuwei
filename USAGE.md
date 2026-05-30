# Wuwei 2.0 用户使用文档

> Wuwei - 无为而治的 AI 智能体框架

## 目录

- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [LLM 网关](#llm-网关)
- [工具系统](#工具系统)
- [状态图编排](#状态图编排)
- [中间件系统](#中间件系统)
- [MCP 支持](#mcp-支持)
- [技能系统](#技能系统)
- [多平台网关](#多平台网关)
- [示例代码](#示例代码)

## 快速开始

### 安装

```bash
# 最小安装
pip install wuwei

# 按需安装扩展
pip install wuwei[graph]        # 状态图
pip install wuwei[middleware]   # 中间件
pip install wuwei[mcp]          # MCP 支持
pip install wuwei[all]          # 全部扩展
```

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

## 核心概念

### Runnable 接口

所有可执行组件都实现 Runnable 接口：

```python
from wuwei.core import Runnable, RunnableConfig

class MyRunnable(Runnable):
    async def invoke(self, input, config=None):
        return f"处理: {input}"

# 使用管道操作符
chain = MyRunnable() | MyRunnable()
result = await chain.invoke("hello")
```

### 消息体系

```python
from wuwei.core import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    ToolCall,
    FunctionCall,
)

# 创建消息
user_msg = HumanMessage(content="你好")
ai_msg = AIMessage(content="你好！")

# 转换为 OpenAI 格式
openai_format = user_msg.to_openai()
```

### 错误处理

```python
from wuwei.core import (
    WuweiError,
    ToolError,
    LLMError,
    TimeoutError,
    ValidationError,
)

try:
    result = await agent.run("test")
except ToolError as e:
    print(f"工具错误: {e}")
except LLMError as e:
    print(f"LLM 错误: {e}")
```

## LLM 网关

### 多提供商支持

```python
from wuwei import LLMGateway

# OpenAI（默认）
llm = LLMGateway.from_env(provider="openai")

# Anthropic
llm = LLMGateway.from_env(provider="anthropic")

# 智谱 AI
llm = LLMGateway.from_env(provider="zhipu")

# 阿里云 DashScope
llm = LLMGateway.from_env(provider="dashscope")

# Ollama（本地模型）
llm = LLMGateway.from_env(provider="ollama")
```

### 配置文件方式

```python
from wuwei import LLMGateway

llm = LLMGateway({
    "provider": "zhipu",
    "api_key": "your-api-key",
    "model": "glm-4",
    "temperature": 0.7,
    "max_tokens": 2048,
})
```

## 工具系统

### 使用 @tool 装饰器

```python
from wuwei import tool

@tool
def search(query: str, max_results: int = 5) -> str:
    """搜索文档

    Args:
        query: 搜索关键词
        max_results: 最大结果数
    """
    return f"找到 {max_results} 个结果"

@tool(name="custom_name", description="自定义描述")
def my_func(x: int) -> str:
    return str(x)
```

### 手动定义工具

```python
from wuwei import Tool

tool = Tool(
    name="search",
    description="搜索文档",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
        },
        "required": ["query"],
    },
    handler=search_func,
    timeout_seconds=30.0,
    requires_approval=False,
)
```

### 从函数自动生成

```python
from wuwei import Tool

def search(query: str, max_results: int = 5) -> str:
    """搜索文档"""
    return "结果"

tool = Tool.from_function(search)
```

## 状态图编排

### 基本用法

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
graph.add_edge("tool", END)
graph.set_entry_point("llm")

# 编译并执行
app = graph.compile()
state = await app.invoke(State())
```

### 条件边

```python
async def should_continue(state: State) -> str:
    """根据状态决定下一步"""
    last_msg = state.get_last_ai_message()
    if last_msg and last_msg.tool_calls:
        return "tool"
    return "end"

graph.add_conditional_edges(
    "llm",
    should_continue,
    {"tool": "tool", "end": END},
)
```

### 检查点

```python
from wuwei.graph import StateGraph, MemoryCheckpointer

graph = StateGraph(State)
# ... 添加节点和边

app = graph.compile()
app.set_checkpointer(MemoryCheckpointer())

# 执行时自动保存检查点
state = await app.invoke(State())
```

## 中间件系统

### 基本用法

```python
from wuwei.middleware import Middleware, MiddlewareContext, MiddlewareStack

class MyMiddleware(Middleware):
    async def before_llm(self, ctx):
        # LLM 调用前
        return ctx

    async def after_llm(self, ctx, response):
        # LLM 调用后
        return ctx

# 创建中间件栈
stack = MiddlewareStack()
stack.add(MyMiddleware())

# 使用
agent = Agent(llm=llm, tools=tools, middleware=stack)
```

### 内置中间件

```python
from wuwei.middleware import LoggingMiddleware
from wuwei.middleware.hitl import HitlMiddleware

# 日志中间件
stack.add(LoggingMiddleware())

# HITL 中间件
async def approval_provider(tool_call):
    print(f"是否允许 {tool_call.function.name}? (y/n)")
    return input().lower() == "y"

stack.add(HitlMiddleware(
    approval_provider=approval_provider,
    auto_approve_tools=["safe_tool"],
    auto_reject_tools=["dangerous_tool"],
))
```

## MCP 支持

### 配置

创建 `.mcp.json` 文件：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "transport": "stdio"
    },
    "remote-server": {
      "url": "https://mcp.example.com/mcp",
      "transport": "http"
    }
  }
}
```

### 使用

```python
from wuwei.mcp import MCPConfig, MCPSessionManager

# 加载配置
config = MCPConfig.load()

# 连接服务器
session = MCPSessionManager(config)
await session.connect_all()

# 获取工具
tools = session.get_all_tools()

# 使用工具
agent = Agent(llm=llm, tools=tools)
```

## 技能系统

### 定义技能

创建 `skills/my-skill/SKILL.md`：

```markdown
---
name: code-review
description: "代码审查技能"
version: 1.0.0
when_to_use: "当用户要求审查代码时"
allowed_tools:
  - read_file
  - grep
tags: [code-quality, review]
---

# 代码审查

## 检查清单

1. 代码风格
2. 错误处理
3. 安全性
```

### 使用

```python
from wuwei import SkillManager, FileSystemSkillProvider
from wuwei.skill import SkillViewerTool

provider = FileSystemSkillProvider("skills/")
manager = SkillManager([provider])

# 创建 SkillViewer 工具
viewer = SkillViewerTool(manager)
agent = Agent(llm=llm, tools=[viewer, ...])
```

## 多平台网关

### Webhook 网关

```python
from wuwei.gateway import WebhookGateway

async def agent_factory():
    return Agent(llm=llm, tools=tools)

gateway = WebhookGateway(
    agent_factory=agent_factory,
    host="0.0.0.0",
    port=8080,
)

await gateway.start()
```

### 微信网关

```python
from wuwei.gateway.adapters import WeChatGateway

gateway = WeChatGateway(
    agent_factory=agent_factory,
    webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
)

await gateway.start()
```

## 示例代码

查看 `examples/` 目录获取更多示例：

- `examples/minimal.py` - 最小示例
- `examples/graph.py` - 状态图示例
- `examples/middleware.py` - 中间件示例
- `examples/mcp.py` - MCP 示例
- `examples/skill.py` - 技能系统示例
- `examples/gateway.py` - 网关示例

## API 参考

详细的 API 参考请查看源代码中的 docstring，或访问：
- `wuwei.core` - 核心模块
- `wuwei.llm` - LLM 网关
- `wuwei.tools` - 工具系统
- `wuwei.graph` - 状态图
- `wuwei.middleware` - 中间件
- `wuwei.mcp` - MCP 支持
- `wuwei.skill` - 技能系统
- `wuwei.gateway` - 网关
