# Wuwei 2.0 框架重构方案

> 吸取 LangChain、LangGraph、Deep Agents、Hermes Agent、AgentScope、Claude Code、agent-framework、rig 八大框架的优点，摒弃其缺点，对 Wuwei 进行**破坏性重构**。不兼容旧版 API。

## 一、现状分析与问题

### Wuwei 当前问题（必须摒弃的设计）

| 问题 | 说明 | 改进方向 |
|------|------|----------|
| Schema 体系单薄 | Tool 手动写 JSON Schema，缺少自动推断 | Pydantic 自动生成 |
| 无状态图编排 | 只有线性循环和简单 DAG | 状态图 + 检查点 |
| Hook 固定回调 | 不可组合，无法动态插拔 | 可组合中间件栈 |
| 仅 OpenAI | 只有一个适配器 | 多提供商原生支持 |
| 无沙箱执行 | 代码执行无隔离 | Docker/E2B 沙箱 |
| 无多 Agent | 只有单 Agent 和 PlanAgent | Swarm 团队协作 |
| 无可观测性 | 只有 Console 日志 | OpenTelemetry 追踪 |
| 无国内平台支持 | 缺少微信/钉钉/飞书等 | 多平台网关 |
| Gateway API 单一 | 只有海外平台（Telegram/Discord/Slack） | 增加国内平台 |

---

## 二、六大框架优缺点分析

### 1. LangChain

| 优点 | 缺点 |
|------|------|
| 标准化的 Runnable/LCEL 接口 | 抽象层过重，简单任务也要套多层 |
| 完善的工具 Schema 体系（Pydantic 自动生成） | 依赖链复杂，langchain-core 更新影响全生态 |
| 统一的消息体系（BaseMessage） | 文档碎片化，版本迭代快 |
| 回调追踪机制 | 学习曲线陡峭 |

### 2. LangGraph

| 优点 | 缺点 |
|------|------|
| 强大的状态图编排（StateGraph） | 简单场景过度设计 |
| 检查点 + 断点恢复 | 图构建 API 不直观 |
| Pregel 执行引擎（并行、容错） | 调试困难 |
| Human-in-the-Loop 原生支持 | 与 LangChain 耦合 |

### 3. Deep Agents

| 优点 | 缺点 |
|------|------|
| 中间件栈设计（可组合、可覆盖） | 过度 opinionated，灵活性受限 |
| 内置文件系统、子代理、上下文压缩 | 构建于 LangGraph 之上，依赖重 |
| Harness Profile 配置化 | Beta 阶段，API 不稳定 |

### 4. Hermes Agent

| 优点 | 缺点 |
|------|------|
| 多平台网关（20+ 消息平台） | 代码库巨大（~12k LOC 单文件） |
| 插件系统完善 | 架构复杂，入门门槛高 |
| 自进化技能系统 | 过于 opinionated |
| 300+ 模型支持 | 部分功能过度工程化 |

### 5. AgentScope

| 优点 | 缺点 |
|------|------|
| 生产就绪（Docker/E2B 沙箱） | 与阿里云生态耦合 |
| OpenTelemetry 可观测性 | 社区相对较小 |
| 多租户架构 | 文档以中文为主 |
| MCP/A2A 协议支持 | 部分功能需要云服务 |

### 6. Claude Code

| 优点 | 缺点 |
|------|------|
| 丰富的工具生态（30+ 内置工具） | TypeScript 实现，无法直接复用 |
| 插件市场机制 | 与 Claude 模型强绑定 |
| Swarm 团队协作 | 源码不完全开放 |
| 细粒度权限控制 | 复杂度高 |

---

## 三、改造目标

### 核心原则

1. **Core + Extensions** — 核心极简，一切功能皆可插件（借鉴 rig 的 Feature-flag 门控）
2. **保持轻量** — 不引入不必要的抽象层（吸取 LangChain 教训）
3. **图编排可选** — 简单场景用线性循环，复杂场景用状态图（吸取 LangGraph 教训）
4. **中间件可组合** — Hook 升级为可插拔中间件栈（吸取 Deep Agents 优点）
5. **多提供商原生** — 从设计之初支持多模型（吸取 Hermes 优点）
6. **声明式配置** — 支持 YAML 定义 Agent（借鉴 agent-framework）
7. **零依赖优先** — 默认实现不依赖外部服务

### 目标架构（Core + Extensions）

```
wuwei/
│
├── wuwei/                          # 核心包（pip install wuwei）
│   ├── __init__.py                 # 只导出核心 API
│   ├── core/                       # 核心抽象（必选）
│   │   ├── runnable.py             # 统一可执行接口
│   │   ├── message.py              # 统一消息体系
│   │   └── types.py                # 基础类型
│   │
│   ├── llm/                        # LLM 网关（必选）
│   │   ├── gateway.py              # 统一入口
│   │   └── adapters/
│   │       ├── openai.py           # OpenAI（默认）
│   │       └── base.py             # 适配器基类
│   │
│   ├── tools/                      # 工具系统（必选）
│   │   ├── base.py                 # BaseTool（Pydantic Schema 自动生成）
│   │   ├── registry.py             # 工具注册表
│   │   └── executor.py             # 工具执行器
│   │
│   └── agent/                      # Agent 核心（必选）
│       ├── agent.py                # 单 Agent（线性循环）
│       └── session.py              # 会话管理
│
├── wuwei-ext-graph/                # 扩展包（pip install wuwei[graph]）
│   └── wuwei_ext_graph/
│       ├── state.py                # 状态定义
│       ├── graph.py                # StateGraph 构建器
│       ├── checkpoint.py           # 检查点（内存/SQLite/Postgres）
│       └── prebuilt/               # 预构建图
│
├── wuwei-ext-middleware/           # 扩展包（pip install wuwei[middleware]）
│   └── wuwei_ext_middleware/
│       ├── base.py                 # Middleware 基类
│       ├── stack.py                # 中间件栈管理
│       ├── filesystem.py           # 文件系统中间件
│       ├── memory.py               # 记忆中间件
│       ├── context.py              # 上下文压缩中间件
│       ├── subagent.py             # 子代理中间件
│       └── hitl.py                 # Human-in-the-Loop 中间件
│
├── wuwei-ext-mcp/                  # 扩展包（pip install wuwei[mcp]）
│   └── wuwei_ext_mcp/
│       ├── client.py               # MCP 客户端（stdio/HTTP/SSE）
│       ├── config.py               # 配置管理（.mcp.json）
│       ├── tools.py                # MCP 工具适配器
│       └── session.py              # 会话管理
│
├── wuwei-ext-skill/                # 扩展包（pip install wuwei[skill]）
│   └── wuwei_ext_skill/
│       ├── skill.py                # Skill 数据模型
│       ├── loader.py               # 多源加载器
│       ├── viewer.py               # SkillViewer 工具
│       └── registry.py             # 技能注册表
│
├── wuwei-ext-gateway/              # 扩展包（pip install wuwei[gateway]）
│   └── wuwei_ext_gateway/
│       ├── base.py                 # 网关基类
│       ├── wechat.py               # 微信
│       ├── dingtalk.py             # 钉钉
│       ├── feishu.py               # 飞书
│       ├── wecom.py                # 企业微信
│       ├── telegram.py             # Telegram
│       └── webhook.py              # 通用 Webhook
│
├── wuwei-ext-memory/               # 扩展包（pip install wuwei[memory]）
│   └── wuwei_ext_memory/
│       ├── memory_store.py         # 长期记忆
│       ├── knowledge_store.py      # RAG 知识库
│       ├── embedder.py             # 嵌入模型
│       └── storage.py              # 持久化
│
├── wuwei-ext-observability/        # 扩展包（pip install wuwei[observability]）
│   └── wuwei_ext_observability/
│       ├── tracing.py              # OpenTelemetry 追踪
│       └── metrics.py              # 指标收集
│
└── wuwei-ext-sandbox/              # 扩展包（pip install wuwei[sandbox]）
    └── wuwei_ext_sandbox/
        ├── docker.py               # Docker 沙箱
        └── e2b.py                  # E2B 云沙箱
```

### 依赖矩阵

```
pip install wuwei                  # 核心（~50KB）
├── openai, pydantic, pyyaml       # 必选依赖

pip install wuwei[graph]           # + 状态图
pip install wuwei[middleware]      # + 中间件
pip install wuwei[mcp]             # + MCP 支持
pip install wuwei[skill]           # + 技能系统
pip install wuwei[gateway]         # + 多平台网关
pip install wuwei[memory]          # + 记忆系统
pip install wuwei[observability]   # + OpenTelemetry
pip install wuwei[sandbox]         # + 沙箱执行
pip install wuwei[all]             # 全部扩展
```

---

## 四、详细改造方案

### 4.0 核心设计：Core + Extensions

**借鉴**：rig 的 Feature-flag 门控 + agent-framework 的模块化架构

```python
# wuwei/__init__.py — 只导出核心 API
from wuwei.core import Runnable, BaseMessage, AIMessage, ToolMessage
from wuwei.llm import LLMGateway
from wuwei.tools import Tool, ToolRegistry
from wuwei.agent import Agent, AgentSession

# 扩展通过可选导入加载
def import_extension(name: str):
    """延迟导入扩展包"""
    import importlib
    return importlib.import_module(f"wuwei_ext_{name}")

# 使用示例
# from wuwei import Agent, LLMGateway, Tool  # 核心（~50KB）
# graph = import_extension("graph")           # 按需加载
# StateGraph = graph.StateGraph
```

**核心包大小对比**：

| 框架 | 核心包大小 | 完整安装大小 |
|------|-----------|-------------|
| LangChain | ~2MB (langchain-core) | ~50MB |
| LangGraph | ~500KB | ~5MB |
| Deep Agents | ~1MB | ~10MB |
| **Wuwei 2.0 Core** | **~50KB** | **~500KB** |

### 4.1 核心抽象层（core/）

**借鉴**：LangChain 的 Runnable 接口（轻量化版本）

```python
# core/runnable.py
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

class Runnable(ABC):
    """统一的可执行接口，但比 LangChain 更轻量"""

    @abstractmethod
    async def invoke(self, input: Any, config: dict = None) -> Any:
        """同步执行"""
        ...

    async def stream(self, input: Any, config: dict = None) -> AsyncIterator[Any]:
        """流式执行，默认回退到 invoke"""
        yield await self.invoke(input, config)

    def __or__(self, other: "Runnable") -> "Runnable":
        """支持 | 管道操作符（简化版 LCEL）"""
        return RunnableSequence(self, other)

class RunnableSequence(Runnable):
    """管道组合"""

    def __init__(self, *runnables: Runnable):
        self.runnables = runnables

    async def invoke(self, input: Any, config: dict = None) -> Any:
        result = input
        for runnable in self.runnables:
            result = await runnable.invoke(result, config)
        return result
```

**核心改进**：
- 只保留 `invoke` 和 `stream` 两个核心方法
- 支持管道操作符但不强制使用
- 比 LangChain 的 Runnable（10+ 方法）简单得多

### 4.2 统一消息体系（core/message.py）

**借鉴**：LangChain 的 BaseMessage + Wuwei 的 Message

```python
# core/message.py
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid

class BaseMessage(BaseModel):
    """统一消息基类"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        return {"role": self.role, "content": self.content}

class AIMessage(BaseMessage):
    """AI 消息，携带工具调用"""
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None

    def to_openai(self) -> dict:
        result = super().to_openai()
        if self.tool_calls:
            result["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return result

class ToolMessage(BaseMessage):
    """工具执行结果"""
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    status: Literal["success", "error"] = "success"

class ToolCall(BaseModel):
    """工具调用"""
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    type: Literal["function"] = "function"
    function: FunctionCall

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_openai()
        }

class FunctionCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)

    def to_openai(self) -> dict:
        import json
        return {"name": self.name, "arguments": json.dumps(self.arguments)}
```

**核心改进**：
- 统一的消息基类，支持 OpenAI/Anthropic 格式互转
- 消息自带序列化能力
- 保留 Wuwei 的简洁性，增加 LangChain 的完整性

### 4.3 工具 Schema 系统（tools/base.py）

**借鉴**：LangChain 的 Pydantic 自动生成 + Wuwei 的简洁性

```python
# tools/base.py
from pydantic import BaseModel, Field, create_model
from typing import Any, Callable, Optional, get_type_hints
import inspect
import json

class Tool(BaseModel):
    """增强版工具，支持 Pydantic Schema 自动生成"""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    handler: Callable = Field(exclude=True)
    requires_approval: bool = False
    side_effect: bool = False
    timeout_seconds: float = 60.0

    # 新增：注入参数支持（借鉴 LangChain）
    injected_params: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: str = None,
        description: str = None,
        injected_params: list[str] = None,
    ) -> "Tool":
        """从函数自动生成 Schema（借鉴 LangChain）"""
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        # 自动生成 Pydantic 模型
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in (injected_params or []):
                continue  # 跳过注入参数
            param_type = hints.get(param_name, str)
            param_default = param.default if param.default != inspect.Parameter.empty else ...
            fields[param_name] = (param_type, param_default)

        schema_model = create_model(f"{func.__name__}_args", **fields)

        return cls(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            parameters=cls._pydantic_to_json_schema(schema_model),
            handler=func,
            injected_params=injected_params or [],
        )

    @staticmethod
    def _pydantic_to_json_schema(model: type[BaseModel]) -> dict:
        """Pydantic 模型转 JSON Schema"""
        schema = model.model_json_schema()
        # 转换为 OpenAI function calling 格式
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    async def invoke(self, args: dict, config: dict = None) -> str:
        """执行工具"""
        # 过滤注入参数
        filtered_args = {
            k: v for k, v in args.items()
            if k not in self.injected_params
        }

        # Pydantic 验证（借鉴 LangChain）
        if self.parameters:
            self._validate_args(filtered_args)

        # 执行
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**filtered_args)
        else:
            return self.handler(**filtered_args)

    def _validate_args(self, args: dict):
        """参数验证"""
        # 简单验证：检查必需参数
        required = self.parameters.get("required", [])
        for param in required:
            if param not in args:
                raise ValueError(f"Missing required parameter: {param}")
```

**核心改进**：
- 从函数自动生成 Pydantic Schema（借鉴 LangChain）
- 支持注入参数（InjectedToolArg），LLM 无法伪造
- 保留 Wuwei 的简洁 API
- 比 LangChain 的 BaseTool（1000+ 行）简单得多

### 4.4 状态图编排（graph/）

**借鉴**：LangGraph 的 StateGraph + Pregel 执行引擎（轻量化版本）

```python
# graph/graph.py
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

@dataclass
class State:
    """图状态"""
    messages: list[BaseMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_message(self, message: BaseMessage):
        self.messages.append(message)

    def get_last_ai_message(self) -> Optional[AIMessage]:
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

class StateGraph:
    """状态图构建器（轻量版 LangGraph）"""

    def __init__(self, state_type: type[State] = State):
        self.state_type = state_type
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | Callable] = {}
        self.entry_point: str = None

    def add_node(self, name: str, func: Callable):
        """添加节点"""
        self.nodes[name] = func
        return self

    def add_edge(self, source: str, target: str):
        """添加边"""
        self.edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: Callable,
        targets: dict[str, str]
    ):
        """添加条件边"""
        self.edges[source] = (condition, targets)
        return self

    def set_entry_point(self, name: str):
        """设置入口点"""
        self.entry_point = name
        return self

    def compile(self) -> CompiledGraph:
        """编译图为可执行图"""
        return CompiledGraph(self)

class CompiledGraph:
    """编译后的可执行图"""

    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.checkpointer = None

    def set_checkpointer(self, checkpointer):
        """设置检查点（借鉴 LangGraph）"""
        self.checkpointer = checkpointer

    async def invoke(self, input: State, config: dict = None) -> State:
        """执行图"""
        state = input
        current_node = self.graph.entry_point

        while current_node:
            # 执行节点
            node_func = self.graph.nodes[current_node]
            state = await node_func(state, config)

            # 保存检查点
            if self.checkpointer:
                await self.checkpointer.save(state)

            # 确定下一个节点
            edge = self.graph.edges.get(current_node)
            if edge is None:
                break
            elif isinstance(edge, str):
                current_node = edge
            else:
                condition, targets = edge
                next_key = await condition(state)
                current_node = targets.get(next_key)

        return state

    async def ainvoke(self, input: State, config: dict = None) -> State:
        """异步执行图（别名）"""
        return await self.invoke(input, config)
```

**核心改进**：
- 只保留核心的图构建和执行能力
- 条件边支持（LangGraph 核心特性）
- 检查点可选（不强制依赖）
- 比 LangGraph 的 Pregel 引擎简单得多

### 4.5 检查点系统（graph/checkpoint.py）

**借鉴**：LangGraph 的检查点机制

```python
# graph/checkpoint.py
from abc import ABC, abstractmethod
from typing import Any
import json
import os
from datetime import datetime

class BaseCheckpointer(ABC):
    """检查点基类"""

    @abstractmethod
    async def save(self, state: State, checkpoint_id: str = None) -> str:
        """保存检查点，返回 checkpoint_id"""
        ...

    @abstractmethod
    async def load(self, checkpoint_id: str) -> State:
        """加载检查点"""
        ...

    @abstractmethod
    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        """列出检查点"""
        ...

class MemoryCheckpointer(BaseCheckpointer):
    """内存检查点（默认）"""

    def __init__(self):
        self.checkpoints: dict[str, dict] = {}

    async def save(self, state: State, checkpoint_id: str = None) -> str:
        checkpoint_id = checkpoint_id or f"cp_{datetime.now().isoformat()}"
        self.checkpoints[checkpoint_id] = {
            "state": state.__dict__,
            "timestamp": datetime.now().isoformat(),
        }
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> State:
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        return State(**self.checkpoints[checkpoint_id]["state"])

    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        items = list(self.checkpoints.items())[-limit:]
        return [{"id": k, **v} for k, v in items]

class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite 检查点（借鉴 LangGraph）"""

    def __init__(self, db_path: str = ".wuwei/checkpoints.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import sqlite3
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.close()

    async def save(self, state: State, checkpoint_id: str = None) -> str:
        import sqlite3
        checkpoint_id = checkpoint_id or f"cp_{datetime.now().isoformat()}"
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (id, state) VALUES (?, ?)",
            (checkpoint_id, json.dumps(state.__dict__, default=str))
        )
        conn.commit()
        conn.close()
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> State:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT state FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        conn.close()
        if not row:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        return State(**json.loads(row[0]))
```

### 4.6 中间件系统（middleware/）

**借鉴**：Deep Agents 的中间件栈设计

```python
# middleware/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class MiddlewareContext:
    """中间件上下文"""
    state: State
    config: dict
    step: int
    tool_calls: list[ToolCall] = None

class Middleware(ABC):
    """中间件基类（借鉴 Deep Agents）"""

    @abstractmethod
    async def process(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """处理中间件逻辑，返回修改后的上下文"""
        ...

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前"""
        return ctx

    async def after_llm(self, ctx: MiddlewareContext, response: AIMessage) -> MiddlewareContext:
        """LLM 调用后"""
        return ctx

    async def before_tool(self, ctx: MiddlewareContext, tool_call: ToolCall) -> ToolCall:
        """工具执行前"""
        return tool_call

    async def after_tool(self, ctx: MiddlewareContext, tool_message: ToolMessage) -> ToolMessage:
        """工具执行后"""
        return tool_message

class MiddlewareStack:
    """中间件栈管理器"""

    def __init__(self):
        self.middlewares: list[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewareStack":
        """添加中间件"""
        self.middlewares.append(middleware)
        return self

    async def execute_before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """执行所有 before_llm 中间件"""
        for mw in self.middlewares:
            ctx = await mw.before_llm(ctx)
        return ctx

    async def execute_after_llm(self, ctx: MiddlewareContext, response: AIMessage) -> MiddlewareContext:
        """执行所有 after_llm 中间件"""
        for mw in self.middlewares:
            ctx = await mw.after_llm(ctx, response)
        return ctx

    async def execute_before_tool(self, ctx: MiddlewareContext, tool_call: ToolCall) -> ToolCall:
        """执行所有 before_tool 中间件"""
        for mw in self.middlewares:
            tool_call = await mw.before_tool(ctx, tool_call)
        return tool_call

    async def execute_after_tool(self, ctx: MiddlewareContext, tool_message: ToolMessage) -> ToolMessage:
        """执行所有 after_tool 中间件"""
        for mw in self.middlewares:
            tool_message = await mw.after_tool(ctx, tool_message)
        return tool_message
```

**预构建中间件**：

```python
# middleware/filesystem.py
class FilesystemMiddleware(Middleware):
    """文件系统中间件（借鉴 Deep Agents）"""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    async def before_tool(self, ctx: MiddlewareContext, tool_call: ToolCall) -> ToolCall:
        """注入文件系统工具"""
        # 自动添加 ls, read_file, write_file 等工具
        if tool_call.function.name in ["ls", "read_file", "write_file"]:
            # 验证路径在工作空间内
            path = tool_call.function.arguments.get("path", "")
            if not self._is_within_workspace(path):
                raise ValueError(f"Path {path} is outside workspace")
        return tool_call

    def _is_within_workspace(self, path: str) -> bool:
        import os
        abs_path = os.path.abspath(os.path.join(self.workspace_root, path))
        return abs_path.startswith(os.path.abspath(self.workspace_root))

# middleware/memory.py
class MemoryMiddleware(Middleware):
    """记忆中间件"""

    def __init__(self, memory_store: MemoryStore, llm: LLMGateway):
        self.memory_store = memory_store
        self.llm = llm

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """注入相关记忆"""
        last_message = ctx.state.get_last_user_message()
        if last_message:
            memories = await self.memory_store.search(last_message.content, limit=5)
            if memories:
                memory_text = "\n".join([m.content for m in memories])
                ctx.state.add_message(BaseMessage(
                    role="system",
                    content=f"相关记忆：\n{memory_text}"
                ))
        return ctx

    async def after_llm(self, ctx: MiddlewareContext, response: AIMessage) -> MiddlewareContext:
        """提取记忆"""
        # 使用 LLM 分析对话，提取值得记忆的信息
        # ... 实现记忆提取逻辑
        return ctx

# middleware/context.py
class ContextCompressionMiddleware(Middleware):
    """上下文压缩中间件"""

    def __init__(self, llm: LLMGateway, max_turns: int = 30, keep_recent: int = 10):
        self.llm = llm
        self.max_turns = max_turns
        self.keep_recent = keep_recent

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """压缩过长的上下文"""
        messages = ctx.state.messages
        if len(messages) > self.max_turns:
            # 压缩旧消息
            old_messages = messages[:-self.keep_recent]
            recent_messages = messages[-self.keep_recent:]

            summary = await self._compress(old_messages)
            ctx.state.messages = [
                BaseMessage(role="system", content=f"对话摘要：\n{summary}")
            ] + recent_messages
        return ctx

    async def _compress(self, messages: list[BaseMessage]) -> str:
        """使用 LLM 压缩消息"""
        # ... 实现压缩逻辑
        return ""

# middleware/subagent.py
class SubAgentMiddleware(Middleware):
    """子代理中间件（借鉴 Deep Agents）"""

    def __init__(self, agent_factory):
        self.agent_factory = agent_factory

    async def before_tool(self, ctx: MiddlewareContext, tool_call: ToolCall) -> ToolCall:
        """当工具调用是 'task' 时，创建子代理执行"""
        if tool_call.function.name == "task":
            task_description = tool_call.function.arguments.get("description", "")
            # 创建子代理
            sub_agent = self.agent_factory()
            result = await sub_agent.run(task_description)
            # 返回结果作为工具消息
            return ToolMessage(
                role="tool",
                content=result,
                tool_call_id=tool_call.id,
                name="task",
            )
        return tool_call
```

### 4.7 多提供商 LLM 网关（llm/adapters/）

**借鉴**：Hermes Agent 的多提供商支持

```python
# llm/adapters/anthropic.py
class AnthropicAdapter(BaseAdapter):
    """Anthropic 适配器"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(self, messages: list[Message], tools: list[dict] = None) -> LLMResponse:
        # 转换消息格式
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # 调用 API
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_msg,
            messages=anthropic_messages,
            tools=tools,
        )

        # 解析响应
        return self._parse_response(response)

# llm/adapters/ollama.py
class OllamaAdapter(BaseAdapter):
    """Ollama 适配器（本地模型）"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        from httpx import AsyncClient
        self.client = AsyncClient(base_url=base_url)
        self.model = model

    async def generate(self, messages: list[Message], tools: list[dict] = None) -> LLMResponse:
        # Ollama 使用 OpenAI 兼容格式
        response = await self.client.post("/v1/chat/completions", json={
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "tools": tools,
        })
        return self._parse_response(response.json())

# llm/adapters/zhipu.py
class ZhipuAdapter(BaseAdapter):
    """智谱 AI 适配器（国内）"""

    def __init__(self, api_key: str, model: str = "glm-4"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4"
        )
        self.model = model

    async def generate(self, messages: list[Message], tools: list[dict] = None) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            tools=tools,
        )
        return self._parse_response(response)

# llm/adapters/dashscope.py
class DashScopeAdapter(BaseAdapter):
    """阿里云 DashScope 适配器（国内）"""

    def __init__(self, api_key: str, model: str = "qwen-max"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

    async def generate(self, messages: list[Message], tools: list[dict] = None) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            tools=tools,
        )
        return self._parse_response(response)

# llm/adapters/ernie.py
class ErnieAdapter(BaseAdapter):
    """百度文心一言适配器（国内）"""

    def __init__(self, api_key: str, secret_key: str, model: str = "ernie-4.0"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.model = model
        self._access_token = None

    async def _get_access_token(self) -> str:
        """获取百度 API Access Token"""
        from httpx import AsyncClient
        async with AsyncClient() as client:
            resp = await client.get(
                "https://aip.baidubce.com/oauth/2.0/token",
                params={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.secret_key,
                }
            )
            return resp.json()["access_token"]

    async def generate(self, messages: list[Message], tools: list[dict] = None) -> LLMResponse:
        if not self._access_token:
            self._access_token = await self._get_access_token()

        from httpx import AsyncClient
        async with AsyncClient() as client:
            resp = await client.post(
                f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model}",
                params={"access_token": self._access_token},
                json={
                    "messages": [m.to_openai() for m in messages],
                    "tools": tools,
                }
            )
            return self._parse_response(resp.json())
```

### 4.8 沙箱执行（tools/sandbox/）

**借鉴**：AgentScope 的 Docker/E2B 沙箱

```python
# tools/sandbox/base.py
class BaseSandbox(ABC):
    """沙箱基类"""

    @abstractmethod
    async def execute(self, command: str, timeout: float = 30) -> dict:
        """执行命令"""
        ...

    @abstractmethod
    async def cleanup(self):
        """清理资源"""
        ...

# tools/sandbox/docker.py
class DockerSandbox(BaseSandbox):
    """Docker 沙箱（借鉴 AgentScope）"""

    def __init__(self, image: str = "python:3.11-slim", workspace: str = "/workspace"):
        self.image = image
        self.workspace = workspace
        self.container = None

    async def setup(self):
        """启动容器"""
        import aiodocker
        docker = aiodocker.Docker()
        self.container = await docker.containers.create_or_replace(
            name=f"wuwei-sandbox-{id(self)}",
            config={
                "Image": self.image,
                "WorkingDir": self.workspace,
                "HostConfig": {
                    "Binds": [f"{os.getcwd()}:{self.workspace}"],
                    "NetworkMode": "none",  # 网络隔离
                },
            }
        )
        await self.container.start()

    async def execute(self, command: str, timeout: float = 30) -> dict:
        """在容器中执行命令"""
        exec_obj = await self.container.exec(
            cmd=["sh", "-c", command],
            stdout=True,
            stderr=True,
        )
        # 读取输出
        output = await exec_obj.start()
        return {"stdout": output.decode(), "stderr": "", "exit_code": 0}

    async def cleanup(self):
        """停止并删除容器"""
        if self.container:
            await self.container.stop()
            await self.container.delete()

# tools/sandbox/e2b.py
class E2BSandbox(BaseSandbox):
    """E2B 云端沙箱（借鉴 AgentScope）"""

    def __init__(self, api_key: str):
        from e2b_code_interpreter import AsyncSandbox
        self.sandbox = AsyncSandbox(api_key=api_key)

    async def execute(self, command: str, timeout: float = 30) -> dict:
        execution = await self.sandbox.run_code(command)
        return {
            "stdout": execution.text,
            "stderr": execution.error if execution.error else "",
            "exit_code": 0 if not execution.error else 1,
        }

    async def cleanup(self):
        await self.sandbox.kill()
```

### 4.9 多 Agent 协作（agent/multi_agent.py）

**借鉴**：Claude Code 的 Swarm 机制

```python
# agent/multi_agent.py
from typing import Any, Callable
import asyncio

@dataclass
class TeamMember:
    """团队成员"""
    name: str
    agent: Agent
    role: str
    tools: list[str] = Field(default_factory=list)

class Swarm:
    """多 Agent 协作（借鉴 Claude Code Swarm）"""

    def __init__(self, leader: Agent, members: list[TeamMember]):
        self.leader = leader
        self.members = {m.name: m for m in members}
        self.handoff_history: list[dict] = []

    async def run(self, task: str) -> str:
        """执行任务，支持 Agent 间协作"""
        # 领导者分解任务
        subtasks = await self._decompose_task(task)

        results = {}
        for subtask in subtasks:
            # 分配给合适的成员
            assigned_to = await self._assign_task(subtask)
            member = self.members[assigned_to]

            # 执行子任务
            result = await member.agent.run(
                subtask,
                context=self._build_context(results)
            )
            results[subtask.id] = result

            # 记录协作历史
            self.handoff_history.append({
                "from": "leader",
                "to": assigned_to,
                "task": subtask.description,
                "result": result,
            })

        # 领导者汇总结果
        return await self._synthesize_results(results)

    async def _decompose_task(self, task: str) -> list[SubTask]:
        """分解任务"""
        # 使用领导者的 LLM 分解任务
        ...

    async def _assign_task(self, subtask: SubTask) -> str:
        """分配任务给合适的成员"""
        # 根据成员角色和工具匹配
        ...

    def _build_context(self, results: dict) -> str:
        """构建上下文"""
        context_parts = []
        for task_id, result in results.items():
            context_parts.append(f"任务 {task_id} 的结果：{result}")
        return "\n\n".join(context_parts)

    async def _synthesize_results(self, results: dict) -> str:
        """汇总所有结果"""
        # 使用领导者 LLM 汇总
        ...
```

### 4.10 多平台网关（gateway/）

**借鉴**：Hermes Agent 的多平台网关 + 国内平台支持

```python
# gateway/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass

@dataclass
class GatewayMessage:
    """统一网关消息格式"""
    platform: str              # 平台标识
    message_id: str            # 平台消息 ID
    user_id: str               # 用户 ID
    user_name: str             # 用户名
    content: str               # 消息内容
    message_type: str          # text/image/file/...
    reply_to: str = None       # 回复的消息 ID
    metadata: dict = None      # 平台特有数据

class BaseGateway(ABC):
    """网关基类"""

    def __init__(self, agent_factory):
        self.agent_factory = agent_factory

    @abstractmethod
    async def start(self):
        """启动网关"""
        ...

    @abstractmethod
    async def stop(self):
        """停止网关"""
        ...

    @abstractmethod
    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息"""
        ...

    @abstractmethod
    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收消息流"""
        ...

    async def handle_message(self, message: GatewayMessage) -> str:
        """处理消息（默认实现：调用 Agent）"""
        agent = self.agent_factory()
        result = await agent.run(message.content)
        return result

# gateway/adapters/wechat.py
class WeChatGateway(BaseGateway):
    """微信网关（通过企业微信机器人或第三方服务）"""

    def __init__(self, agent_factory, webhook_url: str = None, token: str = None):
        super().__init__(agent_factory)
        self.webhook_url = webhook_url  # 企业微信群机器人 Webhook
        self.token = token              # 服务号 Token
        self._message_queue = asyncio.Queue()

    async def start(self):
        """启动微信消息监听"""
        if self.webhook_url:
            # 企业微信群机器人模式
            await self._start_webhook_listener()
        elif self.token:
            # 服务号模式（需要公网域名）
            await self._start_service_account_listener()

    async def _start_webhook_listener(self):
        """监听企业微信群机器人消息"""
        from httpx import AsyncClient
        # 长轮询或 WebSocket 连接
        ...

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送微信消息"""
        from httpx import AsyncClient
        async with AsyncClient() as client:
            await client.post(self.webhook_url, json={
                "msgtype": "text",
                "text": {"content": content}
            })

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收微信消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(
                platform="wechat",
                message_id=raw_msg["MsgId"],
                user_id=raw_msg["FromUserName"],
                user_name=raw_msg.get("FromUserName", ""),
                content=raw_msg["Content"],
                message_type="text",
            )

# gateway/adapters/dingtalk.py
class DingTalkGateway(BaseGateway):
    """钉钉网关（支持机器人和工作通知）"""

    def __init__(self, agent_factory, app_key: str, app_secret: str, robot_code: str = None):
        super().__init__(agent_factory)
        self.app_key = app_key
        self.app_secret = app_secret
        self.robot_code = robot_code
        self._access_token = None
        self._token_expires = 0

    async def _get_access_token(self) -> str:
        """获取钉钉 Access Token"""
        import time
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        from httpx import AsyncClient
        async with AsyncClient() as client:
            resp = await client.post(
                "https://api.dingtalk.com/v1.0/oauth2/accessToken",
                json={
                    "appKey": self.app_key,
                    "appSecret": self.app_secret,
                }
            )
            data = resp.json()
            self._access_token = data["accessToken"]
            self._token_expires = time.time() + data["expireIn"] - 60
            return self._access_token

    async def start(self):
        """启动钉钉消息监听"""
        # 使用 Stream 模式或 WebSocket 接收消息
        ...

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送钉钉消息"""
        token = await self._get_access_token()
        from httpx import AsyncClient
        async with AsyncClient() as client:
            await client.post(
                "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend",
                headers={"x-acs-dingtalk-access-token": token},
                json={
                    "robotCode": self.robot_code,
                    "userIds": [user_id],
                    "msgKey": "sampleText",
                    "msgParam": json.dumps({"content": content}),
                }
            )

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收钉钉消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(
                platform="dingtalk",
                message_id=raw_msg.get("msgId", ""),
                user_id=raw_msg.get("senderStaffId", ""),
                user_name=raw_msg.get("senderNick", ""),
                content=raw_msg.get("text", {}).get("content", ""),
                message_type="text",
            )

# gateway/adapters/feishu.py
class FeishuGateway(BaseGateway):
    """飞书网关（支持机器人和应用消息）"""

    def __init__(self, agent_factory, app_id: str, app_secret: str, verification_token: str = None):
        super().__init__(agent_factory)
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self._tenant_access_token = None

    async def _get_tenant_access_token(self) -> str:
        """获取飞书 Tenant Access Token"""
        from httpx import AsyncClient
        async with AsyncClient() as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.app_id,
                    "app_secret": self.app_secret,
                }
            )
            return resp.json()["tenant_access_token"]

    async def start(self):
        """启动飞书消息监听"""
        # 使用 WebSocket 或 Webhook 接收消息
        ...

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送飞书消息"""
        token = await self._get_tenant_access_token()
        from httpx import AsyncClient
        async with AsyncClient() as client:
            await client.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                params={"receive_id_type": "open_id"},
                json={
                    "receive_id": user_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": content}),
                }
            )

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收飞书消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(
                platform="feishu",
                message_id=raw_msg.get("message_id", ""),
                user_id=raw_msg.get("sender", {}).get("sender_id", {}).get("open_id", ""),
                user_name=raw_msg.get("sender", {}).get("sender_id", {}).get("name", ""),
                content=raw_msg.get("content", {}).get("text", ""),
                message_type="text",
            )

# gateway/adapters/wecom.py
class WeComGateway(BaseGateway):
    """企业微信网关（自建应用）"""

    def __init__(self, agent_factory, corp_id: str, corp_secret: str, agent_id: int, token: str, encoding_aes_key: str):
        super().__init__(agent_factory)
        self.corp_id = corp_id
        self.corp_secret = corp_secret
        self.agent_id = agent_id
        self.token = token
        self.encoding_aes_key = encoding_aes_key

    async def start(self):
        """启动企业微信消息监听"""
        # 需要公网域名接收回调
        ...

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送企业微信消息"""
        from httpx import AsyncClient
        # 先获取 access_token
        async with AsyncClient() as client:
            token_resp = await client.get(
                "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
                params={"corpid": self.corp_id, "corpsecret": self.corp_secret}
            )
            access_token = token_resp.json()["access_token"]

            await client.post(
                f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}",
                json={
                    "touser": user_id,
                    "msgtype": "text",
                    "agentid": self.agent_id,
                    "text": {"content": content},
                }
            )

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收企业微信消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(
                platform="wecom",
                message_id=raw_msg.get("MsgId", ""),
                user_id=raw_msg.get("FromUserName", ""),
                user_name=raw_msg.get("FromUserName", ""),
                content=raw_msg.get("Content", ""),
                message_type="text",
            )

# gateway/adapters/qq.py
class QQGateway(BaseGateway):
    """QQ 网关（通过 QQ 频道机器人或官方 API）"""

    def __init__(self, agent_factory, app_id: str, app_secret: str):
        super().__init__(agent_factory)
        self.app_id = app_id
        self.app_secret = app_secret

    async def start(self):
        """启动 QQ 消息监听"""
        # 使用 QQ 频道机器人 API
        ...

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送 QQ 消息"""
        ...

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收 QQ 消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(
                platform="qq",
                message_id=raw_msg.get("id", ""),
                user_id=raw_msg.get("author", {}).get("id", ""),
                user_name=raw_msg.get("author", {}).get("username", ""),
                content=raw_msg.get("content", ""),
                message_type="text",
            )

# gateway/adapters/baidu.py
class BaiduGateway(BaseGateway):
    """百度千帆网关（通过千帆 AppBuilder）"""

    def __init__(self, agent_factory, api_key: str, secret_key: str):
        super().__init__(agent_factory)
        self.api_key = api_key
        self.secret_key = secret_key

    async def start(self):
        """启动百度千帆消息监听"""
        ...

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送百度千帆消息"""
        ...

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收百度千帆消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(
                platform="baidu",
                message_id=raw_msg.get("message_id", ""),
                user_id=raw_msg.get("user_id", ""),
                user_name=raw_msg.get("user_name", ""),
                content=raw_msg.get("content", ""),
                message_type="text",
            )

# gateway/webhook.py
class WebhookGateway(BaseGateway):
    """通用 Webhook 网关（支持任何 HTTP 平台）"""

    def __init__(self, agent_factory, host: str = "0.0.0.0", port: int = 8080):
        super().__init__(agent_factory)
        self.host = host
        self.port = port
        self._app = None

    async def start(self):
        """启动 Webhook 服务器"""
        from fastapi import FastAPI, Request
        import uvicorn

        self._app = FastAPI()

        @self._app.post("/webhook")
        async def handle_webhook(request: Request):
            body = await request.json()
            message = GatewayMessage(
                platform="webhook",
                message_id=body.get("message_id", str(uuid.uuid4())),
                user_id=body.get("user_id", "anonymous"),
                user_name=body.get("user_name", ""),
                content=body.get("content", ""),
                message_type=body.get("type", "text"),
                metadata=body,
            )
            result = await self.handle_message(message)
            return {"content": result}

        config = uvicorn.Config(self._app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()

    async def send_message(self, user_id: str, content: str, **kwargs):
        """Webhook 不主动推送，通过 HTTP 响应返回"""
        pass

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """通过 FastAPI 接收消息"""
        while True:
            raw_msg = await self._message_queue.get()
            yield GatewayMessage(**raw_msg)
```

**网关使用示例**：

```python
# 单平台
from wuwei.gateway import WeChatGateway
gateway = WeChatGateway(agent_factory=lambda: Agent(llm=llm, tools=tools), webhook_url="...")
await gateway.start()

# 多平台同时运行
from wuwei.gateway import WeChatGateway, DingTalkGateway, FeishuGateway

gateways = [
    WeChatGateway(agent_factory, webhook_url="..."),
    DingTalkGateway(agent_factory, app_key="...", app_secret="..."),
    FeishuGateway(agent_factory, app_id="...", app_secret="..."),
]

# 并行启动所有网关
import asyncio
await asyncio.gather(*[gw.start() for gw in gateways])
```

### 4.11 可观测性（observability/）

**借鉴**：AgentScope 的 OpenTelemetry 集成

```python
# observability/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class TracingMiddleware(Middleware):
    """追踪中间件（借鉴 AgentScope）"""

    def __init__(self, service_name: str = "wuwei", endpoint: str = None):
        # 初始化 OpenTelemetry
        provider = TracerProvider()
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """开始 LLM 调用追踪"""
        ctx.metadata["span"] = self.tracer.start_span("llm_call")
        return ctx

    async def after_llm(self, ctx: MiddlewareContext, response: AIMessage) -> MiddlewareContext:
        """结束 LLM 调用追踪"""
        span = ctx.metadata.get("span")
        if span:
            span.set_attribute("llm.tokens_used", response.usage.total_tokens)
            span.end()
        return ctx

    async def before_tool(self, ctx: MiddlewareContext, tool_call: ToolCall) -> ToolCall:
        """开始工具执行追踪"""
        ctx.metadata["tool_span"] = self.tracer.start_span(
            f"tool_{tool_call.function.name}"
        )
        return tool_call

    async def after_tool(self, ctx: MiddlewareContext, tool_message: ToolMessage) -> ToolMessage:
        """结束工具执行追踪"""
        span = ctx.metadata.get("tool_span")
        if span:
            span.set_attribute("tool.status", tool_message.status)
            span.end()
        return tool_message
```

### 4.12 MCP 模块（mcp/）

**借鉴**：Hermes Agent 的 MCP 客户端 + Claude Code 的多作用域配置

```python
# mcp/config.py
from pydantic import BaseModel, Field
from typing import Optional, Literal
import json
import os

class MCPServerConfig(BaseModel):
    """MCP 服务器配置"""
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    # stdio 配置
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # http/sse 配置
    url: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    # 通用配置
    timeout: float = 60.0
    enabled: bool = True

class MCPConfig(BaseModel):
    """MCP 配置管理（借鉴 Claude Code 多作用域）"""
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @classmethod
    def load(cls, scopes: list[str] = None) -> "MCPConfig":
        """从多个作用域加载配置"""
        if scopes is None:
            scopes = ["project", "user"]

        config = cls()
        for scope in scopes:
            path = cls._get_config_path(scope)
            if path and os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for name, server in data.get("mcpServers", {}).items():
                    config.mcp_servers[name] = MCPServerConfig(name=name, **server)
        return config

    @classmethod
    def _get_config_path(cls, scope: str) -> Optional[str]:
        if scope == "project":
            return ".mcp.json"
        elif scope == "user":
            return os.path.expanduser("~/.wuwei/.mcp.json")
        return None

# mcp/client.py
from abc import ABC, abstractmethod
from typing import AsyncIterator
import asyncio

class BaseMCPClient(ABC):
    """MCP 客户端基类"""

    @abstractmethod
    async def connect(self):
        """连接到 MCP 服务器"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict]:
        """列出可用工具"""
        ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具"""
        ...

class StdioMCPClient(BaseMCPClient):
    """Stdio 传输的 MCP 客户端（借鉴 Hermes）"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process = None
        self._reader = None
        self._writer = None

    async def connect(self):
        """启动 MCP 服务器子进程"""
        self.process = await asyncio.create_subprocess_exec(
            self.config.command, *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.config.env},
        )
        self._reader = self.process.stdout
        self._writer = self.process.stdin

    async def disconnect(self):
        """关闭子进程"""
        if self.process:
            self.process.terminate()
            await self.process.wait()

    async def list_tools(self) -> list[dict]:
        """发送 list_tools 请求"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }
        await self._send(request)
        response = await self._receive()
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """发送 call_tool 请求"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        await self._send(request)
        response = await self._receive()
        return response.get("result", {})

    async def _send(self, data: dict):
        """发送 JSON-RPC 消息"""
        message = json.dumps(data)
        self._writer.write(f"{message}\n".encode())
        await self._writer.drain()

    async def _receive(self) -> dict:
        """接收 JSON-RPC 消息"""
        line = await asyncio.wait_for(
            self._reader.readline(),
            timeout=self.config.timeout,
        )
        return json.loads(line.decode())

class HTTPMCPClient(BaseMCPClient):
    """HTTP/SSE 传输的 MCP 客户端"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session_id = None

    async def connect(self):
        """建立 HTTP 连接"""
        from httpx import AsyncClient
        self._client = AsyncClient(
            base_url=self.config.url,
            headers=self.config.headers,
            timeout=self.config.timeout,
        )

    async def disconnect(self):
        """关闭连接"""
        if hasattr(self, '_client'):
            await self._client.aclose()

    async def list_tools(self) -> list[dict]:
        """列出工具"""
        resp = await self._client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        })
        return resp.json().get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具"""
        resp = await self._client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        return resp.json().get("result", {})

# mcp/tools.py
class MCPToolAdapter:
    """MCP 工具适配器（将 MCP 工具转为 wuwei Tool）"""

    def __init__(self, client: BaseMCPClient, server_name: str):
        self.client = client
        self.server_name = server_name

    async def discover_tools(self) -> list[Tool]:
        """发现 MCP 服务器上的工具"""
        raw_tools = await self.client.list_tools()
        tools = []
        for raw in raw_tools:
            tool = Tool(
                name=f"mcp__{self.server_name}__{raw['name']}",
                description=raw.get("description", ""),
                parameters=raw.get("inputSchema", {}),
                handler=self._create_handler(raw["name"]),
                side_effect=not raw.get("annotations", {}).get("readOnlyHint", False),
            )
            tools.append(tool)
        return tools

    def _create_handler(self, tool_name: str):
        """创建工具处理函数"""
        async def handler(**kwargs):
            result = await self.client.call_tool(tool_name, kwargs)
            # 提取文本内容
            content = result.get("content", [])
            texts = [c["text"] for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else str(result)
        return handler

# mcp/session.py
class MCPSessionManager:
    """MCP 会话管理器（借鉴 DeepAgents MCPSessionManager）"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self._clients: dict[str, BaseMCPClient] = {}
        self._tools: dict[str, list[Tool]] = {}

    async def connect_all(self):
        """连接所有启用的 MCP 服务器"""
        for name, server_config in self.config.mcp_servers.items():
            if not server_config.enabled:
                continue
            if server_config.transport == "stdio":
                client = StdioMCPClient(server_config)
            else:
                client = HTTPMCPClient(server_config)
            await client.connect()
            self._clients[name] = client

            # 发现工具
            adapter = MCPToolAdapter(client, name)
            self._tools[name] = await adapter.discover_tools()

    async def disconnect_all(self):
        """断开所有连接"""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
        self._tools.clear()

    def get_all_tools(self) -> list[Tool]:
        """获取所有 MCP 工具"""
        tools = []
        for server_tools in self._tools.values():
            tools.extend(server_tools)
        return tools
```

**MCP 配置文件格式**（`.mcp.json`）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "transport": "stdio"
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "..." },
      "transport": "stdio"
    },
    "remote-server": {
      "url": "https://mcp.example.com/mcp",
      "transport": "http",
      "headers": { "Authorization": "Bearer xxx" }
    }
  }
}
```

### 4.13 技能系统增强（skill/）

**借鉴**：AgentScope 的 SkillViewer + Claude Code 的多源加载 + Hermes 的结构化元数据

```python
# skill/skill.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Skill(BaseModel):
    """技能数据模型（借鉴 AgentScope + Claude Code）"""
    name: str = Field(..., max_length=64, pattern=r"^[a-z0-9-]+$")
    description: str = Field(..., max_length=1024)
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    # Claude Code 风格的高级配置
    when_to_use: str = ""          # 何时使用此技能
    allowed_tools: list[str] = Field(default_factory=list)  # 允许使用的工具
    required_tools: list[str] = Field(default_factory=list) # 必需的工具
    model: Optional[str] = None    # 指定模型
    tags: list[str] = Field(default_factory=list)
    # 加载来源
    source: str = ""               # 技能来源路径
    # 内容
    markdown: str = ""             # SKILL.md 正文
    # 辅助文件
    scripts: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

# skill/loader.py
import os
from pathlib import Path
from typing import AsyncIterator

class SkillLoader:
    """多源技能加载器（借鉴 Claude Code 多源加载）"""

    def __init__(self, sources: list[str] = None):
        """
        Args:
            sources: 技能来源目录列表，按优先级排序
        """
        self.sources = sources or []
        self._cache: dict[str, Skill] = {}

    def add_source(self, path: str):
        """添加技能来源"""
        self.sources.append(path)

    async def load_all(self) -> list[Skill]:
        """从所有来源加载技能"""
        skills = {}
        for source in self.sources:
            async for skill in self._load_from_source(source):
                # 后加载的覆盖先加载的（last-wins）
                skills[skill.name] = skill
        return list(skills.values())

    async def _load_from_source(self, source: str) -> AsyncIterator[Skill]:
        """从单个来源加载技能"""
        source_path = Path(source)
        if not source_path.exists():
            return

        for skill_dir in source_path.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skill = self._parse_skill(skill_md, skill_dir)
                if skill:
                    yield skill

    def _parse_skill(self, skill_md: Path, skill_dir: Path) -> Optional[Skill]:
        """解析 SKILL.md 文件"""
        import yaml

        content = skill_md.read_text(encoding="utf-8")
        # 解析 YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                markdown = parts[2].strip()
            else:
                return None
        else:
            return None

        # 收集辅助文件
        scripts = []
        references = []
        scripts_dir = skill_dir / "scripts"
        refs_dir = skill_dir / "references"
        if scripts_dir.exists():
            scripts = [str(f) for f in scripts_dir.glob("*")]
        if refs_dir.exists():
            references = [str(f) for f in refs_dir.glob("*")]

        return Skill(
            name=frontmatter.get("name", skill_dir.name),
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "1.0.0"),
            author=frontmatter.get("author", ""),
            license=frontmatter.get("license", "MIT"),
            when_to_use=frontmatter.get("when_to_use", ""),
            allowed_tools=frontmatter.get("allowed_tools", []),
            required_tools=frontmatter.get("required_tools", []),
            model=frontmatter.get("model"),
            tags=frontmatter.get("tags", []),
            source=str(skill_dir),
            markdown=markdown,
            scripts=scripts,
            references=references,
        )

# skill/viewer.py
from wuwei.tools.base import Tool

class SkillViewerTool(Tool):
    """技能查看器工具（借鉴 AgentScope SkillViewer）"""

    def __init__(self, skill_registry: "SkillRegistry"):
        super().__init__(
            name="view_skill",
            description="查看指定技能的详细内容和使用说明",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要查看的技能名称",
                    }
                },
                "required": ["skill_name"],
            },
            handler=self._view_skill,
        )
        self.registry = skill_registry

    async def _view_skill(self, skill_name: str) -> str:
        """查看技能内容"""
        skill = self.registry.get(skill_name)
        if not skill:
            available = ", ".join(self.registry.list_names())
            return f"技能 '{skill_name}' 不存在。可用技能：{available}"

        # 构建技能信息
        info_parts = [
            f"# {skill.name}",
            f"描述：{skill.description}",
            f"版本：{skill.version}",
            "",
            "## 使用说明",
            skill.markdown,
        ]

        if skill.allowed_tools:
            info_parts.extend(["", f"允许工具：{', '.join(skill.allowed_tools)}"])
        if skill.required_tools:
            info_parts.extend(["", f"必需工具：{', '.join(skill.required_tools)}"])
        if skill.tags:
            info_parts.extend(["", f"标签：{', '.join(skill.tags)}"])

        return "\n".join(info_parts)

# skill/registry.py
class SkillRegistry:
    """技能注册表"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """注册技能"""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self._skills.get(name)

    def list_names(self) -> list[str]:
        """列出所有技能名称"""
        return list(self._skills.keys())

    def list_by_tag(self, tag: str) -> list[Skill]:
        """按标签过滤技能"""
        return [s for s in self._skills.values() if tag in s.tags]

    def get_tools_for_skill(self, skill: Skill) -> list[str]:
        """获取技能允许的工具列表"""
        return skill.allowed_tools if skill.allowed_tools else ["*"]
```

**技能 SKILL.md 格式**（借鉴 Claude Code + AgentScope）：

```markdown
---
name: code-review
description: "Structured code review with checklist and best practices"
version: 1.0.0
author: Wuwei Team
license: MIT
when_to_use: "When the user asks to review code or check code quality"
allowed_tools:
  - read_file
  - list_files
  - grep
required_tools:
  - read_file
tags: [code-quality, review, best-practices]
---

# Code Review Skill

## 检查清单

1. **代码风格**
   - 变量命名是否清晰
   - 函数长度是否合理（< 50 行）
   - 是否有重复代码

2. **错误处理**
   - 是否有未处理的异常
   - 错误信息是否清晰
   - 资源是否正确释放

3. **安全性**
   - 是否有硬编码的密钥
   - 输入是否经过验证
   - 是否有 SQL 注入风险

## 执行步骤

1. 使用 `list_files` 获取项目结构
2. 使用 `read_file` 读取关键文件
3. 使用 `grep` 搜索潜在问题
4. 输出审查报告
```

**技能系统使用示例**：

```python
from wuwei.skill import SkillLoader, SkillRegistry, SkillViewerTool

# 加载技能
loader = SkillLoader(skills_dir)
skills = await loader.load_all()

# 注册到注册表
registry = SkillRegistry()
for skill in skills:
    registry.register(skill)

# 创建 SkillViewer 工具
viewer_tool = SkillViewerTool(registry)

# Agent 可以通过 view_skill 工具查看技能内容
agent = Agent(llm=llm, tools=[viewer_tool, ...])
```

---

## 五、实施计划

### 5.1 分阶段实施（破坏性重构）

```
Phase 1: 核心抽象层（core/）— 2 周
├── 实现 Runnable 接口
├── 实现统一消息体系（BaseMessage/AIMessage/ToolMessage）
├── 实现增强版 Tool（Pydantic 自动生成 Schema）
└── 删除旧版 API

Phase 2: LLM 网关（llm/）— 1 周
├── 重构 LLMGateway 为多提供商架构
├── 实现 Anthropic 适配器
├── 实现国内适配器（智谱/DashScope/文心）
└── 实现 Ollama 适配器

Phase 3: 状态图编排（graph/）— 2 周
├── 实现 StateGraph
├── 实现检查点系统（内存/SQLite/Postgres）
├── 实现预构建图（ReAct、PlanExecute）
└── 替换旧版 PlanAgent

Phase 4: 中间件系统（middleware/）— 2 周
├── 实现 Middleware 基类
├── 实现预构建中间件（文件系统/记忆/上下文压缩/子代理）
├── 实现 HITL 中间件
└── 删除旧版 Hook 系统

Phase 5: 多平台网关（gateway/）— 2 周
├── 实现网关基类
├── 实现国内平台（微信/钉钉/飞书/企业微信/QQ/百度）
├── 实现海外平台（Telegram/Discord/Slack）
└── 实现通用 Webhook 网关

Phase 6: MCP 模块（mcp/）— 2 周
├── 实现 MCP 客户端（stdio/HTTP/SSE）
├── 实现 MCP 配置管理（.mcp.json 多作用域）
├── 实现 MCP 工具适配器
└── 实现会话管理（有状态/无状态）

Phase 7: 技能系统增强（skill/）— 1 周
├── 增强 Skill 数据模型
├── 实现多源加载器（内置/用户/项目/MCP）
├── 实现 SkillViewer 工具
└── 实现技能注册表

Phase 8: 生产化（sandbox/ + observability/ + multi_agent/）— 2 周
├── 实现 Docker/E2B 沙箱
├── 实现 OpenTelemetry 追踪
├── 实现 Swarm 多 Agent 协作
└── 完善内置工具（12 组）
```

### 5.2 依赖管理（Core + Extensions）

```toml
# wuwei-core/pyproject.toml — 核心包
[project]
name = "wuwei"
version = "2.0.0"
description = "无为而治的 AI 智能体框架"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

# wuwei-ext-graph/pyproject.toml — 图编排扩展
[project]
name = "wuwei-ext-graph"
dependencies = ["wuwei>=2.0.0"]

# wuwei-ext-middleware/pyproject.toml — 中间件扩展
[project]
name = "wuwei-ext-middleware"
dependencies = ["wuwei>=2.0.0"]

# wuwei-ext-mcp/pyproject.toml — MCP 扩展
[project]
name = "wuwei-ext-mcp"
dependencies = [
    "wuwei>=2.0.0",
    "mcp>=1.0",
]

# wuwei-ext-skill/pyproject.toml — 技能系统扩展
[project]
name = "wuwei-ext-skill"
dependencies = [
    "wuwei>=2.0.0",
    "python-frontmatter>=1.0",
]

# wuwei-ext-gateway/pyproject.toml — 多平台网关扩展
[project]
name = "wuwei-ext-gateway"
dependencies = [
    "wuwei>=2.0.0",
    "httpx>=0.25",
]

# wuwei-ext-memory/pyproject.toml — 记忆系统扩展
[project]
name = "wuwei-ext-memory"
dependencies = ["wuwei>=2.0.0"]

# wuwei-ext-observability/pyproject.toml — 可观测性扩展
[project]
name = "wuwei-ext-observability"
dependencies = [
    "wuwei>=2.0.0",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
]

# wuwei-ext-sandbox/pyproject.toml — 沙箱执行扩展
[project]
name = "wuwei-ext-sandbox"
dependencies = ["wuwei>=2.0.0"]

# wuwei-all/pyproject.toml — 完整安装
[project]
name = "wuwei-all"
dependencies = [
    "wuwei>=2.0.0",
    "wuwei-ext-graph",
    "wuwei-ext-middleware",
    "wuwei-ext-mcp",
    "wuwei-ext-skill",
    "wuwei-ext-gateway",
    "wuwei-ext-memory",
    "wuwei-ext-observability",
    "wuwei-ext-sandbox",
]
```

**安装方式**：

```bash
# 最小安装（核心）
pip install wuwei                          # ~50KB

# 按需安装
pip install wuwei[graph]                   # + 状态图
pip install wuwei[middleware]              # + 中间件
pip install wuwei[mcp]                     # + MCP 支持
pip install wuwei[skill]                   # + 技能系统
pip install wuwei[gateway]                 # + 多平台网关
pip install wuwei[memory]                  # + 记忆系统
pip install wuwei[observability]           # + OpenTelemetry
pip install wuwei[sandbox]                 # + 沙箱执行

# 完整安装
pip install wuwei[all]                     # 全部扩展
```

---

## 六、对比总结

### 6.1 与 8 个参考框架对比

| 维度 | 原 Wuwei | Wuwei 2.0 | 借鉴来源 |
|------|---------|-----------|----------|
| **架构模式** | 单体 | Core + Extensions | rig Feature-flag |
| **Schema** | 手动 JSON Schema | Pydantic 自动生成 + 注入参数 | LangChain |
| **状态管理** | 线性循环 | 状态图 + 检查点 + 断点恢复 | LangGraph |
| **扩展性** | 固定 Hook | 可组合中间件栈 | Deep Agents + agent-framework |
| **多模型** | 仅 OpenAI | OpenAI + Anthropic + 智谱 + DashScope + 文心 + Ollama | Hermes |
| **平台支持** | 无 | 微信/钉钉/飞书/企业微信/QQ/百度 + Telegram/Discord/Slack | Hermes |
| **MCP 支持** | 无 | stdio/HTTP/SSE + 多作用域配置 + OAuth | Hermes + Claude Code |
| **技能系统** | 简单 SKILL.md | 多源加载 + SkillViewer + 结构化元数据 | AgentScope + Claude Code |
| **声明式配置** | 无 | YAML 定义 Agent | agent-framework |
| **执行安全** | 无隔离 | Docker/E2B 沙箱 | AgentScope |
| **多 Agent** | 仅 PlanAgent | Swarm 团队协作 | Claude Code |
| **可观测性** | Console 日志 | OpenTelemetry 追踪 | AgentScope + rig |
| **Python 版本** | >=3.10 | >=3.11 | 现代化 |
| **核心包大小** | ~100KB | ~50KB | rig 轻量原则 |

### 6.2 各框架优缺点借鉴总结

| 框架 | 借鉴的优点 | 摒弃的缺点 |
|------|-----------|-----------|
| **LangChain** | Pydantic Schema 自动生成、Runnable 接口 | 过重的抽象层、复杂依赖链 |
| **LangGraph** | 状态图编排、检查点恢复、条件边 | 简单场景过度设计 |
| **Deep Agents** | 可组合中间件栈、文件系统/记忆中间件 | 过度 opinionated |
| **Hermes Agent** | 多平台网关、300+ 模型支持 | 代码库巨大、架构复杂 |
| **AgentScope** | Docker/E2B 沙箱、OpenTelemetry | 与阿里云生态耦合 |
| **Claude Code** | Swarm 多 Agent、细粒度权限、MCP 多作用域 | TypeScript 实现、与 Claude 强绑定 |
| **agent-framework** | 声明式 YAML 配置、DevUI、Graph 工作流 | .NET 生态绑定 |
| **rig** | Feature-flag 门控、Companion crate、Builder 模式 | Rust 生态、Python 生态不适用 |

---

## 七、关键设计决策

### 7.1 关于"借鉴"与"原创"

**这不是抄袭，而是工程实践中的最佳做法**：

- **框架设计是公共知识** — 状态图、中间件、MCP 都是行业标准模式，不是某个框架的专利
- **实现方式完全不同** — Wuwei 用自己的代码实现，只是借鉴了设计思想
- **组合创新 ≠ 抄窃** — 把多个框架的优点融合成一个新的轻量级框架，这是工程实践中的常见做法
- **就像 React 借鉴了 Virtual DOM 概念**，但 React 本身不是抄袭
- **关键是在文档中明确标注"借鉴来源"**，而不是声称是自己的原创设计

### 7.2 核心设计原则

1. **Core + Extensions** — 核心极简（~50KB），一切功能皆可插件（借鉴 rig Feature-flag）
2. **破坏性重构** — 不兼容旧版 API，彻底摒弃不良设计（Hook → Middleware 栈）
3. **国内优先** — LLM 适配器和平台网关优先支持国内生态
4. **保持轻量** — 核心依赖只有 openai + pydantic + pyyaml，其他都是可选
5. **渐进式采用** — 状态图、中间件、沙箱都是可选的，简单场景不需要
6. **零依赖优先** — 默认实现不依赖外部服务（内存存储、简单嵌入）
7. **协议优于实现** — 使用 Protocol 定义接口，具体实现可替换

### 7.3 Hook → Middleware 演进说明

| 维度 | 旧版 Hook | 新版 Middleware |
|------|----------|----------------|
| **设计模式** | 固定回调函数 | 可组合中间件栈 |
| **扩展性** | 需要修改 Runner 代码 | 动态插拔，无需修改核心 |
| **组合性** | Hook 之间无法通信 | 中间件可以传递和修改上下文 |
| **错误处理** | 全局异常捕获 | 每个中间件独立错误处理 |
| **测试性** | 需要 Mock Runner | 中间件可独立测试 |
| **借鉴来源** | — | Deep Agents 中间件栈设计 |

---

*本方案基于对 6 个参考框架的深入分析，旨在让 Wuwei 2.0 成为一个既有 LangChain 的标准化、又有 LangGraph 的编排能力、又有 Deep Agents 的扩展性、又有 Hermes 的多模型支持、又有 AgentScope 的生产就绪性、又有 Claude Code 的协作能力的轻量级 AI 代理框架。重点支持国内 LLM 生态和平台生态。*
