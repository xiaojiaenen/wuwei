# Wuwei

Wuwei 是一个轻量、可扩展的 Python Agent 框架，目标是把模型调用、会话管理、工具注册与执行、plan-and-execute 拆成边界清晰的模块，方便学习和继续扩展。

## 当前目录结构

```text
wuwei/
├─ examples/          # 可直接运行的示例
├─ tests/             # pytest 测试
├─ wuwei/             # 框架源码
│  ├─ agent/          # Agent、PlanAgent、Session、基础抽象
│  ├─ runtime/        # AgentRunner、PlannerExecutorRunner
│  ├─ planning/       # Planner、Task
│  ├─ memory/         # Context
│  ├─ llm/            # Gateway、Types、Adapters
│  └─ tools/          # Tool、Registry、Executor
├─ pyproject.toml
└─ README.md
```

这次目录调整的核心是：

- `tests/` 只放测试
- `examples/` 只放示例
- `agent/` 只放门面对象和会话
- `runtime/` 单独承接执行器
- `planning/` 单独承接规划相关模型
- `memory/` 单独承接上下文

## 安装

要求：

- Python `>=3.10`

使用 `pip`：

```bash
pip install -e .
```

安装开发依赖：

```bash
pip install -e ".[dev]"
```

如果你使用 `uv`：

```bash
uv sync
```

## 核心模块

- `wuwei.agent`
  - `Agent`：普通单 agent 门面
  - `PlanAgent`：plan-and-execute 门面
  - `AgentSession`：会话对象
  - `BaseAgent / BaseSessionAgent`：基础抽象

- `wuwei.runtime`
  - `AgentRunner`：普通 agent 执行器
  - `PlannerExecutorRunner`：plan-and-execute 执行器

- `wuwei.planning`
  - `Planner`：任务规划器
  - `Task / TaskList`：任务模型

- `wuwei.memory`
  - `Context`：消息上下文

- `wuwei.llm`
  - `LLMGateway`：统一模型调用入口
  - `Message / ToolCall / LLMResponse / LLMResponseChunk`：统一类型定义

- `wuwei.tools`
  - `ToolRegistry`：工具注册
  - `ToolExecutor`：工具执行
  - `Tool / ToolParameters`：工具 schema

## 快速开始

最简单的方式是直接运行 `examples/` 里的示例。

离线示例：

```bash
python examples/tool_executor_minimal.py
```

在线示例：

```powershell
$env:WUWEI_API_KEY="your_key"
python examples/agent_minimal.py
python examples/agent_session_minimal.py
python examples/plan_agent_minimal.py
```

更详细的说明可以看：

- [examples/README.md](examples/README.md)
- [docs/agent_framework_core_features.md](docs/agent_framework_core_features.md)：上下文压缩、滑动窗口、MySQL 历史、长期记忆、HITL 等核心能力实现指南

## 示例目录

- [tool_executor_minimal.py](examples/tool_executor_minimal.py)
  - 完全离线可运行，适合先理解工具系统

- [agent_minimal.py](examples/agent_minimal.py)
  - 最小 `Agent` 示例，演示流式输出和工具调用

- [agent_session_minimal.py](examples/agent_session_minimal.py)
  - 演示 session 多轮复用

- [plan_agent_minimal.py](examples/plan_agent_minimal.py)
  - 演示 `PlanAgent` 的“先规划、再执行”

## 当前建议的扩展方向

- 先完善 `PlanAgent / PlannerExecutorRunner`
  - 比如 task 并行执行、失败恢复、重规划

- 再增强 `ToolExecutor`
  - 比如更丰富的执行事件、超时控制、重试策略

- 最后再考虑可观测和 Web 可视化
  - 这样不会过早把主链路搞复杂
