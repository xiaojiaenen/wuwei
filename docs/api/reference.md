# API 参考

## 模块索引

| 模块 | 说明 |
|------|------|
| `wuwei.agent` | Agent 与 PlanAgent 门面、会话管理 |
| `wuwei.llm` | LLM 网关、消息类型、响应模型 |
| `wuwei.memory` | Context、存储、压缩、窗口管理 |
| `wuwei.planning` | Planner、Task、PlanRunResult |
| `wuwei.runtime` | Runner、Hook 系统、HITL |
| `wuwei.skill` | 技能管理、Provider 协议 |
| `wuwei.tools` | 工具模型、注册表、执行器 |

## 关键类一览

### Agent 层

| 类 | 模块 | 说明 |
|----|------|------|
| `Agent` | `wuwei.agent` | 普通单 Agent 门面 |
| `PlanAgent` | `wuwei.agent` | Plan-and-Execute Agent 门面 |
| `BaseAgent` | `wuwei.agent` | 所有 Agent 的抽象基类 |
| `BaseSessionAgent` | `wuwei.agent` | 带会话能力的公共基类 |
| `AgentSession` | `wuwei.agent` | 会话配置 + 上下文 |

### LLM 层

| 类 | 模块 | 说明 |
|----|------|------|
| `LLMGateway` | `wuwei.llm` | 统一 LLM 调用网关 |
| `LLMResponse` | `wuwei.llm` | 非流式响应 |
| `LLMResponseChunk` | `wuwei.llm` | 流式响应块 |
| `Message` | `wuwei.llm` | 消息模型（role, content, tool_calls） |
| `ToolCall` | `wuwei.llm` | 工具调用 |
| `FunctionCall` | `wuwei.llm` | 函数调用详情 |
| `AgentEvent` | `wuwei.llm` | 结构化事件流事件 |
| `AgentRunResult` | `wuwei.llm` | 运行结果（content + usage） |

### Memory 层

| 类 | 模块 | 说明 |
|----|------|------|
| `Context` | `wuwei.memory` | 消息上下文管理 |
| `Storage` | `wuwei.memory` | 持久化协议 |
| `FileStorage` | `wuwei.memory` | 文件存储实现 |
| `SimpleContextWindow` | `wuwei.memory` | 上下文窗口构建器 |
| `ContextCompressor` | `wuwei.memory` | 压缩器协议 |
| `LLMContextCompressor` | `wuwei.memory` | LLM 压缩器实现 |

### Planning 层

| 类 | 模块 | 说明 |
|----|------|------|
| `Planner` | `wuwei.planning` | 任务规划器 |
| `Task` | `wuwei.planning` | 任务节点 |
| `TaskList` | `wuwei.planning` | 任务列表包装 |
| `PlanRunResult` | `wuwei.planning` | 运行结果汇总 |

### Runtime 层

| 类 | 模块 | 说明 |
|----|------|------|
| `AgentRunner` | `wuwei.runtime` | 普通 Agent 执行器 |
| `PlannerExecutorRunner` | `wuwei.runtime` | Plan-Execute 执行器 |
| `RuntimeHook` | `wuwei.runtime` | Hook 基类 |
| `HookManager` | `wuwei.runtime` | Hook 管理器 |
| `ConsoleHook` | `wuwei.runtime` | 控制台日志 Hook |
| `ContextCompressionHook` | `wuwei.runtime` | 上下文压缩 Hook |
| `StorageHook` | `wuwei.runtime` | 持久化 Hook |
| `SkillHook` | `wuwei.runtime` | 技能注入 Hook |
| `HitlHook` | `wuwei.runtime` | HITL 审批 Hook |

### HITL 层

| 类 | 模块 | 说明 |
|----|------|------|
| `ApprovalProvider` | `wuwei.runtime.hitl` | 审批提供者协议 |
| `ApprovalPolicy` | `wuwei.runtime.hitl` | 审批策略 |
| `ApprovalRequest` | `wuwei.runtime.hitl` | 审批请求 |
| `ApprovalDecision` | `wuwei.runtime.hitl` | 审批决定 |
| `ConsoleApprovalProvider` | `wuwei.runtime.hitl` | 控制台审批实现 |
| `ToolApprovalRejected` | `wuwei.runtime.hitl` | 审批拒绝异常 |

### Skill 层

| 类 | 模块 | 说明 |
|----|------|------|
| `Skill` | `wuwei.skill` | 技能数据模型 |
| `SkillProvider` | `wuwei.skill` | 技能提供者协议 |
| `SkillManager` | `wuwei.skill` | 技能管理器 |
| `FileSystemSkillProvider` | `wuwei.skill.fs_provider` | 文件系统提供者 |

### Tools 层

| 类 | 模块 | 说明 |
|----|------|------|
| `Tool` | `wuwei.tools` | 工具模型 |
| `ToolParameters` | `wuwei.tools` | 工具参数 Schema |
| `ToolRegistry` | `wuwei.tools` | 工具注册表 |
| `ToolExecutor` | `wuwei.tools` | 工具执行器 |
