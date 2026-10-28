# Wuwei 使用指南

## 选择模型和资源所有权

```python
from wuwei import LLMGateway

llm = LLMGateway.from_env(provider="openai")
# provider="responses" 使用 Responses API；provider="anthropic" 使用 Claude。
# 兼容端点：from_env(provider="openai", env_prefix="DEEPSEEK")
```

环境变量使用 `<PREFIX>_API_KEY`、`<PREFIX>_BASE_URL`、`<PREFIX>_MODEL`。显式配置可传 `LLMGateway({...})`，不要把认证信息写进源文件。

需要自定义认证、代理或连接池时，优先传 `client` 或 `client_factory`，而不是修改消息转换器：

```python
from openai import AsyncOpenAI
from wuwei import LLMGateway

client = AsyncOpenAI(max_retries=0)  # 从环境变量读取认证
llm = LLMGateway({"provider": "openai", "client": client,
                  "model": "gpt-4o", "temperature": 0.2,
                  "retry": {"max_attempts": 2}, "timeout": 30})
# 使用完 await llm.aclose()；借入 client 默认由调用方 await client.close()。
```

框架自建 SDK client 默认关闭 SDK 重试，让网关负责分类重试。借入客户端也应明确 retry 所有权；否则 SDK 与框架尝试次数可能相乘。`set_fallback(other_gateway)` 借用完整 fallback 策略，调用方管理 other_gateway 生命周期。已发布流式内容、工具或 usage 元数据后不重放请求。确定性参数错误不做传输重试。

## 注册有参数契约的工具

```python
from wuwei import ToolRegistry

registry = ToolRegistry()

@registry.tool(side_effect=False)
def join_names(names: list[str], separator: str = ", ") -> str:
    """连接姓名列表。"""
    return separator.join(names)

assert registry.get("join_names").parameters.properties["names"]["type"] == "array"
```

传 `Agent(llm=llm, tools=registry)`。`registry.register_callable(func)` 适合不能使用装饰器的函数；复杂远程 schema 可显式构造 `Tool` 和 `ToolParameters`。参数先验证再执行，同步 handler 在线程中执行。

写入类工具声明 `side_effect=True`，按需设置 `requires_approval=True`、`timeout_seconds`。超时不保证线程或远端动作停止；unknown 结果必须先对账。框架不会盲目重试已声明的副作用工具。

`load_builtins="minimal"` 为默认策略。可用 `load_builtins=["calc", "file"]` 按名选择，或 False 完全禁用内置插件。增加插件不等于授予业务权限。

## 让普通运行和流式共享会话

```python
from wuwei import Agent

agent = Agent(llm=llm, tools=registry, load_builtins=False)
session = agent.create_session()
# 在 async 函数中：
# first = await agent.run("记住项目名 Wuwei", session=session)
# async for event in agent.stream_events("项目叫什么？", session=session):
#     print(event.type, event.data)
```

两入口不传 session 时也复用 Agent 默认会话。HTTP AG-UI 的 threadId 映射到 session_id；应用仍负责验证线程归属和访问身份。完整消费事件流到 terminal，提前退出应关闭生成器；text_delta 不是执行完成证明。

`AgentRunResult` 包含 content、status、usage、llm_calls、latency_ms、interrupt、termination_reason、output。下游只将 completed 当完成；interrupted 保留审批信息，incomplete 表示未完成，unknown 表示结果不确定。

## 结构化输出仍经过同一运行链路

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]

# 在 async 函数中：
# output, result = await agent.run_structured("总结当前会话", Summary, session=session)
# if result.status == "completed":
#     print(output.title)
# elif result.status == "interrupted":
#     assert output is None
```

返回 `(Summary | None, AgentRunResult)`；校验纠正调用计入同一 run 预算。JSON 提示与验证不等于每个 provider 都支持原生严格 schema。

## 审批、权限和恢复

```python
from wuwei import Agent, HitlMiddleware, MiddlewareStack
from wuwei.graph import SQLiteCheckpointer

stack = MiddlewareStack()
stack.add(HitlMiddleware(mode="interrupt", approval_policy="declared"))
checkpointer = SQLiteCheckpointer("checkpoints.sqlite")
agent = Agent(llm=llm, tools=registry, middleware=stack,
              checkpointer=checkpointer, load_builtins=False)
# result = await agent.run("执行任务")
# if result.status == "interrupted":
#     result = await agent.resume(result.interrupt, approved=True)
```

审批决定绑定具体挂起动作；恢复仍通过其他治理。SQLite 后端支持原子 claim；自定义持久化后端必须实现可靠原子领取，不能用普通 load/save 冒充防重。应用要验证审批人、会话归属，保护检查点和恢复凭证。检查点不能解决外部成功、本地结果未落盘的跨系统窗口；unknown 状态需目的端幂等键或权威对账。

`PermissionMiddleware(PermissionConfig(...))` 控制能力授权、workspace 内目录/网络边界、配额与不可信内容后的写入策略；`BudgetMiddleware` 控制累计 token/估算成本/步数。用 `stack.add(...)` 添加中间件，先明确安全策略再启用副作用。工具权限、人工审批、内容 guardrail 与 Docker/E2B 隔离分别配置。

## 计划与多 Agent

`Planner` 生成 `Task` DAG，`PlanExecutor.execute(plan)` 验证依赖并调度。`PlanExecutor.resume(plan, task_id, approved=True)` 续跑审批任务及新就绪后继。保留返回的任务状态、interrupt、session_id 和 usage；挂起不是失败重试。

同步 SubAgent 与后台 AsyncSubAgent 使用共享 Runner 与继承治理。后台工具包括 start/check/cancel/resume；`resume_async_task` 可续跑挂起的子任务。句柄与调度任务保存在当前进程，不能宣称重启后仍有同一个后台句柄；子运行检查点可通过持久后端保存。

`MultiAgentGraph` 成员挂起或未完成时抛出 `DelegationStoppedError`，异常 `.result` 保留成员 status、usage 与 interrupt。应用取得成员恢复结果后再决定继续协作，不能把挂起成员当成功文本交给后继。

## 图和分叉调试

`StateGraph` 节点返回 State 或字段更新 dict；`writes={"field": reducer}` 指定归并。未知字段写入 state.metadata，invoke/stream 使用相同 reducer。完整示例见 [离线示例](docs/examples/offline.md)。

`await agent.fork(checkpoint_id)` 创建隔离会话与新 run，不修改原分支，默认模拟副作用。明确确认需要真实动作后才传 `allow_side_effects=True`。它是反事实调试，不是跨系统 exactly-once 重放。

## 上下文、存储和检索

- `Context` 保存会话消息；窗口与压缩中间件减少模型上下文，不是原文存储。
- `StorageMiddleware` / `FileStorage` 以 session 归档；业务应定义保存期限和访问策略。
- `ContextLedger(persist_dir=...)` + `LedgerMiddleware` 外置大块结果；注册 `ledger_tools(ledger)` 供 read_context/pin_context/list_context 召回。持久原文与缓存分离；不持久模式为保留引用仍保留原文，总内存不保证有界。
- `MemoryMiddleware` / `CuratedMemoryStore` 用于记忆策展；`RagMiddleware` / KnowledgeStore 用于检索。vectorstore 提供内存及 Chroma、Qdrant、pgvector、Milvus 可选后端。

## 插件、技能和代码执行

Plugin 的 setup(ctx) 通过 ctx.tool_registry 与 ctx.middleware_stack 注册能力，由 PluginManager 管理生命周期。Agent 支持 plugins、plugin_dir。FileSystemSkillProvider 读取 SKILL.md，SkillManager 提供按需指令加载和 allowed-tools 约束。

Code Mode 用 `enable_code_mode(agent)` 注册 run_code；嵌套工具经当前运行治理。直接 `run_code(code, executor, governed_execute=callback)` 必须显式提供可信治理入口。它运行进程内 Python，拥有宿主权限，不是沙箱，不能安全执行任意不可信代码或强制终止死循环。隔离需求使用独立 Docker/E2B 执行路径，并配置资源/网络限制。

## MCP 与 HTTP 协议

MCP client 支持 stdio、Streamable HTTP 和独立旧 HTTP+SSE，transport 分别为 stdio/http/sse。`MCPSessionManager.connect_all()` 隔离服务器失败；使用完 `disconnect_all()`。动态发现只缓存发现结果，每次模型调用注入概览，tool_search 按需注册工具。远端 readOnlyHint 不是本地授权依据。

`build_a2a_app(agent, name=..., description=..., url=...)` 返回 ASGI 应用。声明 **A2A 0.3.0 JSON-RPC**，向卡片 URL 发标准 `message/send`。`tasks/send` 仅为弃用兼容入口，不声明 1.0：

```json
{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"kind":"message","messageId":"m1","role":"user","parts":[{"kind":"text","text":"你好"}]}}}
```

interrupted 映射 input-required，恢复数据在任务 metadata.interrupt 中。没有实现的协议方法返回错误，不假装支持完整 A2A 方法集。

`build_ag_ui_app(agent)` 的 POST /run 接收 threadId/messages，输出 SSE 或 NDJSON；`build_mcp_server`/`serve_mcp` 提供 MCP Server 出口。应用服务器部署、认证、TLS、限流由宿主负责。

## 事件、评估与集成

EventBus 订阅运行事件，ActionLog 记录动作；OpenTelemetry 需 observability extra。`wuwei-evals` 运行数据集与轨迹/输出/rubric 断言；LLM judge 需要模型调用预算。`wuwei-debug` 浏览检查点。多平台 gateway 包含飞书、钉钉、Telegram、微信与 Webhook 适配器；DecisionClient 提供决策服务入口。每种远程服务都应做部署环境独立验收。
