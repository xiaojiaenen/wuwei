# Wuwei

一个面向工具调用、多轮会话与可恢复审批的 Python Agent 框架。

Wuwei 将普通运行、流式事件、结构化输出和委派接入同一执行生命周期。工具、权限、预算、审批和检查点可组合；应用仍需为真实副作用提供幂等或对账机制。

## 安装

需要 Python 3.11 或更高版本。

```bash
pip install wuwei
pip install 'wuwei[anthropic,mcp]'  # 按需接入 Claude 与 MCP
```

图、中间件、内存存储和本地执行驱动随核心提供。`graph`、`middleware`、`memory`、`sandbox` 是保留的兼容 extra 名，不代表额外依赖。`all` 安装所有可选 Python 扩展；Docker、数据库服务和供应商账号仍需自行配置。

## 开始一次对话

设置 `OPENAI_API_KEY`，然后运行：

```python
import asyncio
from wuwei import Agent, LLMGateway, ToolRegistry

registry = ToolRegistry()

@registry.tool()
def greet(name: str) -> str:
    """按姓名生成问候。"""
    return f"你好，{name}"

async def main():
    async with LLMGateway.from_env(provider="openai") as llm:
        agent = Agent(llm=llm, tools=registry, load_builtins="minimal")
        session = agent.create_session()
        result = await agent.run("向小明问好", session=session)
        print(result.status, result.content)
        async for event in agent.stream_events("改成更简短的问候", session=session):
            if event.type == "text_delta":
                print(event.data.get("content", ""), end="", flush=True)

asyncio.run(main())
```

`minimal` 只加载 calc/time/json/text。文件、网络、Git、npm 和脚本执行需按名选择；`load_builtins="all"` 会扩大可用能力，不应作为生产默认值。权限中间件、人工审批与 OS 沙箱是不同边界。

## 能力地图

| 需求 | 入口 | 使用要点 |
|---|---|---|
| 模型接入 | `LLMGateway` | OpenAI Chat Completions、Responses、Anthropic；兼容端点用 base_url；client/factory 可注入 |
| 会话与事件 | `Agent`、`AgentSession` | run/stream 共享显式或默认会话；同会话并发需遵循执行锁 |
| 工具 | `ToolRegistry`、`ToolExecutor` | 类型 schema、执行前参数验证、并发分组、超时与副作用状态 |
| 审批与恢复 | `HitlMiddleware`、checkpointer、`Agent.resume` | 持久完整批次；重复恢复不能重复领取；结果 unknown 需对账 |
| 结构化输出 | `Agent.run_structured` | 返回 `(typed_output, AgentRunResult)`；挂起时 output 为 None |
| 任务规划 | `Planner`、`PlanExecutor` | DAG 验证、拓扑调度、`PlanExecutor.resume`；保留非完成状态 |
| 多 Agent | SubAgent、AsyncSubAgent、MultiAgentGraph | 继承治理与统计；后台句柄在进程内，子检查点可持久化 |
| 图编排 | `StateGraph`、reducers、`END` | invoke/stream 共用状态归并；Memory/SQLite 检查点 |
| 权限与预算 | Permission/Budget middleware | 本地能力授权、路径边界、跨步骤配额与累计预算 |
| 上下文 | Context、压缩、Ledger、Memory、RAG | 摘要不替代原文；持久账本支持召回与 pin |
| 存储与检索 | FileStorage、CuratedMemoryStore、vectorstore | 内存后端与 Chroma/Qdrant/pgvector/Milvus 可选后端 |
| 插件与技能 | PluginManager、SkillManager | setup 注册工具/中间件；文件系统技能与 allowed-tools |
| 文件与代码 | LocalFS/InMemoryFS、sandbox、Code Mode | Code Mode 是可信进程内 Python，不是沙箱 |
| 协议 | MCP、A2A、AG-UI | MCP stdio/HTTP/旧 SSE；A2A 0.3 message/send；AG-UI SSE/NDJSON |
| 集成 | gateway、DecisionClient | 飞书/钉钉/Telegram/微信/Webhook 适配与决策模型客户端 |
| 运维与评估 | EventBus、ActionLog、evals、debug | 事件、动作记录、评估 CLI；fork 默认模拟副作用 |
| 配置与解析 | YAML、output parsers、Runnable | 配置错误启动失败；JSON/Pydantic 输出解析与管道组合 |

## 内置插件

基础插件：calc、time、json、text。按需插件：file、git、npm、python、http、decision、skill、rag、mcp、vfs。实际工具名与参数以 `agent.tool_registry.list_tools()` / `tool.to_schema()` 为准，避免把插件名当工具名。

## 结果不是只有成功或异常

检查 `result.status`：`completed`、`interrupted`、`incomplete`、`unknown`、`error`、`cancelled`。达到步数上限是 incomplete；动作可能已经发生却没有确定结果时是 unknown。两者不能作为成功交给下游，也不能盲目重试。

## 文档与测试

- [使用指南](USAGE.md)：模型、会话、工具、审批、委派和协议。
- [迁移指南](MIGRATION.md)：兼容入口与行为变化。
- [在线文档入口](docs/index.md)及[可执行离线示例](docs/examples/offline.md)。
- [更新日志](CHANGELOG.md)。

```bash
pytest                 # 默认离线测试
pytest --live          # 显式启用在线测试；需要对应账号、密钥与服务
pytest tests/test_documentation_hardening.py -q
```

离线通过证明本地契约，不代表所有供应商、远程 MCP、设备或部署环境已经互操作验收。

## 许可证

Apache-2.0。
