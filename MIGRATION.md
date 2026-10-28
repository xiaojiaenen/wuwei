# 迁移到 Wuwei 3.3

先在独立环境升级并运行本地回归，再切换业务流量。保留旧检查点、会话数据与供应商配置的恢复副本。

## 保留的兼容入口

- provider `zhipu`、`dashscope`、`ollama` 作为弃用别名仍可使用，并保留各自环境变量命名空间；新代码使用 `provider="openai"` 配合 base_url/model，或 `from_env(..., env_prefix=...)`。
- `LLMResponse(content=..., tool_calls=..., reasoning_content=...)` 仍受支持；规范形态是 `LLMResponse(message=...)`。不要同时传两种形态。旧属性仍可读取。
- `gateway.adapter` 仍可访问，已有自定义适配器不必立即重写。新集成优先注入 client/client_factory 或 adapter/adapter_factory，明确资源与重试所有权。
- graph/middleware/memory/sandbox 兼容 extra 保留为空；相应核心能力无需单独安装。
- A2A tasks/send 为弃用兼容入口；新客户端使用 0.3 JSON-RPC message/send。当前不声明 A2A 1.0。

## 替换旧示例

用 `ToolRegistry().tool()` 或 register_callable 注册函数，再将 registry 传入 Agent。用 PluginManager + Plugin.setup(ctx) 管理扩展。图节点用 StateGraph、END 和 reducers；技能用 FileSystemSkillProvider/SkillManager。中间件栈先构造，再逐个 add。

```python
from wuwei import ToolRegistry
registry = ToolRegistry()
@registry.tool()
def count(values: list[str]) -> int:
    return len(values)
assert registry.get("count").parameters.required == ["values"]
```

默认内置工具只加载 minimal；需要文件/网络/脚本能力时按名加载，并配置权限及审批。

## 必须处理的状态与返回值

- run/stream 共用默认会话；业务隔离用显式 session，AG-UI threadId 映射 session_id。
- run_structured 返回 `(typed_output, AgentRunResult)`，interrupted 时 output=None；不再把第二项当裸 LLMResponse。
- incomplete、unknown、error、cancelled 均不是 completed。unknown 不能盲目重试副作用。
- Agent.resume 消费具体审批，重复领取会失败；自定义 checkpointer 需要原子 claim。重启恢复要保留完整检查点配置和业务访问控制。
- fork 默认隔离新分支并模拟副作用，真实重放需 allow_side_effects=True。
- PlanExecutor.resume 续跑计划审批；resume_async_task 续跑进程内后台子任务。后台句柄不跨进程恢复，子运行检查点可以持久化。
- MultiAgentGraph 的 DelegationStoppedError.result 保留可恢复成员结果，调用方必须处理而不是把它当普通成功字符串。

## 代码执行与重试

工具参数无效时不再调用 handler。同步 handler 在线程执行，超时不证明动作停止。声明 side_effect 的工具不自动重试。

Code Mode 的嵌套工具必须有治理 callback；直接调用需要显式 `governed_execute`。这是可信进程内代码，不是 OS 沙箱。

自建 SDK client 默认 max_retries=0；借入 client 应关闭 SDK 重试或明确唯一重试所有者。模型 gateway 的 retry 和输出校验纠正是不同预算。MCP 的 http 与旧 sse 传输分开配置。

## 升级验证

```bash
pytest
pytest tests/test_documentation_hardening.py -q
# 配置真实服务后显式开启：
pytest --live
```

依赖范围以 pyproject.toml 为准：OpenAI/Anthropic 的声明下限对应当前测试版本；不把本地 fake/MockTransport 测试解释成所有兼容网关已验收。

## 3.3 新功能的迁移注意事项

- 普通工具返回的status/ok字段只属于业务正文。自定义中间件要表达执行unknown/error时，改用ToolOutcome或显式trusted tool_outcome信封。
- run_structured在真实LLMGateway上默认auto模式。若输出是开放字典或提供商不支持严格schema，明确选output_mode="prompted"；旧自定义LLM保留提示后校验。
- 有output_model的子代理现在返回验证后的结构结果；未配置时保持字符串结果。业务依赖、文件和上下文都显式选择，不自动继承敏感资源。
- EvalCase默认要求completed且有断言；状态smoke需allow_empty_assertions=True。工具断言默认检查成功执行，不再用assistant的调用意图冒充执行；需要意图测试时选tool_trajectory="requested"。
- rubric默认需要judge；允许缺失时设judge_required=False，报告记为skipped而不是通过门禁。baseline用例名必须唯一。
- RunState v1快照可迁移到v2，不支持的未来版本明确拒绝。上线前保留原检查点和会话备份，不把新快照交给旧运行时重放。
- SessionStore、ArtifactStore和JobStore的namespace/owner由可信宿主鉴权后提供；不替代账号、租户或审批人权限验证。
- 参见[持久运行指南](docs/guides/durable-execution.md)和[模型契约](docs/model-contracts.md)。上述新能力从3.3.0开始提供。
