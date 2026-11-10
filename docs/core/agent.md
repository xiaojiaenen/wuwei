# Agent 与运行结果

`Agent(llm=..., tools=registry, load_builtins="minimal")` 是门面，AgentRunner 执行模型/工具循环，AgentSession 持有会话配置与历史。

| 方法 | 结果 |
|---|---|
| await agent.run(text, session=...) | AgentRunResult |
| agent.stream_events(text, session=... / session_id=...) | 异步 AgentEvent 流 |
| await agent.run_structured(text, Model, session=...) | (Model 或 None, AgentRunResult) |
| await agent.resume(interrupt, approved=...) | 恢复同一挂起 run |
| await agent.fork(checkpoint_id) | 隔离的新会话/运行，默认模拟副作用 |
| await agent.run_plan(goal) | PlanRunResult |


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

结构化约束会并入首条 system 消息（而不是新增一条），保证 system 唯一且在首位——OpenAI 兼容端点（DeepSeek / vLLM / 内网网关）对此有硬校验。适配器在发请求前还会合并开头连续的 system，RAG、记忆等中间件注入的说明不会造成相邻双 system。

## 终态决定后续动作

completed 可交给下游；interrupted 展示审批并保存 interrupt；incomplete 保留未完成工作；unknown 先核对动作事实；error 处理故障；cancelled 保留已发生动作记录。达到步数上限不是成功。

`AgentRunResult` 的 `steps_used` 与 `max_steps` 可直接判断是否被截断（`termination_reason == "max_steps"`），不要匹配 content 文案。

usage、llm_calls、latency_ms 是本次结果统计。估算成本依赖价格配置，不能替代供应商账单。
