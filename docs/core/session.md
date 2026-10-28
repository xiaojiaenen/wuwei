# 会话与流式事件

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
