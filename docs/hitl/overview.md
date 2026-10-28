# 审批与恢复

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
