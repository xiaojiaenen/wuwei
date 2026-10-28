# 执行与治理边界

Agent → AgentRunner → LLMGateway / ToolExecutor。MiddlewareStack 围绕模型、工具和运行生命周期提供扩展；PluginManager 向注册表安装能力，而不是另建执行循环。

普通、流式、结构化输出与子代理共享治理链路。每次模型请求读取当前工具能力视图，动态发现后下一步可获得新 schema。父子预算与 usage 统一归因，子代理不能丢掉父权限。

## 状态的归属

- session：历史、会话配置、持久标识。
- run：累计使用量、治理状态、完整工具批次与中断。
- step：单次模型与工具阶段。
- checkpoint：可恢复运行快照；原子 claim 防止重复恢复。

副作用的外部结果与本地检查点不是同一事务。工具已开始但结果无法确定时保留 unknown，不自动重放。自定义后端必须支持原子领取，内存后端不能跨进程恢复。

## 安全组件各司其职

PermissionMiddleware 检查能力/路径/配额，HITL 记录一次人工决定，预算中间件限制累计开销。Docker/E2B 才是独立执行环境；Code Mode 的进程内 Python 不能充当沙箱。

## 可观测与协议

EventBus 输出事件，ActionLog 记录动作；协议出口映射结果状态，不改变内核运行完成条件。参见[协议](../protocols.md)与[API 索引](../api/reference.md)。
