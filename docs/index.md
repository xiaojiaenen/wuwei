# Wuwei

面向工具调用、多轮会话与可恢复审批的 Python Agent 框架。Python 3.11+，Apache-2.0。

从[安装](getting-started/installation.md)和[快速上手](getting-started/quickstart.md)开始；不调用在线模型也可以运行[离线示例](examples/offline.md)。

## 按任务进入

- [工具注册与参数验证](tools/custom.md)：将自己的函数交给模型调用。
- [会话](core/session.md)：普通与流式入口共用历史。
- [审批与恢复](hitl/overview.md)：等待授权，不丢失待执行批次。
- [规划与委派](planning/overview.md)：DAG 调度与可恢复成员结果。
- [架构](core/architecture.md)：执行、权限、存储和协议的边界。
- [协议与部署](protocols.md)：MCP、A2A 0.3 和 AG-UI。
- [API 索引](api/reference.md)：模块和公共入口。

completed 才代表完成；incomplete、unknown 与 interrupted 需要不同后续动作。权限检查不是 OS 隔离，检查点不是跨系统副作用 exactly-once。
