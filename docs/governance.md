# 工具治理与代码执行


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

## 权限、审批与预算

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

## 可信代码执行

Plugin 的 setup(ctx) 通过 ctx.tool_registry 与 ctx.middleware_stack 注册能力，由 PluginManager 管理生命周期。Agent 支持 plugins、plugin_dir。FileSystemSkillProvider 读取 SKILL.md，SkillManager 提供按需指令加载和 allowed-tools 约束。

Code Mode 用 `enable_code_mode(agent)` 注册 run_code；嵌套工具经当前运行治理。直接 `run_code(code, executor, governed_execute=callback)` 必须显式提供可信治理入口。它运行进程内 Python，拥有宿主权限，不是沙箱，不能安全执行任意不可信代码或强制终止死循环。隔离需求使用独立 Docker/E2B 执行路径，并配置资源/网络限制。
