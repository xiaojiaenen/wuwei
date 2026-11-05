# 工具执行契约

工具经过注册 → 模型选择 → 参数验证 → 治理 → handler → 结果反馈。`ToolRegistry` 提供 register/register_callable/tool/get/list_tools；工具对象的 to_schema() 返回模型可见 JSON schema。

同步 handler 在线程执行，async handler 直接 await。参数不是对象、缺少必填字段或类型不符时不执行。`ToolExecutor.execute(..., concurrent=True)` 依 concurrency_safe 分组：`is_concurrency_safe=None`（默认）时按 side_effect 推导，写类工具串行；并发度受 `max_concurrency` 限制。并发声明不能代替真实资源冲突分析。

副作用声明 side_effect=True；timeout_seconds 到期只限制等待，不证明远端或线程已停止。超时结果在 `side_effect=True` 且未声明 `idempotent=True` 时标 unknown，须先对账；声明幂等的工具标失败并允许重试。unknown 结果须先对账。需要审批时设置 requires_approval，再安装 HITL 中间件。

单个工具输出超过 `max_output_chars`（默认 100k 字符）会截断并标注丢弃长度，避免一次调用占满上下文。

默认插件 minimal 包括 calc/time/json/text。file/git/npm/python/http/decision/skill/rag/mcp/vfs 按需加载。实际工具名与参数通过 registry.list_tools() 与 tool.to_schema() 查询。

参见[自定义工具](custom.md)和[离线执行示例](../examples/offline.md)。
