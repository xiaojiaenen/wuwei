# 工具执行契约

工具经过注册 → 模型选择 → 参数验证 → 治理 → handler → 结果反馈。`ToolRegistry` 提供 register/register_callable/tool/get/list_tools；工具对象的 to_schema() 返回模型可见 JSON schema。

同步 handler 在线程执行，async handler 直接 await。参数不是对象、缺少必填字段或类型不符时不执行。`ToolExecutor.execute(..., concurrent=True)` 依 is_concurrency_safe 分组；并发声明不能代替真实资源冲突分析。

副作用声明 side_effect=True；timeout_seconds 到期只限制等待，不证明远端或线程已停止。unknown 结果须先对账。需要审批时设置 requires_approval，再安装 HITL 中间件。

默认插件 minimal 包括 calc/time/json/text。file/git/npm/python/http/decision/skill/rag/mcp/vfs 按需加载。实际工具名与参数通过 registry.list_tools() 与 tool.to_schema() 查询。

参见[自定义工具](custom.md)和[离线执行示例](../examples/offline.md)。
