# 计划与多 Agent

`Planner` 生成 `Task` DAG，`PlanExecutor.execute(plan)` 验证依赖并调度。`PlanExecutor.resume(plan, task_id, approved=True)` 续跑审批任务及新就绪后继。保留返回的任务状态、interrupt、session_id 和 usage；挂起不是失败重试。

同步 SubAgent 与后台 AsyncSubAgent 使用共享 Runner 与继承治理。后台工具包括 start/check/cancel/resume；`resume_async_task` 可续跑挂起的子任务。句柄与调度任务保存在当前进程，不能宣称重启后仍有同一个后台句柄；子运行检查点可通过持久后端保存。

`MultiAgentGraph` 成员挂起或未完成时抛出 `DelegationStoppedError`，异常 `.result` 保留成员 status、usage 与 interrupt。应用取得成员恢复结果后再决定继续协作，不能把挂起成员当成功文本交给后继。
