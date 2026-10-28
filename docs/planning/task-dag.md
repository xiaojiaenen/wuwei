# 显式构建任务 DAG

```python
from wuwei.planning import Task, TaskList
plan = TaskList(tasks=[
    Task(id=1, description="整理输入", next=[2], status="pending"),
    Task(id=2, description="输出摘要", next=[], status="pending"),
])
assert plan.tasks[0].next == [2]
```

PlanExecutor 在执行前验证重复 ID、未知后继与循环依赖。只有上游完成的任务才就绪，独立任务可并行。返回 Task 的 result、error、usage、interrupt、session_id 用于应用呈现与恢复。

`await executor.resume(plan, task_id=1, approved=True)` 消费该任务的审批，然后调度新就绪后继。interrupted、incomplete、unknown 不能当 completed；未知副作用不进入自动重试。参见[规划概览](overview.md)。
