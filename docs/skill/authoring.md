# 编写技能

在 skills/report/SKILL.md 中声明元信息与任务说明：

```markdown
---
name: report
description: 把给定材料整理成摘要
allowed-tools: text_trim, text_split
---
先确认输入材料，再按结论、依据、待办输出。没有依据时保留不确定性。
```

allowed-tools 中使用实际注册工具名，不是插件名。引用放在同一技能目录，脚本要写明输入输出与真实副作用。修改技能后 provider.refresh() 刷新索引。

技能指令不能覆盖宿主权限、审批与预算；它是任务方法，不是提权配置。
