# 技能系统

技能将可复用任务说明、引用与脚本放进目录。FileSystemSkillProvider 读取 SKILL.md，SkillManager 管理索引与按需指令加载。

```python
from wuwei import FileSystemSkillProvider, SkillManager
provider = FileSystemSkillProvider("./skills")
manager = SkillManager([provider])
```

将 skill_manager=manager 传入 Agent，按需选择 skill 插件。allowed-tools 限制技能活跃期间的工具视图/调用，不应放宽父级权限。脚本与外部引用仍需按可信程度和运行环境处理。

参见[编写技能](authoring.md)。
