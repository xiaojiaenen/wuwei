"""SkillViewer 工具"""

from typing import Any

from wuwei.tools.base import Tool
from wuwei.skill.skill import SkillManager


class SkillViewerTool(Tool):
    """技能查看器工具

    Agent 可以通过此工具查看技能的详细内容和使用说明。
    """

    # Pydantic model_config 允许额外字段
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, skill_manager: SkillManager, **data):
        super().__init__(
            name="view_skill",
            description="查看指定技能的详细内容和使用说明",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要查看的技能名称",
                    }
                },
                "required": ["skill_name"],
            },
            handler=self._view_skill,
            **data,
        )
        # 使用 object.__setattr__ 绕过 Pydantic 字段校验
        object.__setattr__(self, 'skill_manager', skill_manager)

    async def _view_skill(self, skill_name: str) -> str:
        """查看技能内容"""
        mgr = object.__getattribute__(self, 'skill_manager')
        try:
            skill = mgr.get_skill(skill_name)
        except ValueError:
            available = ", ".join(mgr.list_names())
            return f"技能 '{skill_name}' 不存在。可用技能：{available}"

        # 构建技能信息
        info_parts = [
            f"# {skill.name}",
            f"描述：{skill.description}",
            f"版本：{skill.version}",
            "",
            "## 使用说明",
            skill.instruction,
        ]

        if skill.when_to_use:
            info_parts.extend(["", f"何时使用：{skill.when_to_use}"])
        if skill.allowed_tools:
            info_parts.extend(["", f"允许工具：{', '.join(skill.allowed_tools)}"])
        if skill.required_tools:
            info_parts.extend(["", f"必需工具：{', '.join(skill.required_tools)}"])
        if skill.tags:
            info_parts.extend(["", f"标签：{', '.join(skill.tags)}"])

        return "\n".join(info_parts)
