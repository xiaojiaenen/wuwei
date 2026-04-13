from wuwei.skill.skill import SkillProvider, Skill


root_dir=".wuwei/skills"

class FileSystemSkillProvider(SkillProvider):
    """从文件系统加载技能。"""
    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""

    def load_skill_instruction(self, skill_name: str) -> str|None:
        """根据技能名称加载完整的指令正文（Markdown 主体）"""
