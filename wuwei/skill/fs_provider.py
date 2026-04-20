from pathlib import Path

import yaml

from wuwei.skill.skill import Skill, SkillProvider


class FileSystemSkillProvider(SkillProvider):
    """从文件系统加载技能。"""

    def __init__(self, skill_path: str):
        self.root_dir = Path(skill_path).resolve()

    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""
        skills = []
        for path in self.root_dir.rglob("SKILL.md"):
            content = path.read_text(encoding="utf-8")
            skill = self._parse_skill(content, path=path)
            skills.append(skill)
        return skills

    def load_skill_instruction(self, skill_name: str) -> str | None:
        """根据技能名称加载完整的指令正文（Markdown 主体）"""
        for path in self.root_dir.rglob("SKILL.md"):
            content = path.read_text(encoding="utf-8")
            skill = self._parse_skill(content, path=path)
            if skill.name == skill_name:
                return skill.instruction
        return None

    def _split_yaml_frontmatter(self, content: str) -> tuple[dict, str]:
        """解析技能元信息"""
        content = content.strip()
        if not content.startswith("---"):
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        yaml_text = parts[1].strip()
        md_body = parts[2].strip()

        try:
            frontmatter = yaml.safe_load(yaml_text)
            if not isinstance(frontmatter, dict):
                frontmatter = {}
        except yaml.YAMLError:
            frontmatter = {}

        return frontmatter, md_body

    def _parse_skill(self, content: str, *, path: Path | None = None) -> Skill:
        frontmatter, md_body = self._split_yaml_frontmatter(content)
        name = frontmatter.get("name", "unknown")
        description = frontmatter.get("description", "")
        instruction = md_body.strip() if md_body else ""
        skill_path = str(path.parent) if path is not None else None
        return Skill(
            name=name,
            description=description,
            instruction=instruction,
            path=skill_path,
        )
