from pathlib import Path

import yaml

from wuwei.skill.skill import Skill, SkillProvider


class FileSystemSkillProvider(SkillProvider):
    """从文件系统加载技能。"""

    def __init__(self, skill_path: str):
        self.root_dir = Path(skill_path).resolve()
        self._skill_index: dict[str, Skill] | None = None

    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""
        return list(self._ensure_index().values())

    def load_skill_instruction(self, skill_name: str) -> str | None:
        """根据技能名称加载完整的指令正文（Markdown 主体）"""
        skill = self._ensure_index().get(skill_name)
        return skill.instruction if skill is not None else None

    def refresh(self) -> None:
        """清空缓存，并重新扫描 skill 目录。"""
        self._skill_index = self._build_index()

    def _ensure_index(self) -> dict[str, Skill]:
        if self._skill_index is None:
            self._skill_index = self._build_index()
        return self._skill_index

    def _build_index(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for path in sorted(self.root_dir.rglob("SKILL.md")):
            content = path.read_text(encoding="utf-8")
            skill = self._parse_skill(content, path=path)
            skills[skill.name] = skill
        return skills

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
        scripts = self._collect_relative_files(path.parent / "scripts") if path is not None else []
        references = (
            self._collect_relative_files(path.parent / "references") if path is not None else []
        )

        # agentskills.io 开放标准兼容：
        # - allowed-tools（连字符，逗号分隔字符串或列表）与 allowed_tools 等价
        # - license / metadata 字段直接透传
        allowed_tools = frontmatter.get("allowed_tools")
        if allowed_tools is None:
            raw_allowed = frontmatter.get("allowed-tools")
            if isinstance(raw_allowed, str):
                allowed_tools = [t.strip() for t in raw_allowed.split(",") if t.strip()]
            else:
                allowed_tools = raw_allowed or []

        # 递归解析 metadata 子表（开放标准的自由扩展位）
        extra_metadata = frontmatter.get("metadata") or {}

        return Skill(
            name=name,
            description=description,
            instruction=instruction,
            path=skill_path,
            scripts=scripts,
            references=references,
            version=str(frontmatter.get("version", "1.0.0")),
            author=frontmatter.get("author", ""),
            license=frontmatter.get("license", "MIT"),
            when_to_use=frontmatter.get("when_to_use", ""),
            allowed_tools=list(allowed_tools),
            required_tools=frontmatter.get("required_tools", []),
            model=frontmatter.get("model"),
            tags=frontmatter.get("tags", []),
            metadata=extra_metadata if isinstance(extra_metadata, dict) else {},
        )

    def _collect_relative_files(self, root: Path) -> list[str]:
        if not root.is_dir():
            return []
        skill_root = root.parent
        return sorted(
            path.relative_to(skill_root).as_posix() for path in root.rglob("*") if path.is_file()
        )
