from dataclasses import dataclass
from typing import Protocol


@dataclass
class Skill:
    name: str
    description: str
    instruction: str
    path: str | None = None


class SkillProvider(Protocol):
    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""
        ...

    def load_skill_instruction(self, skill_name: str) -> str | None:
        """根据技能名称加载完整的指令正文（Markdown 主体）"""
        ...


class SkillManager:
    def __init__(self, skill_providers: list[SkillProvider]):
        self.skill_providers = skill_providers
        self._meta_index: dict[str, tuple[SkillProvider, Skill]] = {}
        self._rebuild_index()

    def _rebuild_index(self):
        self._meta_index.clear()
        for provider in self.skill_providers:
            for meta in provider.list_skills():
                self._meta_index[meta.name] = (provider, meta)

    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""
        return [meta for _, meta in self._meta_index.values()]

    def get_skill(self, skill_name: str) -> Skill:
        """根据技能名称获取技能元数据。"""
        try:
            provider, meta = self._meta_index[skill_name]
        except KeyError as exc:
            raise ValueError(f"Skill '{skill_name}' not found") from exc
        return meta

    def load_skill_instruction(self, skill_name: str) -> str | None:
        """根据技能名称加载完整的指令正文（Markdown 主体）"""
        try:
            provider, meta = self._meta_index[skill_name]
        except KeyError as exc:
            raise ValueError(f"Skill '{skill_name}' not found") from exc
        return provider.load_skill_instruction(skill_name)
