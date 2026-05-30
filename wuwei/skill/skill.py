from dataclasses import dataclass, field
from typing import Protocol, Optional
from datetime import datetime


@dataclass
class Skill:
    """技能数据模型

    借鉴 AgentScope + Claude Code 的结构化元数据。
    """
    name: str
    description: str
    instruction: str
    path: str | None = None
    scripts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # 增强的元数据
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    when_to_use: str = ""  # 何时使用此技能
    allowed_tools: list[str] = field(default_factory=list)  # 允许使用的工具
    required_tools: list[str] = field(default_factory=list)  # 必需的工具
    model: str | None = None  # 指定模型
    tags: list[str] = field(default_factory=list)
    source: str = ""  # 技能来源路径
    created_at: datetime = field(default_factory=datetime.now)


class SkillProvider(Protocol):
    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""
        ...

    def load_skill_instruction(self, skill_name: str) -> str | None:
        """根据技能名称加载完整的指令正文（Markdown 主体）"""
        ...


class SkillManager:
    """技能管理器

    支持多 provider 技能加载和管理。
    """

    def __init__(self, skill_providers: list[SkillProvider] = None):
        self.skill_providers = skill_providers or []
        self._meta_index: dict[str, tuple[SkillProvider, Skill]] = {}
        self._rebuild_index()

    def add_provider(self, provider: SkillProvider) -> "SkillManager":
        """添加技能 provider"""
        self.skill_providers.append(provider)
        for meta in provider.list_skills():
            self._meta_index[meta.name] = (provider, meta)
        return self

    def _rebuild_index(self):
        """重建索引"""
        self._meta_index.clear()
        for provider in self.skill_providers:
            for meta in provider.list_skills():
                self._meta_index[meta.name] = (provider, meta)

    def refresh(self) -> None:
        """刷新所有 provider 的索引。"""
        for provider in self.skill_providers:
            refresh = getattr(provider, "refresh", None)
            if callable(refresh):
                refresh()
        self._rebuild_index()

    def list_skills(self) -> list[Skill]:
        """列出所有可用的技能。"""
        return [meta for _, meta in self._meta_index.values()]

    def list_names(self) -> list[str]:
        """列出所有技能名称"""
        return list(self._meta_index.keys())

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

    def list_by_tag(self, tag: str) -> list[Skill]:
        """按标签过滤技能"""
        return [
            meta
            for _, meta in self._meta_index.values()
            if tag in meta.tags
        ]

    def list_by_tool(self, tool_name: str) -> list[Skill]:
        """按工具过滤技能"""
        return [
            meta
            for _, meta in self._meta_index.values()
            if tool_name in meta.allowed_tools or not meta.allowed_tools
        ]
