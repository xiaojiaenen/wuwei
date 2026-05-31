"""技能中间件"""

from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import SystemMessage
from wuwei.skill.skill import SkillManager
from wuwei.tools.base import Tool
from wuwei.tools.builtin.skill_tools import register_skill_tools


class SkillMiddleware(Middleware):
    """技能中间件

    整合技能指引注入和工具注册。
    一个入口完成两件事：
    1. 注入技能指引到系统提示词
    2. 提供技能工具列表供 Agent 注册

    示例：
        from wuwei.skill import SkillManager, FileSystemSkillProvider

        provider = FileSystemSkillProvider("skills/")
        manager = SkillManager([provider])
        middleware = SkillMiddleware(skill_manager=manager)

        agent = Agent(
            llm=llm,
            tools=middleware.get_tools(),  # 自动注册技能工具
            middleware=MiddlewareStack([middleware]),
        )
    """

    def __init__(self, skill_manager: SkillManager):
        """
        Args:
            skill_manager: 技能管理器
        """
        self.skill_manager = skill_manager
        self._injected = False
        self._tools: list[Tool] | None = None

    def get_tools(self) -> list[Tool]:
        """获取技能工具列表（懒加载）"""
        if self._tools is None:
            from wuwei.tools.registry import ToolRegistry
            registry = ToolRegistry()
            register_skill_tools(registry, self.skill_manager)
            self._tools = registry.list_tools()
        return self._tools

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前注入技能指引"""
        if self._injected:
            return ctx

        skills = self.skill_manager.list_skills()
        if not skills:
            return ctx

        # 构建技能指引
        skill_lines = []
        for skill in skills:
            skill_lines.append(f"- {skill.name}: {skill.description}")

        skill_text = "\n".join(skill_lines)

        system_prompt = f"""你可以使用以下技能：
{skill_text}

使用 view_skill 工具查看技能的详细内容和使用说明。"""

        # 注入到消息开头
        ctx.state.messages.insert(
            0,
            SystemMessage(content=system_prompt),
        )

        self._injected = True
        return ctx
