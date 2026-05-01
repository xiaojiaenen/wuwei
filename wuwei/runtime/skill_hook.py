from __future__ import annotations

from typing import TYPE_CHECKING

from wuwei.llm import Message
from wuwei.runtime.hooks import RuntimeHook
from wuwei.tools import Tool

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.planning import Task


DEFAULT_SKILL_INSTRUCTION = (
    "Skill 是可选的专门能力，不是处理每个请求的默认步骤。"
    "对普通问答、直接的代码修改、常规文件读取、普通工具调用，不要使用 skill 相关工具。"
    "只有当用户请求明显像某种可复用的 playbook、checklist、领域专用流程，"
    "或你有充分理由怀疑存在高度匹配的专门技能时，才调用 `list_skills` 查看技能摘要。"
    "不要为了例行检查而在每轮都调用 `list_skills`。"
    "只有当某个 skill 的描述与当前任务明确匹配时，才调用 `load_skill` 加载正文；"
    "如果没有清晰匹配的 skill，就继续用普通方式完成任务。"
    "如果已加载 skill 声明了 references，只有当正文要求或任务确实需要补充细节时，"
    "才调用 `load_skill_reference` 按需读取单个参考文件。"
    "只有在已加载的 skill 正文明确要求时，"
    "才调用 `run_skill_python_script` 执行该 skill 自带的 Python 脚本。"
    "调用脚本时，必须传入 `load_skill` 返回的 `load_token`。"
)


class SkillHook(RuntimeHook):
    def __init__(self, instruction: str = DEFAULT_SKILL_INSTRUCTION) -> None:
        self.instruction = instruction.strip()

    async def before_llm(
        self,
        session: AgentSession,
        messages: list[Message],
        tools: list[Tool],
        *,
        step: int,
        task: Task | None,
    ) -> tuple[list[Message], list[Tool]]:
        if not self.instruction:
            return messages, tools

        for msg in messages:
            if msg.role == "system":
                base_prompt = (msg.content or "").rstrip()
                msg.content = (
                    f"{base_prompt}\n\n{self.instruction}" if base_prompt else self.instruction
                )
                break
        else:
            messages.insert(0, Message(role="system", content=self.instruction))

        return messages, tools
