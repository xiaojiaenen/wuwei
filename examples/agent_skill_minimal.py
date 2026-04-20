import asyncio
from pathlib import Path

from wuwei import Agent, FileSystemSkillProvider, LLMGateway, SkillHook, SkillManager
from wuwei.runtime import ConsoleHook
from wuwei.tools import ToolRegistry
from wuwei.tools.builtin import register_file_tools, register_skill_tools, register_time_tools


def build_llm() -> LLMGateway:
    return LLMGateway.from_env(env_prefix="WUWEI")


def build_skill_manager() -> SkillManager:
    skills_dir = Path(__file__).parent / "skills"
    provider = FileSystemSkillProvider(str(skills_dir))
    return SkillManager([provider])


async def main() -> None:
    skill_manager = build_skill_manager()

    registry = ToolRegistry()
    register_file_tools(registry)
    register_time_tools(registry)
    register_skill_tools(registry, skill_manager)

    agent = Agent(
        llm=build_llm(),
        tools=registry,
        default_system_prompt=(
            "你是一个中文助手。"
            "普通问题直接回答；只有在任务明显匹配某个专门 workflow 时才使用 skill 工具。"
        ),
        default_max_steps=8,
        hooks=[SkillHook(), ConsoleHook()],
    )

    session = agent.create_session(session_id="agent-skill-minimal")
    question = (
        "请帮我做一个 Python 项目发版前检查清单。"
        "如果你已经有现成的 skill/workflow，就按它执行；"
        "如果 skill 里要求跑脚本，也请照做。"
    )

    print("question:", question)
    print("answer:")
    async for chunk in await agent.run(question, session=session, stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
