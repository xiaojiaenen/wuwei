from __future__ import annotations

import asyncio
from pathlib import Path

from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.runtime import ApprovalPolicy, ConsoleApprovalProvider, HitlHook
from wuwei.tools import ToolRegistry


NOTES_FILE = Path("agent_notes.txt")
ENV_FILE = Path(__file__).with_name(".env")


def save_note(content: str) -> dict:
    """Save a note to a local text file.

    :param content: note content to append
    """
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write(content.strip() + "\n")
    return {"ok": True, "path": str(NOTES_FILE)}


def create_agent() -> Agent:
    llm = LLMGateway.from_env(env_file=str(ENV_FILE))

    tools = ToolRegistry.from_builtin(["time"])
    tools.register_callable(save_note)

    return Agent(
        llm=llm,
        tools=tools,
        default_system_prompt=(
            "你是一个简洁可靠的命令行助手。"
            "可以正常回答问题；需要记录信息时可以调用 save_note。"
            "如果工具返回 ok=false，或者工具结果说明人工拒绝、未执行、未完成，"
            "必须明确告诉用户操作没有完成，不要声称已经记录或已经执行。"
        ),
        hooks=[
            HitlHook(
                provider=ConsoleApprovalProvider(),
                policy=ApprovalPolicy(require_approval_tools={"save_note"}),
            )
        ],
    )


async def print_stream(agent: Agent, session, user_input: str) -> None:
    has_text = False

    async for event in agent.stream_events(user_input, session=session):
        if event.type == "text_delta":
            if not has_text:
                print("\n助手> ", end="", flush=True)
                has_text = True
            print(event.data.get("content", ""), end="", flush=True)
            continue

        if event.type == "tool_start":
            print(
                f"\n[tool:start] {event.data.get('tool_name')} "
                f"{event.data.get('args')}",
                flush=True,
            )
            continue

        if event.type == "tool_end":
            print(f"\n[tool:end] {event.data.get('tool_name')}", flush=True)
            continue

        if event.type == "error":
            print(f"\n[error] {event.data.get('message')}", flush=True)
            continue

        if event.type == "done":
            if has_text:
                print()
            usage = event.data.get("usage")
            if usage:
                print(f"[usage] {usage}", flush=True)


async def main() -> None:
    agent = create_agent()
    session = agent.create_session(session_id="while-true-demo")

    print("Wuwei Agent 已启动。输入 exit / quit / q 退出。")
    print("提示：让助手记录内容时会触发 HITL 审批。")
    print(f"LLM 配置文件：{ENV_FILE}")

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        try:
            await print_stream(agent, session, user_input)
        except Exception as exc:
            print(f"Agent 执行失败：{type(exc).__name__}: {exc}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
