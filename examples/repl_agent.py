"""最小 REPL 示例：自定义工具 + HITL 审批 + 流式事件

运行前在项目根目录准备 .env（示例）：
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini
或使用任意 OpenAI 兼容端点：
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://api.deepseek.com
    OPENAI_MODEL=deepseek-flash

    python examples/repl_agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.middleware import HitlMiddleware, MiddlewareStack
from wuwei.tools import ToolParameters, Tool, ToolRegistry

NOTES_FILE = Path("agent_notes.txt")


def save_note(content: str) -> dict:
    """Save a note to a local text file.

    :param content: note content to append
    """
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write(content.strip() + "\n")
    return {"ok": True, "path": str(NOTES_FILE)}


def create_agent() -> Agent:
    llm = LLMGateway.from_env()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="save_note",
            description="把一条笔记追加保存到本地文件",
            parameters=ToolParameters(),
            handler=save_note,
            execution={"side_effect": True, "requires_approval": True},
        )
    )

    # HITL：requires_approval=True 的工具（policy="declared"）需要人工确认；
    # 拒绝时结果会回喂给模型，agent 会向用户说明而不是假装成功
    async def approval_provider(tool_call):
        print(f"\n是否允许执行 {tool_call.function.name}？(y/n)")
        return input().strip().lower() == "y"

    middleware = MiddlewareStack()
    middleware.add(HitlMiddleware(approval_provider=approval_provider, approval_policy="declared"))

    return Agent(
        llm=llm,
        tools=registry,
        load_builtins=["time"],  # 额外装载内置时间工具；默认只装 minimal 组
        default_system_prompt=(
            "你是一个简洁可靠的命令行助手。"
            "需要记录信息时调用 save_note。"
            "如果工具返回 ok=false 或被拒绝，必须明确告诉用户操作没有完成。"
        ),
        middleware=middleware,
    )


async def print_stream(agent: Agent, user_input: str) -> None:
    has_text = False
    async for event in agent.stream_events(user_input):
        if event.type == "text_delta":
            if not has_text:
                print("\n助手> ", end="", flush=True)
                has_text = True
            print(event.data.get("content", ""), end="", flush=True)
        elif event.type == "tool_start":
            print(f"\n[tool:start] {event.data.get('tool_name')}", flush=True)
        elif event.type == "tool_end":
            print(f"\n[tool:end] {event.data.get('tool_name')}", flush=True)
        elif event.type == "tool_error":
            print(f"\n[tool:error] {event.data.get('output')}", flush=True)
        elif event.type == "error":
            print(f"\n[error] {event.data.get('message')}", flush=True)
    if has_text:
        print()


async def main() -> None:
    agent = create_agent()
    print("wuwei REPL（exit 退出）。run() 不传 session 时复用默认会话，多轮上下文连续。")
    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "exit":
            break
        await print_stream(agent, user_input)

    usage = agent._default_session.last_usage if agent._default_session else {}
    print(f"\n本次会话 token 用量: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
