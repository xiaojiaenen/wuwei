"""
Wuwei Agent 框架完整流程示例。

演示组件：
- LLMGateway           LLM 调用网关（OpenAI 兼容协议）
- Agent / AgentSession 单 Agent 入口 + 会话容器
- Context              内存消息容器
- Tool / ToolRegistry  工具定义与注册
- Skill 体系：
    FileSystemSkillProvider  从文件系统扫描 SKILL.md
    SkillManager             聚合多个 SkillProvider
    SkillHook                注入 skill 使用指引到 system prompt
    skill 工具                list_skills / load_skill / run_skill_python_script
- RuntimeHook 体系：
    SkillHook            注入 skill 使用指引
    ContextCompressionHook  历史压缩 + 滑动窗口 + 内存裁剪
    StorageHook           对话持久化（增量追加）
    HitlHook              人类审批（save_note 需确认）
    ConsoleHook           调试日志
- FileStorage           文件存储（meta.json + jsonl）

流程：
  用户输入 → Agent.stream_events
    → AgentRunner 执行循环：
      1. copy context → before_llm hooks:
           SkillHook: 注入 skill 使用指引到 system prompt
           ContextCompressionHook: 超阈值则压缩旧轮次、裁剪内存
           StorageHook(step=0): 存 meta + user msg
           HitlHook: 透传（before_llm 无逻辑）
           ConsoleHook: 打印调用日志
      2. LLM 调用（可能触发 list_skills → load_skill → run_skill_python_script）
      3. 流式结束后触发 after_ai_message → StorageHook 持久化完整 AI 回复
      4. 若有 tool_calls → before_tool/after_tool hooks（审批/持久化工具结果）
      5. 重复直到 finish_reason=stop 或达到 max_steps
    → 返回结构化事件流
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from wuwei import (
    Agent,
    AgentSession,
    ConsoleHook,
    ContextCompressionHook,
    FileStorage,
    FileSystemSkillProvider,
    HitlHook,
    LLMGateway,
    SkillHook,
    SkillManager,
    StorageHook,
    ToolRegistry,
)
from wuwei.runtime import ApprovalPolicy, ConsoleApprovalProvider
from wuwei.memory.context_compressor import LLMContextCompressor
from wuwei.tools.builtin.skill_tools import register_skill_tools

# ── 配置 ──────────────────────────────────────────────
ENV_FILE = Path(__file__).with_name(".env")
STORAGE_ROOT = ".wuwei_demo_sessions"
SKILLS_DIR = Path(__file__).parent / "skills"
SESSION_ID = "demo-session"


# ── 自定义工具 ─────────────────────────────────────────
def add(a: int, b: int) -> dict:
    """两数相加。"""
    return {"result": a + b}


def save_note(content: str) -> dict:
    """保存笔记到本地文件。"""
    path = Path("agent_notes.txt")
    with path.open("a", encoding="utf-8") as file:
        file.write(content.strip() + "\n")
    return {"ok": True, "path": str(path)}


# ── 构建 Agent ────────────────────────────────────────
def create_agent() -> Agent:
    # 1. LLM 网关：
    #    - 如果 examples/.env 存在，优先读取它
    #    - 否则走 LLMGateway 默认查找逻辑 / 当前环境变量
    env_file = str(ENV_FILE) if ENV_FILE.exists() else None
    llm = LLMGateway.from_env(env_file=env_file)

    # 2. Skill 体系
    skill_provider = FileSystemSkillProvider(skill_path=str(SKILLS_DIR))
    skill_manager = SkillManager([skill_provider])

    # 3. 工具注册
    tools = ToolRegistry.from_builtin(["time"])          # 内置：get_now
    tools.register_callable(add)                         # 自定义：add
    tools.register_callable(save_note)                   # 自定义：save_note
    # 内置 skill 工具：list_skills / load_skill / run_skill_python_script
    register_skill_tools(tools, skill_manager)

    # 4. 持久化存储
    storage = FileStorage(STORAGE_ROOT)

    # 5. Hook 链（按注册顺序依次执行 before_* / after_*）
    hooks = [
        # 5a. Skill 钩子：注入 skill 使用指引到 system prompt（需在最前面）
        SkillHook(),
        # 5b. 上下文压缩：超过 16 轮触发摘要，内存保留最近 4 轮
        ContextCompressionHook(
            compressor=LLMContextCompressor(llm),
            compress_after_turns=16,
            keep_recent_turns=4,
        ),
        # 5c. 持久化：每条消息即时落盘
        #     非流式 assistant 走 after_llm；流式 assistant 走 after_ai_message。
        StorageHook(storage),
        # 5d. HITL 审批：save_note 需人类确认
        HitlHook(
            provider=ConsoleApprovalProvider(),
            policy=ApprovalPolicy(require_approval_tools={"save_note"}),
        ),
        # 5e. 调试日志
        ConsoleHook(),
    ]

    return Agent(
        llm=llm,
        tools=tools,
        default_system_prompt="你是一个简洁可靠的命令行助手。",
        default_max_steps=8,
        hooks=hooks,
    )


# ── 流式输出 ──────────────────────────────────────────
async def print_stream(agent: Agent, session: AgentSession, user_input: str):
    has_text = False

    async for event in agent.stream_events(user_input, session=session):
        if event.type == "text_delta":
            if not has_text:
                print("\n助手> ", end="", flush=True)
                has_text = True
            print(event.data.get("content", ""), end="", flush=True)

        elif event.type == "tool_start":
            print(
                f"\n  [tool:start] {event.data.get('tool_name')} "
                f"args={event.data.get('args')}",
                flush=True,
            )

        elif event.type == "tool_end":
            output = str(event.data.get("output", ""))[:80]
            print(f"\n  [tool:end]   {event.data.get('tool_name')} → {output}", flush=True)

        elif event.type == "error":
            print(f"\n  [error] {event.data.get('message')}", flush=True)

        elif event.type == "done":
            if has_text:
                print()
            usage = event.data.get("usage")
            if usage:
                print(f"  [done] tokens={usage} latency={event.data.get('latency_ms')}ms")


# ── 主流程 ────────────────────────────────────────────
async def main():
    agent = create_agent()

    # 尝试恢复已有会话
    storage = FileStorage(STORAGE_ROOT)
    session = await storage.load(SESSION_ID)

    if session:
        print(f"已恢复会话 {SESSION_ID}，历史 {len(session.context.get_messages())} 条消息")
    else:
        session = agent.create_session(session_id=SESSION_ID)
        print(f"新建会话 {SESSION_ID}")

    skills = agent.tool_registry.list_tools()
    print(f"已注册 {len(skills)} 个工具: {[t.name for t in skills]}")
    print(f"LLM 配置文件：{ENV_FILE}（不存在时使用环境变量或自动查找）")
    print(f"会话存储目录：{STORAGE_ROOT}")
    print("提示：让助手保存笔记会触发 HITL 审批；询问代码审查会触发 skill 流程。")
    print("输入 exit / quit / q 退出\n")

    while True:
        try:
            user_input = input("你> ").strip()
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
            print(f"执行失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
