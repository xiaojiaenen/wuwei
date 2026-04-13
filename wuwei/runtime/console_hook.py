import json

from wuwei.runtime.hooks import RuntimeHook


class ConsoleHook(RuntimeHook):
    """用于调试运行行为的最小日志 hook。"""

    async def after_llm(self, session, response, *, step: int, task=None) -> None:
        print(
            f"[llm] session={session.session_id} "
            f"step={step} finish_reason={response.finish_reason}"
        )

    async def before_tool(self, session, tool_call, *, step: int, task=None) -> None:
        args_text = json.dumps(tool_call.function.arguments, ensure_ascii=False)
        print(
            f"[tool.start] session={session.session_id} "
            f"step={step} name={tool_call.function.name} args={args_text}"
        )

    async def after_tool(self, session, tool_call, tool_message, *, step: int, task=None) -> None:
        print(
            f"[tool.end] session={session.session_id} "
            f"step={step} name={tool_call.function.name} result={tool_message.content}"
        )

    async def on_task_start(self, session, task) -> None:
        print(f"[task.start] session={session.session_id} task={task.id}")

    async def on_task_end(self, session, task) -> None:
        print(f"[task.end] session={session.session_id} task={task.id} status={task.status}")
