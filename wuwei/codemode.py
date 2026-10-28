"""Code Mode — 代码即动作

借鉴 smolagents 的 CodeAct 思路：让模型写一段 Python 一次性完成
"多次工具调用 + 中间数据处理 + 控制流"，替代逐个 JSON tool call——
轮次与 token 显著下降，且不受并行 tool call fan-out 上限（~13）约束。

⚠️ 安全边界：默认实现是**进程内执行**（工具调用需要访问进程内的
ToolRegistry），只应用于可信环境；不可信环境请配合 DockerDriver 沙箱
（见 wuwei.sandbox）并只暴露纯计算类代码路径。

用法：
    agent = Agent(llm=llm, tools=registry)
    enable_code_mode(agent)
    # 模型获得 run_code 工具，代码中可用：
    #   rows = await tools.search_web(query="...")   # 调用已注册工具
    #   summary = "\n".join(r["title"] for r in rows)
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
from typing import Any

from wuwei.core.message import FunctionCall, ToolCall
from wuwei.tools import Tool, ToolExecutor, ToolParameters
from wuwei.tools.executor import governed_tool_execution, bind_tool_execution


class ToolNamespace:
    """把 ToolRegistry 暴露为 `tools.<name>(**kwargs)` 的异步代理。"""

    def __init__(self, executor: ToolExecutor):
        self._executor = executor
        self.calls: list[dict] = []

    def __getattr__(self, name: str):
        tool = self._executor.registry.get(name)
        if tool is None:
            raise AttributeError(f"工具不存在: {name}（可用: {self.names()}）")

        async def _call(**kwargs) -> Any:
            self.calls.append({"tool": name, "arguments": kwargs})
            call = ToolCall(
                type="function",
                function=FunctionCall(name=name, arguments=kwargs),
            )
            callback = governed_tool_execution()
            if callback is None:
                raise RuntimeError("Nested tools require an active governed execution context")
            if name == "run_code":
                raise ValueError("Recursive Code Mode is not supported")
            message = await callback(call)
            return _maybe_json(message.content)

        return _call

    def names(self) -> list[str]:
        return [t.name for t in self._executor.registry.list_tools()]


def _maybe_json(content: str | None) -> Any:
    """工具输出尽量还原为结构化数据（错误信息保持原文）。"""
    if content is None:
        return None
    try:
        value = json.loads(content)
        if isinstance(value, dict) and value.get("ok") is False:
            return content  # 结构化错误保持原文便于模型阅读
        return value
    except (json.JSONDecodeError, TypeError):
        return content


_PRELUDE = (
    "import asyncio, json, math, re\n"
    "from itertools import chain, groupby\n"
)


async def run_code(code: str, executor: ToolExecutor, *, timeout: float = 60.0, governed_execute=None) -> str:
    """执行可信 Python（支持顶层 await）；这不是沙箱，超时不保证终止同步代码。"""
    namespace: dict[str, Any] = {
        "tools": ToolNamespace(executor),
        "math": math,
    }
    # 顶层 await 支持：包进 async 函数再执行
    indented = "\n".join("    " + line for line in code.splitlines())
    wrapped = f"{_PRELUDE}async def __user_code__():\n{indented}\n"
    try:
        compiled = compile(wrapped, "<run_code>", "exec")
    except SyntaxError as exc:
        return json.dumps(
            {"ok": False, "error": {"type": "SyntaxError", "message": str(exc)}},
            ensure_ascii=False,
        )

    stdout_buf = io.StringIO()
    try:
        exec(compiled, namespace)  # noqa: S102 - 受控环境，仅可信场景
        callback = governed_execute or governed_tool_execution()
        with bind_tool_execution(callback), contextlib.redirect_stdout(stdout_buf):
            await asyncio.wait_for(
                namespace["__user_code__"](), timeout=timeout
            )
        output = stdout_buf.getvalue()
    except TimeoutError:
        return json.dumps(
            {"ok": False, "error": {"type": "CodeTimeout", "message": f"执行超过 {timeout}s"}},
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "tool_calls": namespace["tools"].calls,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "ok": True,
            "output": output or None,
            "tool_calls": namespace["tools"].calls,
        },
        ensure_ascii=False,
        default=str,
    )


def enable_code_mode(
    agent,
    *,
    tool_name: str = "run_code",
    timeout: float = 60.0,
) -> Tool:
    """为 Agent 启用 Code Mode：注册 run_code 工具。

    代码中通过 `await tools.<工具名>(**参数)` 调用已注册工具；
    print 输出与每次工具调用记录都会回传给模型。
    """
    executor = agent.tool_executor

    async def _run_code(code: str) -> str:
        """执行一段 Python 代码，代码内可用 `await tools.<工具名>(**参数)` 调用工具。

        :param code: 完整的 Python 代码（支持顶层 await 与 print）
        """
        return await run_code(code, executor, timeout=timeout)

    tool = Tool(
        name=tool_name,
        description=(
            "执行一段 Python 代码以高效完成任务：代码中用 `await tools.<工具名>(**参数)` "
            "调用可用工具，可使用循环/条件/数据处理，print 的内容会返回给你。"
            f"可用工具: {', '.join(t.name for t in agent.tool_registry.list_tools())}"
        ),
        parameters=ToolParameters(properties={"code": {"type": "string"}}, required=["code"]),
        handler=_run_code,
        display_name="代码执行",
        execution={"side_effect": True, "requires_approval": False, "timeout_seconds": timeout + 5},
    )
    agent.tool_registry.register(tool)
    return tool
