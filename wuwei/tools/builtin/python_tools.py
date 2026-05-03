from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from wuwei.tools.registry import ToolRegistry

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_OUTPUT_LIMIT = 4_000


def _resolve_workspace_path(path: str, *, workspace: str = ".") -> Path:
    root = Path(workspace).resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"路径必须位于 workspace 内: {path}")
    return target


def _truncate_output(text: str, *, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit], True


def _parse_args(args_json: str) -> list[str]:
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise ValueError("args_json 必须是合法 JSON 数组") from exc

    if not isinstance(args, list):
        raise ValueError("args_json 必须是 JSON 数组")

    return [str(item) for item in args]


def register_python_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        name="run_python_script",
        description=(
            "执行 workspace 内的 Python 脚本。脚本路径必须位于 workspace 内，"
            "支持 JSON 数组形式的命令行参数、超时和输出截断。"
        ),
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS + 1,
        side_effect=True,
        requires_approval=True,
    )
    def run_python_script(
        script_path: str,
        args_json: str = "[]",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
        workspace: str = ".",
    ) -> dict[str, Any]:
        """执行 Python 脚本。

        :param script_path: 相对 workspace 的 Python 脚本路径
        :param args_json: 传给脚本的命令行参数，必须是 JSON 数组
        :param timeout_seconds: 执行超时时间，默认 10 秒
        :param max_output_chars: stdout/stderr 最大返回字符数
        :param workspace: 工作区根目录，默认当前目录
        """
        root = Path(workspace).resolve()
        target = _resolve_workspace_path(script_path, workspace=workspace)

        if not target.is_file():
            raise FileNotFoundError(f"脚本不存在: {script_path}")
        if target.suffix != ".py":
            raise ValueError("只允许执行 .py 脚本")

        command = [sys.executable, str(target), *_parse_args(args_json)]
        try:
            result = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_truncated = _truncate_output(
                exc.stdout or "",
                limit=max_output_chars,
            )
            stderr, stderr_truncated = _truncate_output(
                exc.stderr or "",
                limit=max_output_chars,
            )
            return {
                "ok": False,
                "error": {
                    "type": "ToolTimeout",
                    "message": f"Python 脚本执行超时（>{timeout_seconds} 秒）",
                    "retryable": True,
                },
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }

        stdout, stdout_truncated = _truncate_output(result.stdout, limit=max_output_chars)
        stderr, stderr_truncated = _truncate_output(result.stderr, limit=max_output_chars)
        return {
            "ok": result.returncode == 0,
            "script_path": str(target),
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
