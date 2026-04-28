from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from wuwei.tools.registry import ToolRegistry

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_OUTPUT_LIMIT = 12_000


def _resolve_workspace(workspace: str = ".") -> Path:
    root = Path(workspace).resolve()
    if not root.exists():
        raise FileNotFoundError(f"workspace 不存在: {workspace}")
    if not root.is_dir():
        raise NotADirectoryError(f"workspace 不是目录: {workspace}")
    return root


def _truncate(text: str, *, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit], True


def _run_npm(
    args: list[str],
    *,
    workspace: str = ".",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    root = _resolve_workspace(workspace)
    command = ["npm", *args]
    try:
        result = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _truncate(exc.stdout or "", limit=max_output_chars)
        stderr, stderr_truncated = _truncate(exc.stderr or "", limit=max_output_chars)
        return {
            "ok": False,
            "command": command,
            "error": {
                "type": "ToolTimeout",
                "message": f"npm 命令执行超时（>{timeout_seconds} 秒）",
                "retryable": True,
            },
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    stdout, stdout_truncated = _truncate(result.stdout, limit=max_output_chars)
    stderr, stderr_truncated = _truncate(result.stderr, limit=max_output_chars)
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def _load_package_json(workspace: str) -> dict[str, Any]:
    root = _resolve_workspace(workspace)
    package_json = root / "package.json"
    if not package_json.is_file():
        raise FileNotFoundError("workspace 下不存在 package.json")

    payload = json.loads(package_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("package.json 内容必须是 JSON object")
    return payload


def register_npm_tools(registry: ToolRegistry) -> None:
    @registry.tool(name="npm_list_scripts", description="读取 package.json 中的 scripts 列表。")
    def npm_list_scripts(workspace: str = ".") -> dict[str, Any]:
        """列出 npm scripts。

        :param workspace: 包含 package.json 的目录
        """
        payload = _load_package_json(workspace)
        scripts = payload.get("scripts", {})
        if not isinstance(scripts, dict):
            scripts = {}
        return {"ok": True, "scripts": scripts}

    @registry.tool(name="npm_run_script", description="运行 package.json 中定义的 npm script。")
    def npm_run_script(
        script_name: str,
        args_json: str = "[]",
        workspace: str = ".",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """运行 npm script。

        :param script_name: package.json scripts 中的脚本名
        :param args_json: 传给 npm script 的额外参数，JSON 数组
        :param workspace: 包含 package.json 的目录
        :param timeout_seconds: 执行超时时间
        :param max_output_chars: stdout/stderr 最大返回字符数
        """
        scripts = _load_package_json(workspace).get("scripts", {})
        if not isinstance(scripts, dict) or script_name not in scripts:
            raise ValueError(f"package.json 中不存在 script: {script_name}")

        try:
            extra_args = json.loads(args_json)
        except json.JSONDecodeError as exc:
            raise ValueError("args_json 必须是合法 JSON 数组") from exc
        if not isinstance(extra_args, list):
            raise ValueError("args_json 必须是 JSON 数组")

        args = ["run", script_name]
        if extra_args:
            args.append("--")
            args.extend(str(item) for item in extra_args)
        return _run_npm(
            args,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )

    @registry.tool(
        name="npm_install_package",
        description="安装 npm 包。会修改 package.json/package-lock.json，通常应配合 HITL 审批使用。",
    )
    def npm_install_package(
        package: str,
        dev: bool = False,
        workspace: str = ".",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """安装 npm 包。

        :param package: 包名，例如 lodash 或 typescript@latest
        :param dev: 是否安装到 devDependencies
        :param workspace: 包含 package.json 的目录
        :param timeout_seconds: 执行超时时间
        :param max_output_chars: stdout/stderr 最大返回字符数
        """
        if not package.strip() or any(char.isspace() for char in package):
            raise ValueError("package 必须是单个 npm 包名，不能包含空白字符")

        _load_package_json(workspace)
        args = ["install", package]
        if dev:
            args.append("--save-dev")
        return _run_npm(
            args,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
