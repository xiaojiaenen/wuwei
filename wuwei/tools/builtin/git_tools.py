from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from wuwei.tools.registry import ToolRegistry

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_OUTPUT_LIMIT = 12_000


def _resolve_workspace(workspace: str = ".") -> Path:
    root = Path(workspace).resolve()
    if not root.exists():
        raise FileNotFoundError(f"workspace 不存在: {workspace}")
    if not root.is_dir():
        raise NotADirectoryError(f"workspace 不是目录: {workspace}")
    return root


def _resolve_workspace_path(path: str, *, workspace: str = ".") -> Path:
    root = _resolve_workspace(workspace)
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"路径必须位于 workspace 内: {path}")
    return target


def _truncate(text: str, *, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit], True


def _run_git(
    args: list[str],
    *,
    workspace: str = ".",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    root = _resolve_workspace(workspace)
    command = ["git", *args]
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
                "message": f"git 命令执行超时（>{timeout_seconds} 秒）",
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


def register_git_tools(registry: ToolRegistry) -> None:
    @registry.tool(name="git_status", description="查看 workspace 内 Git 仓库状态。")
    def git_status(workspace: str = ".", short: bool = True) -> dict[str, Any]:
        """查看 Git 状态。

        :param workspace: Git 仓库目录
        :param short: 是否使用短格式
        """
        args = ["status", "--short", "--branch"] if short else ["status"]
        return _run_git(args, workspace=workspace)

    @registry.tool(name="git_diff", description="查看 workspace 内 Git diff。")
    def git_diff(
        path: str = "",
        staged: bool = False,
        workspace: str = ".",
        max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """查看 Git diff。

        :param path: 可选，相对 workspace 的文件路径
        :param staged: 是否查看已暂存 diff
        :param workspace: Git 仓库目录
        :param max_output_chars: 最大返回字符数
        """
        args = ["diff"]
        if staged:
            args.append("--staged")
        if path:
            _resolve_workspace_path(path, workspace=workspace)
            args.extend(["--", path])
        return _run_git(args, workspace=workspace, max_output_chars=max_output_chars)

    @registry.tool(name="git_log", description="查看 Git 提交日志。")
    def git_log(
        limit: int = 10,
        workspace: str = ".",
        max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """查看 Git 提交日志。

        :param limit: 最多返回提交数
        :param workspace: Git 仓库目录
        :param max_output_chars: 最大返回字符数
        """
        safe_limit = max(1, min(int(limit), 100))
        return _run_git(
            ["log", "--oneline", "--decorate", f"-{safe_limit}"],
            workspace=workspace,
            max_output_chars=max_output_chars,
        )

    @registry.tool(name="git_show", description="查看某个 Git revision 的内容或统计。")
    def git_show(
        revision: str = "HEAD",
        stat_only: bool = True,
        workspace: str = ".",
        max_output_chars: int = DEFAULT_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """查看 Git revision。

        :param revision: commit hash、tag、branch 或 HEAD
        :param stat_only: 是否只显示统计信息
        :param workspace: Git 仓库目录
        :param max_output_chars: 最大返回字符数
        """
        args = ["show", "--stat", revision] if stat_only else ["show", revision]
        return _run_git(args, workspace=workspace, max_output_chars=max_output_chars)

    @registry.tool(
        name="git_add",
        description="暂存 workspace 内的指定文件。",
        side_effect=True,
        requires_approval=True,
    )
    def git_add(path: str, workspace: str = ".") -> dict[str, Any]:
        """暂存文件。

        :param path: 相对 workspace 的文件路径
        :param workspace: Git 仓库目录
        """
        _resolve_workspace_path(path, workspace=workspace)
        return _run_git(["add", "--", path], workspace=workspace)

    @registry.tool(
        name="git_commit",
        description="创建 Git commit。通常应配合 HITL 审批使用。",
        side_effect=True,
        requires_approval=True,
    )
    def git_commit(message: str, workspace: str = ".") -> dict[str, Any]:
        """创建 Git commit。

        :param message: commit message
        :param workspace: Git 仓库目录
        """
        if not message.strip():
            raise ValueError("commit message 不能为空")
        return _run_git(["commit", "-m", message], workspace=workspace)
