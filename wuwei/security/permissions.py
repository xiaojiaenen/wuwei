"""Capability 权限模型 — Agent 的 OS 级安全层

回应"prompt injection 无完整解"的行业共识（Simon Willison 的致命三要素）：
根治不可行，工程上收缩爆炸半径。

三层防御：
1. **声明式 capability（deny-by-default）**：文件系统读写 glob、网络域名
   白名单、shell 开关、工具配额——越权调用返回结构化错误回喂模型，
   模型可改道而不是崩溃。
2. **不可信内容隔离（Rules of Two 工程化）**：来自网络/MCP 的工具结果
   被标记为不可信；一旦本次 run 出现过不可信内容，后续有副作用的工具
   一律拒绝（除非显式列入 trusted_tools）。
3. **审计**：每个允许/拒绝决策可回调 + 落日志。

用法（agent.yaml 也可以自行映射到这个结构）：
    perms = PermissionConfig(
        filesystem_write=["./out/**"],
        network_allow=["api.github.com"],
        shell_enabled=False,
    )
    agent = Agent(llm=llm, middleware=MiddlewareStack(PermissionMiddleware(perms)))
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from pathlib import Path
from collections.abc import Callable
from urllib.parse import urlparse
from typing import Literal

from pydantic import BaseModel, Field

from wuwei.core.message import ToolCall, ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext

logger = logging.getLogger("wuwei.security")

_UNTRUSTED_KEY = "untrusted_seen"


class PermissionConfig(BaseModel):
    """声明式权限配置（全部默认拒绝）。"""

    workspace: str = "."
    filesystem_read: list[str] = Field(default_factory=list)
    filesystem_write: list[str] = Field(default_factory=list)
    network_allow: list[str] = Field(default_factory=list)
    shell_enabled: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    tool_quotas: dict[str, int] = Field(default_factory=dict)
    trusted_tools: list[str] = Field(default_factory=list)
    # 工具分类（按名称前缀自动推断的补充/覆盖）："fs_read"|"fs_write"|"network"|"shell"
    tool_categories: dict[
        str, Literal["pure", "delegate", "fs_read", "fs_write", "network", "shell"]
    ] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}


_DEFAULT_CATEGORIES: list[tuple[str, str]] = [
    # (工具名前缀, 分类)
    ("read_text_file", "fs_read"),
    ("vfs_read_file", "fs_read"),
    ("vfs_ls", "fs_read"),
    ("vfs_grep", "fs_read"),
    ("vfs_write_file", "fs_write"),
    ("vfs_append_file", "fs_write"),
    ("vfs_delete_file", "fs_write"),
    ("file_to_md", "fs_read"),
    ("list_files", "fs_read"),
    ("load_skill_reference", "fs_read"),
    ("write_text_file", "fs_write"),
    ("append_text_file", "fs_write"),
    ("replace_text_in_file", "fs_write"),
    ("delete_file", "fs_write"),
    ("http_get", "network"),
    ("http_post", "network"),
    ("mcp__", "network"),
    ("git_", "shell"),
    ("npm_", "shell"),
    ("run_python_script", "shell"),
    ("run_skill_python_script", "shell"),
    ("run_code", "shell"),
]


class PermissionMiddleware(Middleware):
    """权限中间件：before_tool 校验 + after_tool 不可信标记。"""

    is_tool_guard = True

    def __init__(
        self,
        config: PermissionConfig,
        on_decision: Callable[[ToolCall, bool, str], None] | None = None,
    ):
        """
        Args:
            config: 权限配置
            on_decision: 审计回调 (tool_call, allowed, reason)
        """
        self.config = config
        self.on_decision = on_decision

    # ── 工具分类 ─────────────────────────────────────────────────────

    def _category(self, tool_call: ToolCall) -> str:
        name = tool_call.function.name
        override = self.config.tool_categories.get(name)
        if override:
            return override
        for prefix, category in _DEFAULT_CATEGORIES:
            if name.startswith(prefix):
                return category
        if name.startswith("task_") or name in {
            "start_async_task",
            "check_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }:
            return "delegate"
        if name in {
            "calculate",
            "get_now",
            "get_current_time",
            "tool_search",
            "read_context",
            "list_context",
            "pin_context",
            "list_skills",
            "load_skill",
        } or name.startswith(("json_", "text_")):
            return "pure"
        return "other"

    # ── 校验 ─────────────────────────────────────────────────────────

    def _deny(self, tool_call: ToolCall, reason: str) -> ToolMessage:
        self._audit(tool_call, False, reason)
        return ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": "permission_denied",
                        "message": f"工具 {tool_call.function.name} 被权限系统拒绝：{reason}",
                        "hint": "该操作不在授权范围内。请改用已授权的方式完成任务，或告知用户需要授权。",
                    },
                },
                ensure_ascii=False,
            ),
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            status="error",
        )

    def _audit(self, tool_call: ToolCall, allowed: bool, reason: str) -> None:
        level = logging.INFO if allowed else logging.WARNING
        logger.log(
            level,
            "[permission] %s tool=%s args=%s reason=%s",
            "ALLOW" if allowed else "DENY",
            tool_call.function.name,
            json.dumps(tool_call.function.arguments, ensure_ascii=False, default=str)[:200],
            reason,
        )
        if self.on_decision:
            self.on_decision(tool_call, allowed, reason)

    async def before_tool(self, ctx: MiddlewareContext, tool_call: ToolCall):
        name = tool_call.function.name
        category = self._category(tool_call)
        args = tool_call.function.arguments
        state = ctx.metadata.setdefault("permission_state", {})
        executor = ctx.config.get("tool_executor")
        tool = executor.registry.get(name) if executor is not None else None
        if tool is not None and tool.execution.capability:
            category = self.config.tool_categories.get(name, tool.execution.capability)
        explicitly_allowed = any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.config.allowed_tools
        )
        if (
            category not in {"pure", "delegate", "fs_read", "fs_write", "network", "shell"}
            and not explicitly_allowed
        ):
            return self._deny(tool_call, "未声明工具 capability 或 tool_categories，默认拒绝")

        # 1. 显式黑名单
        if name in self.config.denied_tools:
            return self._deny(tool_call, "在 denied_tools 名单中")

        # 2. 工具配额
        quota = self.config.tool_quotas.get(name)
        if quota is not None:
            used = state.setdefault(f"_quota_{name}", 0)
            identity = f"{ctx.metadata.get('run_id', '')}:{ctx.step}:{tool_call.id}"
            revision = tool_call.model_dump(mode="json")
            seen = state.setdefault("_quota_seen", {})
            if seen.get(identity) != revision:
                if used >= quota:
                    return self._deny(tool_call, f"超出配额 {used}/{quota}")
                state[f"_quota_{name}"] = used + 1
                seen[identity] = revision

        # 3. 不可信内容隔离：出现过不可信内容后，有副作用的工具一律拒绝
        if (
            state.get(_UNTRUSTED_KEY)
            and (
                category in ("fs_write", "shell", "network")
                or bool(tool and tool.execution.side_effect)
            )
            and name not in self.config.trusted_tools
        ):
            return self._deny(
                tool_call,
                "本次运行已出现不可信内容（网络/MCP 返回），有副作用的操作已被冻结（Rules of Two）",
            )

        # 4. 分类校验
        if category == "fs_read":
            path = str(
                args.get("path")
                or args.get("file_path")
                or ("." if name in {"list_files", "vfs_ls", "vfs_grep"} else "")
            )
            if not path or not self._glob_match(
                self.config.filesystem_read, path, self.config.workspace
            ):
                return self._deny(tool_call, f"读取路径 {path} 不在 filesystem_read 白名单")
        elif category == "fs_write":
            path = str(args.get("path") or args.get("file_path") or "")
            if not path or not self._glob_match(
                self.config.filesystem_write, path, self.config.workspace
            ):
                return self._deny(tool_call, f"写入路径 {path} 不在 filesystem_write 白名单")
        elif category == "network":
            url = str(args.get("url") or args.get("base_url") or "")
            host = urlparse(url).hostname or ""
            if not (name.startswith("mcp__") and explicitly_allowed) and (
                not host or not self._host_match(host)
            ):
                return self._deny(tool_call, f"主机 {host or '(未知)'} 不在 network_allow 白名单")
        elif category == "shell" and not self.config.shell_enabled:
            return self._deny(tool_call, "shell_enabled=False，shell 类工具已禁用")

        self._audit(tool_call, True, f"category={category}")
        return tool_call

    async def after_tool(self, ctx: MiddlewareContext, tool_message: ToolMessage):
        name = getattr(tool_message, "name", "") or ""
        category = self.config.tool_categories.get(name) or next(
            (c for p, c in _DEFAULT_CATEGORIES if name.startswith(p)), "other"
        )
        executor = ctx.config.get("tool_executor")
        tool = executor.registry.get(name) if executor is not None else None
        if tool and tool.execution.capability:
            category = self.config.tool_categories.get(name, tool.execution.capability)
        if category == "network":
            ctx.metadata.setdefault("permission_state", {})[_UNTRUSTED_KEY] = True
        return tool_message

    # ── 匹配器 ───────────────────────────────────────────────────────

    @staticmethod
    def _glob_match(patterns: list[str], path: str, workspace: str = ".") -> bool:
        root = Path(workspace).expanduser().resolve()
        candidate = Path(path).expanduser()
        if ".." in candidate.parts:
            return False
        target = (root / candidate).resolve()
        for pattern in patterns:
            p = Path(pattern).expanduser()
            if ".." in p.parts:
                continue
            # Relative rules never acquire access outside the configured root via symlinks.
            if not p.is_absolute() and target != root and root not in target.parents:
                continue
            if pattern.endswith("/**"):
                directory = (root / str(p)[:-3]).resolve()
                if target == directory or directory in target.parents:
                    return True
            elif not any(c in pattern for c in "*?["):
                if target == (root / p).resolve():
                    return True
            else:
                absolute_pattern = str(root / p)
                if fnmatch.fnmatchcase(str(target), absolute_pattern):
                    # A single-star rule never crosses a directory boundary.
                    if len(target.parts) == len(Path(absolute_pattern).parts):
                        return True
        return False

    def _host_match(self, host: str) -> bool:
        for pattern in self.config.network_allow:
            if pattern.startswith("*."):
                if host == pattern[2:] or host.endswith("." + pattern[2:]):
                    return True
            elif re.fullmatch(re.escape(pattern).replace(r"\*", "[^.]*"), host) or host == pattern:
                return True
        return False
