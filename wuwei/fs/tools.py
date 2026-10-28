"""VFS 工具层 — 把 FSBackend 暴露为 wuwei 的 Tool / Plugin。

:func:`make_vfs_tools` 返回六个工具（write_file / read_file / append_file /
delete_file / ls / grep），:func:`make_vfs_plugin` 把它们包装成
:class:`wuwei.plugin.Plugin`，setup 时注册到 ToolRegistry。

工具名带 ``workspace_label`` 前缀（默认 ``vfs``），例如 ``vfs_write_file``，
避免与其它工作区的同名工具冲突。
"""

from __future__ import annotations

from typing import Any

from wuwei.fs.base import FSBackend
from wuwei.plugin import Plugin, PluginContext
from wuwei.tools.tool import Tool, ToolExecutionPolicy, ToolParameters

DEFAULT_READ_LIMIT = 20_000
"""read_file 工具默认的最大返回字符数。"""

DEFAULT_GREP_RESULTS = 100
"""grep 工具默认的最大匹配条数。"""

_DEFAULT_GREP_LINE_LIMIT = 500
"""grep 单行内容的最大返回长度。"""


def _truncate_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    """按最大字符数截断文本，返回 (内容, 是否截断)。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _params(
    properties: dict[str, Any],
    required: list[str],
) -> ToolParameters:
    """构造 JSON Schema 形式的工具参数。"""
    return ToolParameters(properties=properties, required=required)


def make_vfs_tools(
    backend: FSBackend,
    workspace_label: str = "vfs",
) -> list[Tool]:
    """根据 VFS 后端构造六个文件工具。

    :param backend: 实现 :class:`wuwei.fs.base.FSBackend` 协议的后端
    :param workspace_label: 工具名前缀（避免多工作区冲突），默认 ``vfs``
    :return: write_file / read_file / append_file / delete_file / ls / grep 六个 Tool
    """
    prefix = workspace_label.strip() or "vfs"
    label = f"[{prefix}]"

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    async def vfs_write_file(path: str, content: str, overwrite: bool = True) -> dict:
        """写入文本文件到 VFS（覆盖写），路径相对 VFS 根。"""
        if not overwrite:
            try:
                await backend.read_file(path)
            except (FileNotFoundError, IsADirectoryError):
                pass
            else:
                raise FileExistsError(f"文件已存在，设置 overwrite=true 才能覆盖: {path}")
        data = content.encode("utf-8")
        await backend.write_file(path, data)
        return {"ok": True, "workspace": prefix, "path": path, "bytes": len(data)}

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    async def vfs_read_file(path: str, max_chars: int = DEFAULT_READ_LIMIT) -> dict:
        """读取 VFS 文本文件，默认最多返回 20000 字符（超出截断）。"""
        data = await backend.read_file(path)
        text = data.decode("utf-8", errors="replace")
        content, truncated = _truncate_text(text, max_chars=max_chars)
        return {
            "ok": True,
            "workspace": prefix,
            "path": path,
            "content": content,
            "truncated": truncated,
            "size_chars": len(text),
        }

    # ------------------------------------------------------------------
    # append_file
    # ------------------------------------------------------------------

    async def vfs_append_file(path: str, content: str) -> dict:
        """向 VFS 文件末尾追加文本，文件不存在时创建。"""
        data = content.encode("utf-8")
        await backend.append_file(path, data)
        return {"ok": True, "workspace": prefix, "path": path, "bytes": len(data)}

    # ------------------------------------------------------------------
    # delete_file
    # ------------------------------------------------------------------

    async def vfs_delete_file(path: str) -> dict:
        """删除 VFS 中的单个文件（不删除目录）。"""
        await backend.delete_file(path)
        return {"ok": True, "workspace": prefix, "path": path, "deleted": True}

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------

    async def vfs_ls(path: str = ".") -> dict:
        """递归列出 VFS 目录下的所有文件（路径、大小、修改时间）。"""
        entries = await backend.ls(path)
        return {
            "ok": True,
            "workspace": prefix,
            "path": path,
            "files": [entry.model_dump() for entry in entries],
            "count": len(entries),
        }

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    async def vfs_grep(
        pattern: str,
        path: str = ".",
        regex: bool = False,
        max_results: int = DEFAULT_GREP_RESULTS,
    ) -> dict:
        """在 VFS 范围内按行搜索（子串或正则），返回匹配行与行号。"""
        matches, truncated = await backend.grep(
            pattern,
            path=path,
            regex=regex,
            max_results=max_results,
        )
        return {
            "ok": True,
            "workspace": prefix,
            "pattern": pattern,
            "matches": [match.model_dump() for match in matches],
            "count": len(matches),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # 组装 Tool
    # ------------------------------------------------------------------

    def _tool(
        name: str,
        description: str,
        display_name: str,
        handler: Any,
        parameters: ToolParameters,
        *,
        side_effect: bool = False,
    ) -> Tool:
        return Tool(
            name=name,
            description=f"{label} {description}",
            parameters=parameters,
            handler=handler,
            execution=ToolExecutionPolicy(side_effect=side_effect),
            display_name=display_name,
            is_concurrency_safe=not side_effect,
        )

    return [
        _tool(
            f"{prefix}_write_file",
            "写入文本文件（覆盖写，路径相对虚拟文件系统根）。",
            "VFS 写入文件",
            vfs_write_file,
            _params(
                {
                    "path": {"type": "string", "description": "相对 VFS 根的文件路径"},
                    "content": {"type": "string", "description": "要写入的文本内容"},
                    "overwrite": {
                        "type": "boolean",
                        "description": "文件已存在时是否覆盖，默认 true",
                    },
                },
                ["path", "content"],
            ),
            side_effect=True,
        ),
        _tool(
            f"{prefix}_read_file",
            f"读取文本文件，默认最多返回 {DEFAULT_READ_LIMIT} 字符。",
            "VFS 读取文件",
            vfs_read_file,
            _params(
                {
                    "path": {"type": "string", "description": "相对 VFS 根的文件路径"},
                    "max_chars": {
                        "type": "number",
                        "description": "最多返回的字符数，默认 20000",
                    },
                },
                ["path"],
            ),
        ),
        _tool(
            f"{prefix}_append_file",
            "向文件末尾追加文本，文件不存在时创建。",
            "VFS 追加文件",
            vfs_append_file,
            _params(
                {
                    "path": {"type": "string", "description": "相对 VFS 根的文件路径"},
                    "content": {"type": "string", "description": "要追加的文本内容"},
                },
                ["path", "content"],
            ),
            side_effect=True,
        ),
        _tool(
            f"{prefix}_delete_file",
            "删除单个文件（不删除目录）。",
            "VFS 删除文件",
            vfs_delete_file,
            _params(
                {
                    "path": {"type": "string", "description": "相对 VFS 根的文件路径"},
                },
                ["path"],
            ),
            side_effect=True,
        ),
        _tool(
            f"{prefix}_ls",
            "递归列出目录下的所有文件（路径、大小、修改时间）。",
            "VFS 列出文件",
            vfs_ls,
            _params(
                {
                    "path": {
                        "type": "string",
                        "description": "相对 VFS 根的目录路径，默认根目录",
                    },
                },
                [],
            ),
        ),
        _tool(
            f"{prefix}_grep",
            "按行搜索文件内容，支持子串与正则，返回匹配行和行号。",
            "VFS 搜索内容",
            vfs_grep,
            _params(
                {
                    "pattern": {
                        "type": "string",
                        "description": "搜索的子串或正则表达式",
                    },
                    "path": {
                        "type": "string",
                        "description": "搜索范围目录，默认根目录",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "是否按正则解析 pattern，默认 false",
                    },
                    "max_results": {
                        "type": "number",
                        "description": "最多返回的匹配条数，默认 100",
                    },
                },
                ["pattern"],
            ),
        ),
    ]


def make_vfs_plugin(backend: FSBackend, name: str = "vfs") -> Plugin:
    """把 VFS 后端包装成 Plugin，setup 时把六个工具注册到 ToolRegistry。

    :param backend: 实现 :class:`wuwei.fs.base.FSBackend` 协议的后端
    :param name: 插件名，同时作为工具名前缀（workspace_label）
    """
    tools = make_vfs_tools(backend, workspace_label=name)

    def setup(ctx: PluginContext) -> None:
        """把 VFS 工具注册到插件上下文的 ToolRegistry。"""
        for tool in tools:
            ctx.tool_registry.register(tool)

    return Plugin(
        name=name,
        version="1.0.0",
        description=f"虚拟文件系统工具集（工作区: {name}）",
        tags=["fs", "vfs", "tools"],
        _setup_fn=setup,
    )
