from pathlib import Path

from markitdown import MarkItDown

from wuwei.tools.registry import ToolRegistry

DEFAULT_READ_LIMIT = 20_000


def _resolve_workspace_path(path: str, *, workspace: str = ".") -> Path:
    root = Path(workspace).resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"路径必须位于 workspace 内: {path}")
    return target


def _truncate_text(text: str, *, max_chars: int = DEFAULT_READ_LIMIT) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True

def _collect_files(
    root: Path,
    max_depth: int,
    max_files: int,
    workspace_root: Path,
) -> tuple[list[dict[str, str | int]], bool]:
    """
    递归收集目录内容，返回 (结果列表, 是否截断)。

    :param root: 当前遍历的绝对路径目录
    :param max_depth: 剩余递归深度，0 表示不再进入子目录
    :param max_files: 最多收集的文件/目录数
    :param workspace_root: workspace 根目录，用于计算相对路径
    """
    entries: list[dict[str, str | int]] = []
    try:
        items = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return entries, False

    for item in items:
        if len(entries) >= max_files:
            return entries, True

        rel_path = str(item.relative_to(workspace_root))
        if item.is_dir():
            entries.append({
                "path": rel_path,
                "type": "dir",
                "size_bytes": 0,
            })
            if max_depth > 0:
                sub_entries, truncated = _collect_files(
                    item,
                    max_depth - 1,
                    max_files - len(entries),
                    workspace_root,
                )
                entries.extend(sub_entries)
                if truncated:
                    return entries, True
        else:
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            entries.append({
                "path": rel_path,
                "type": "file",
                "size_bytes": size,
            })

    return entries, False


def register_file_tools(registry: ToolRegistry) -> None:
    @registry.tool(name="file_to_md", description="将文件转换为 markdown供大模型阅读，支持常见文本文件、csv、pdf、xlsx、docx 等。")
    def file_to_md(path: str):
        try:
            md_converter = MarkItDown()
            result = md_converter.convert(path)
        except Exception as e:
            return str(e)
        if result is None or result.text_content.strip() == "":
            return "转换失败，该文件无法转换"
        return result.text_content

    @registry.tool(
        name="read_text_file",
        description="读取 workspace 内的文本文件内容。默认最多返回 20000 字符。",
    )
    def read_text_file(path: str, max_chars: int = DEFAULT_READ_LIMIT, workspace: str = ".") -> dict:
        """读取文本文件。

        :param path: 相对 workspace 的文件路径
        :param max_chars: 最多返回字符数，避免超长文件撑爆上下文
        :param workspace: 工作区根目录，默认当前目录
        """
        target = _resolve_workspace_path(path, workspace=workspace)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        text = target.read_text(encoding="utf-8")
        content, truncated = _truncate_text(text, max_chars=max_chars)
        return {
            "ok": True,
            "path": str(target),
            "content": content,
            "truncated": truncated,
            "size_chars": len(text),
        }

    @registry.tool(
        name="write_text_file",
        description="写入 workspace 内的文本文件。默认不覆盖已有文件，overwrite=true 时才覆盖。",
    )
    def write_text_file(
        path: str,
        content: str,
        overwrite: bool = False,
        workspace: str = ".",
    ) -> dict:
        """写入文本文件。

        :param path: 相对 workspace 的文件路径
        :param content: 要写入的文本内容
        :param overwrite: 是否覆盖已有文件
        :param workspace: 工作区根目录，默认当前目录
        """
        target = _resolve_workspace_path(path, workspace=workspace)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"目标是目录，不能写入文件: {path}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"文件已存在，设置 overwrite=true 才能覆盖: {path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(target), "bytes": len(content.encode("utf-8"))}

    @registry.tool(
        name="append_text_file",
        description="向 workspace 内的文本文件末尾追加内容，文件不存在时会创建。",
    )
    def append_text_file(path: str, content: str, workspace: str = ".") -> dict:
        """追加文本文件。

        :param path: 相对 workspace 的文件路径
        :param content: 要追加的文本内容
        :param workspace: 工作区根目录，默认当前目录
        """
        target = _resolve_workspace_path(path, workspace=workspace)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"目标是目录，不能追加文件: {path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.write(content)
        return {"ok": True, "path": str(target), "bytes": len(content.encode("utf-8"))}

    @registry.tool(
        name="replace_text_in_file",
        description="修改 workspace 内文本文件：把 old_text 替换为 new_text。",
    )
    def replace_text_in_file(
        path: str,
        old_text: str,
        new_text: str,
        count: int = -1,
        workspace: str = ".",
    ) -> dict:
        """替换文件中的文本。

        :param path: 相对 workspace 的文件路径
        :param old_text: 要查找的旧文本
        :param new_text: 替换后的新文本
        :param count: 最大替换次数，-1 表示全部替换
        :param workspace: 工作区根目录，默认当前目录
        """
        if not old_text:
            raise ValueError("old_text 不能为空")

        target = _resolve_workspace_path(path, workspace=workspace)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        text = target.read_text(encoding="utf-8")
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise ValueError("未找到 old_text，文件未修改")

        max_replace = count if count is not None and count >= 0 else occurrences
        updated = text.replace(old_text, new_text, max_replace)
        target.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": str(target),
            "replacements": min(occurrences, max_replace),
        }

    @registry.tool(
        name="delete_file",
        description="删除 workspace 内的单个文件。只删除文件，不删除目录。",
    )
    def delete_file(path: str, workspace: str = ".") -> dict:
        """删除文件。

        :param path: 相对 workspace 的文件路径
        :param workspace: 工作区根目录，默认当前目录
        """
        target = _resolve_workspace_path(path, workspace=workspace)
        if not target.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"只允许删除文件，不允许删除目录: {path}")

        target.unlink()
        return {"ok": True, "path": str(target), "deleted": True}






    @registry.tool(
        name="list_files",
        description="列出 workspace 内指定目录下的文件和子目录（默认递归 3 层，最多返回 200 项）。",
    )
    def list_files(
        path: str = ".",
        max_depth: int = 3,
        max_files: int = 200,
        workspace: str = ".",
    ) -> dict:
        """列出目录内容。

        :param path: 相对 workspace 的目录路径，默认根目录
        :param max_depth: 最大递归深度，0 表示仅当前目录，默认 3
        :param max_files: 最多返回的文件/目录数，默认 200
        :param workspace: 工作区根目录，默认当前目录
        :return: 包含 ok, path, files, truncated 等字段的结果字典
        """
        target = _resolve_workspace_path(path, workspace=workspace)
        if not target.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        if not target.is_dir():
            raise NotADirectoryError(f"路径不是目录: {path}")

        workspace_root = Path(workspace).resolve()
        entries, truncated = _collect_files(
            target,
            max_depth,
            max_files,
            workspace_root,
        )

        return {
            "ok": True,
            "path": str(target.relative_to(workspace_root)),
            "files": entries,
            "count": len(entries),
            "truncated": truncated,
        }