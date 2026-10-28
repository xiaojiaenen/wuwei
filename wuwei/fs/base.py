"""VFS 基础定义 — 后端协议、文件元数据与路径规范化。

虚拟文件系统（VFS）把「文件存储」抽象成一个统一的后端协议，
上层工具（见 wuwei.fs.tools）只面向 FSBackend 编程，
底层可以是内存（InMemoryFS）或真实目录（LocalFS）。

约定：
- 所有路径均为相对 VFS 根的 POSIX 风格路径（以 ``/`` 分隔）；
- 拒绝绝对路径与包含 ``..`` 的路径（防止路径逃逸）；
- 文件内容一律是二进制安全的 ``bytes``，编码由调用方决定。
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:")


class FSFile(BaseModel):
    """VFS 中的文件元数据。

    :param path: 相对 VFS 根的 POSIX 风格路径
    :param size: 文件字节数
    :param updated_at: 最后修改时间（UNIX 时间戳，秒）
    """

    path: str
    size: int = 0
    updated_at: float = 0.0


class GrepMatch(BaseModel):
    """grep 的一条匹配结果。

    :param path: 命中文件路径
    :param line: 命中行号（从 1 开始）
    :param content: 命中行内容（可能被截断）
    """

    path: str
    line: int = Field(ge=1)
    content: str = ""


@runtime_checkable
class FSBackend(Protocol):
    """VFS 后端协议 — 所有路径均为相对 VFS 根的 POSIX 风格路径。

    实现方必须：
    - 对非法路径（绝对路径、含 ``..``）抛出 :class:`ValueError`；
    - 对不存在的文件抛出 :class:`FileNotFoundError`；
    - 内容以 ``bytes`` 存储，保证二进制安全。
    """

    async def write_file(self, path: str, content: bytes) -> None:
        """写入文件（覆盖已有内容），必要时自动创建父目录。"""
        ...

    async def read_file(self, path: str) -> bytes:
        """读取文件全部内容。"""
        ...

    async def append_file(self, path: str, content: bytes) -> None:
        """向文件末尾追加内容，文件不存在时创建。"""
        ...

    async def delete_file(self, path: str) -> None:
        """删除单个文件（不删除目录）。"""
        ...

    async def ls(self, path: str = ".") -> list[FSFile]:
        """递归列出路径（默认根目录）下的所有文件元数据。"""
        ...

    async def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        regex: bool = False,
        max_results: int = 200,
    ) -> tuple[list[GrepMatch], bool]:
        """在路径范围内按行搜索，返回 (匹配列表, 是否因 max_results 截断)。"""
        ...

    async def glob(self, pattern: str) -> list[str]:
        """按 glob 模式匹配文件路径，返回排序后的路径列表。"""
        ...


def normalize_vfs_path(path: str) -> str:
    """校验并规范化 VFS 路径，返回相对根的 POSIX 风格字符串。

    规则：
    - 空路径、``.``、``./`` 一律规范化为 ``""``（代表根目录）；
    - 拒绝绝对路径（以 ``/`` 开头或 Windows 盘符开头）；
    - 拒绝任何一级为 ``..`` 的路径（路径逃逸）；
    - 去除多余的 ``/`` 与 ``.`` 分量。

    :raises ValueError: 路径为空、是绝对路径或包含 ``..``
    :return: 规范化后的相对路径，根目录为 ``""``
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("路径不能为空")
    if "\x00" in path:
        raise ValueError(f"路径包含非法字符 NUL: {path!r}")

    raw = path.strip()
    if raw.startswith("/") or _DRIVE_LETTER_RE.match(raw):
        raise ValueError(f"拒绝绝对路径: {path}")

    pure = PurePosixPath(raw)
    parts: list[str] = []
    for part in pure.parts:
        if part == "..":
            raise ValueError(f"拒绝路径逃逸（..）: {path}")
        if part in ("", "."):
            continue
        parts.append(part)

    return "/".join(parts)


def in_scope(path: str, scope: str) -> bool:
    """判断规范化路径 ``path`` 是否位于规范化目录 ``scope`` 之下（含相等）。"""
    if scope == "":
        return True
    return path == scope or path.startswith(scope + "/")
