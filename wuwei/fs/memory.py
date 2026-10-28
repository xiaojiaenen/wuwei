"""内存虚拟文件系统 — dict 存储，二进制安全。

适用于测试、沙箱 agent、临时工作区等不需要落盘的场景。
实现 :class:`wuwei.fs.base.FSBackend` 协议。
"""

from __future__ import annotations

import fnmatch
import re
import time
from dataclasses import dataclass, field

from wuwei.fs.base import FSFile, GrepMatch, in_scope, normalize_vfs_path

_DEFAULT_GREP_LINE_LIMIT = 500


@dataclass
class _Entry:
    """内存中的一个文件条目。"""

    data: bytes
    updated_at: float = field(default_factory=time.time)


class InMemoryFS:
    """基于 dict 的内存文件系统。

    - 内容以 ``bytes`` 存储，天然二进制安全；
    - 所有路径经 :func:`normalize_vfs_path` 校验，拒绝绝对路径与 ``..``；
    - 没有真实的目录对象，目录由路径前缀隐式表达。
    """

    def __init__(self) -> None:
        self._files: dict[str, _Entry] = {}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _require_file(self, path: str) -> tuple[str, _Entry]:
        normalized = normalize_vfs_path(path)
        entry = self._files.get(normalized)
        if entry is None:
            raise FileNotFoundError(f"文件不存在: {path}")
        return normalized, entry

    def _scoped_paths(self, path: str) -> list[str]:
        """返回 scope 目录（或单个文件）下的所有文件路径，已排序。"""
        scope = normalize_vfs_path(path)
        return sorted(p for p in self._files if in_scope(p, scope))

    @staticmethod
    def _match_line(line: str, pattern: str, regex: bool) -> bool:
        if regex:
            return re.search(pattern, line) is not None
        return pattern in line

    @staticmethod
    def _truncate_line(line: str, limit: int = _DEFAULT_GREP_LINE_LIMIT) -> str:
        if limit <= 0 or len(line) <= limit:
            return line
        return line[:limit]

    # ------------------------------------------------------------------
    # FSBackend 协议实现
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: bytes) -> None:
        """写入文件（覆盖），内容必须为 ``bytes``。"""
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content 必须是 bytes")
        normalized = normalize_vfs_path(path)
        if not normalized:
            raise ValueError("不能写根目录")
        self._files[normalized] = _Entry(data=bytes(content), updated_at=time.time())

    async def read_file(self, path: str) -> bytes:
        """读取文件全部内容。"""
        _, entry = self._require_file(path)
        return entry.data

    async def append_file(self, path: str, content: bytes) -> None:
        """向文件末尾追加内容，文件不存在时创建。"""
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content 必须是 bytes")
        normalized = normalize_vfs_path(path)
        if not normalized:
            raise ValueError("不能写根目录")
        payload = bytes(content)
        entry = self._files.get(normalized)
        if entry is None:
            self._files[normalized] = _Entry(data=payload, updated_at=time.time())
        else:
            entry.data += payload
            entry.updated_at = time.time()

    async def delete_file(self, path: str) -> None:
        """删除单个文件，不存在时抛 FileNotFoundError。"""
        normalized, _ = self._require_file(path)
        del self._files[normalized]

    async def ls(self, path: str = ".") -> list[FSFile]:
        """递归列出 scope 下的所有文件（根目录时列出全部）。"""
        return [
            FSFile(
                path=p,
                size=len(self._files[p].data),
                updated_at=self._files[p].updated_at,
            )
            for p in self._scoped_paths(path)
        ]

    async def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        regex: bool = False,
        max_results: int = 200,
    ) -> tuple[list[GrepMatch], bool]:
        """按行搜索 scope 内的文本文件。

        :param pattern: 子串（regex=False）或正则表达式（regex=True）
        :param path: 搜索范围，默认根目录
        :param regex: 是否按正则解析 pattern（非法正则抛 ValueError）
        :param max_results: 最多返回的匹配条数
        :return: (匹配列表, 是否截断)
        """
        if regex:
            try:
                re.compile(pattern)  # 提前校验，非法正则抛 ValueError
            except re.error as exc:
                raise ValueError(f"非法正则表达式: {pattern!r}") from exc
        matches: list[GrepMatch] = []
        truncated = False
        for file_path in self._scoped_paths(path):
            text = self._files[file_path].data.decode("utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if self._match_line(line, pattern, regex):
                    if len(matches) >= max_results:
                        truncated = True
                        return matches, truncated
                    matches.append(
                        GrepMatch(
                            path=file_path,
                            line=lineno,
                            content=self._truncate_line(line),
                        )
                    )
        return matches, truncated

    async def glob(self, pattern: str) -> list[str]:
        """按 glob 模式匹配文件路径（``*`` 也会跨 ``/`` 匹配）。"""
        return sorted(p for p in self._files if fnmatch.fnmatch(p, pattern))
