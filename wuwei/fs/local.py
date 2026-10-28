"""本地目录虚拟文件系统 — 锚定 root 的真实目录实现。

实现 :class:`wuwei.fs.base.FSBackend` 协议：
- 所有路径先经 :func:`normalize_vfs_path` 校验（拒绝绝对路径 / ``..``），
  再 resolve 并确认落在 root 之内（双重防护，含符号链接逃逸）；
- 阻塞 IO 通过 ``asyncio.to_thread`` 包装，保持 async-first。
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path

from wuwei.fs.base import FSFile, GrepMatch, in_scope, normalize_vfs_path

_DEFAULT_GREP_LINE_LIMIT = 500


class LocalFS:
    """锚定 ``root`` 目录的本地文件系统后端。

    :param root: 作为 VFS 根的真实目录，不存在时自动创建。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """校验路径并解析为 root 内的真实绝对路径。

        :raises ValueError: 路径非法或解析后逃逸出 root（含符号链接）
        :raises FileNotFoundError: 当作为文件访问时可由调用方另行抛出
        """
        if not isinstance(path, str) or path.strip() in ("", "."):
            return self.root  # 空路径 / "." 即根目录
        normalized = normalize_vfs_path(path)
        target = (self.root / normalized).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"路径必须位于 VFS 根内: {path}")
        return target

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

    @staticmethod
    def _compile_pattern(pattern: str) -> re.Pattern[str]:
        """编译正则，非法表达式统一抛 ValueError。"""
        try:
            return re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"非法正则表达式: {pattern!r}") from exc

    # ------------------------------------------------------------------
    # FSBackend 协议实现
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: bytes) -> None:
        """写入文件（覆盖），必要时自动创建父目录。"""
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content 必须是 bytes")
        target = self._resolve(path)
        if target == self.root or target.is_dir():
            raise IsADirectoryError(f"目标是目录，不能写入文件: {path}")

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(content))

        await asyncio.to_thread(_write)

    async def read_file(self, path: str) -> bytes:
        """读取文件全部内容。"""
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if target.is_dir():
            raise IsADirectoryError(f"目标是目录，不能作为文件读取: {path}")
        return await asyncio.to_thread(target.read_bytes)

    async def append_file(self, path: str, content: bytes) -> None:
        """向文件末尾追加内容，文件不存在时创建。"""
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content 必须是 bytes")
        target = self._resolve(path)
        if target == self.root or target.is_dir():
            raise IsADirectoryError(f"目标是目录，不能追加文件: {path}")

        def _append() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("ab") as file:
                file.write(bytes(content))

        await asyncio.to_thread(_append)

    async def delete_file(self, path: str) -> None:
        """删除单个文件（不删除目录），不存在时抛 FileNotFoundError。"""
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if target.is_dir():
            raise IsADirectoryError(f"只允许删除文件，不允许删除目录: {path}")
        await asyncio.to_thread(target.unlink)

    async def ls(self, path: str = ".") -> list[FSFile]:
        """递归列出 scope 下的所有文件元数据，按路径排序。"""
        scope = normalize_vfs_path(path)
        base = self._resolve(scope)
        if not base.exists():
            raise FileNotFoundError(f"路径不存在: {path}")

        def _collect() -> list[FSFile]:
            entries: list[FSFile] = []
            if base.is_file():
                stat = base.stat()
                return [
                    FSFile(path=scope, size=stat.st_size, updated_at=stat.st_mtime)
                ]
            for item in sorted(base.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(self.root).as_posix()
                if not in_scope(rel, scope):
                    continue
                try:
                    stat = item.stat()
                except OSError:
                    continue
                entries.append(
                    FSFile(path=rel, size=stat.st_size, updated_at=stat.st_mtime)
                )
            return entries

        return await asyncio.to_thread(_collect)

    async def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        regex: bool = False,
        max_results: int = 200,
    ) -> tuple[list[GrepMatch], bool]:
        """按行搜索 scope 内的文本文件，二进制内容按 ``errors="replace"`` 解码。

        :param pattern: 子串（regex=False）或正则表达式（regex=True）
        :param path: 搜索范围，默认根目录
        :param regex: 是否按正则解析 pattern
        :param max_results: 最多返回的匹配条数
        :return: (匹配列表, 是否截断)
        """
        if regex:
            self._compile_pattern(pattern)
        files = await self.ls(path)
        matches: list[GrepMatch] = []
        truncated = False
        for meta in files:
            if len(matches) >= max_results:
                truncated = True
                break
            data = await self.read_file(meta.path)
            text = data.decode("utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if self._match_line(line, pattern, regex):
                    if len(matches) >= max_results:
                        truncated = True
                        break
                    matches.append(
                        GrepMatch(
                            path=meta.path,
                            line=lineno,
                            content=self._truncate_line(line),
                        )
                    )
        return matches, truncated

    async def glob(self, pattern: str) -> list[str]:
        """按 glob 模式匹配文件路径（``*`` 也会跨 ``/`` 匹配）。"""
        normalize_vfs_path(
            pattern.replace("**", "x")
        )  # 复用路径校验，拒绝绝对路径 / ..
        root = self.root

        def _glob() -> list[str]:
            results: list[str] = []
            for item in root.rglob("*"):
                if not item.is_file():
                    continue
                rel = item.relative_to(root).as_posix()
                if fnmatch.fnmatch(rel, pattern):
                    results.append(rel)
            return sorted(results)

        return await asyncio.to_thread(_glob)
