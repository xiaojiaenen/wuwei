"""虚拟文件系统（VFS）— 统一的文件后端抽象与配套工具。

- :class:`FSBackend` / :class:`FSFile` / :class:`GrepMatch` — 协议与数据模型
- :class:`InMemoryFS` — 内存后端（二进制安全，适合测试与沙箱）
- :class:`LocalFS` — 锚定真实目录的后端（含路径逃逸防护）
- :func:`make_vfs_tools` / :func:`make_vfs_plugin` — 把后端暴露为 wuwei 工具/插件
"""

from wuwei.fs.base import FSBackend, FSFile, GrepMatch, in_scope, normalize_vfs_path
from wuwei.fs.local import LocalFS
from wuwei.fs.memory import InMemoryFS
from wuwei.fs.tools import (
    DEFAULT_GREP_RESULTS,
    DEFAULT_READ_LIMIT,
    make_vfs_plugin,
    make_vfs_tools,
)

__all__ = [
    "DEFAULT_GREP_RESULTS",
    "DEFAULT_READ_LIMIT",
    "FSBackend",
    "FSFile",
    "GrepMatch",
    "InMemoryFS",
    "LocalFS",
    "in_scope",
    "make_vfs_plugin",
    "make_vfs_tools",
    "normalize_vfs_path",
]
