"""沙箱执行模块

驱动器协议与结果类型在 ``base``，实现有 local / docker / e2b，
工厂函数 :func:`get_sandbox` 按名称构造。
"""

from wuwei.sandbox.base import (
    BaseSandbox,
    SandboxDriver,
    SandboxResult,
    SandboxUnavailable,
    run_shell,
)
from wuwei.sandbox.docker import DockerDriver
from wuwei.sandbox.e2b import E2BDriver
from wuwei.sandbox.factory import get_sandbox
from wuwei.sandbox.local import LocalPythonDriver, LocalSandbox

__all__ = [
    # 协议 / 类型
    "BaseSandbox",
    "SandboxDriver",
    "SandboxResult",
    "SandboxUnavailable",
    "run_shell",
    # 驱动器
    "LocalSandbox",
    "LocalPythonDriver",
    "DockerDriver",
    "E2BDriver",
    # 工厂
    "get_sandbox",
]
