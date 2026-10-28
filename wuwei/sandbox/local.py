"""本地沙箱与本地 Python 沙箱驱动器

- ``LocalSandbox``：旧版 shell 命令沙箱（接口向后兼容：``execute`` /
  ``run`` / ``run_command``）。
- ``LocalPythonDriver``：实现 :class:`~wuwei.sandbox.base.SandboxDriver`
  协议，用 ``python -I``（隔离模式）子进程执行代码，支持 files 注入、
  超时 kill 和可选资源限制（RLIMIT_CPU / RLIMIT_AS，仅 Unix）。
"""

import asyncio
import contextlib
import os
import sys
import tempfile
import time

from wuwei.sandbox.base import BaseSandbox, SandboxResult, run_shell


class LocalSandbox(BaseSandbox):
    """本地沙箱

    在本地环境中执行 shell 命令。

    示例：
        async with LocalSandbox() as sandbox:
            result = await sandbox.execute("ls -la")
            print(result.stdout)
    """

    def __init__(self, workspace: str = None):
        """
        Args:
            workspace: 工作目录，默认当前目录
        """
        self.workspace = workspace or os.getcwd()

    async def execute(
        self,
        command: str,
        timeout: float = 30,
        cwd: str = None,
    ) -> SandboxResult:
        """执行命令（旧版接口）"""
        return await run_shell(command, timeout=timeout, cwd=cwd or self.workspace)

    async def run_command(self, command: str, *, timeout: float = 30) -> SandboxResult:
        """执行命令（SandboxDriver 协议）"""
        return await run_shell(command, timeout=timeout, cwd=self.workspace)

    async def run(self, command: str, **kwargs) -> SandboxResult:
        """执行命令（简写别名）"""
        timeout = kwargs.pop("timeout", 30)
        cwd = kwargs.pop("cwd", None)
        return await run_shell(command, timeout=timeout, cwd=cwd or self.workspace)

    async def cleanup(self):
        """清理资源"""
        pass


class LocalPythonDriver:
    """本地 Python 隔离执行驱动器。

    用 ``python -I``（isolated mode：忽略环境变量、不加载用户 site、
    不把当前目录加入 sys.path）子进程执行代码，代码经 stdin 传入。

    Args:
        python_executable: Python 解释器路径，默认 ``sys.executable``。
        cpu_seconds: 可选，子进程 CPU 时间上限（秒，RLIMIT_CPU，仅 Unix）。
        memory_bytes: 可选，子进程地址空间上限（字节，RLIMIT_AS，仅 Unix）。
    """

    def __init__(
        self,
        python_executable: str = None,
        cpu_seconds: int | None = None,
        memory_bytes: int | None = None,
    ):
        self.python_executable = python_executable or sys.executable
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes

    # ---- 内部工具 ----

    def _make_preexec_fn(self):
        """构造子进程资源限制回调（仅 Unix；不可用则返回 None）。"""
        if os.name != "posix" or (self.cpu_seconds is None and self.memory_bytes is None):
            return None

        cpu = self.cpu_seconds
        mem = self.memory_bytes

        def preexec():  # pragma: no cover - 在子进程中执行
            import resource

            # 资源限制是尽力而为：部分平台/场景下 setrlimit 可能失败
            # （如 macOS 在 asyncio fork 后 RLIMIT_RSS 会报错），忽略即可。
            if cpu is not None:
                with contextlib.suppress(OSError, ValueError):
                    cpu_limit = getattr(resource, "RLIMIT_CPU", None)
                    if cpu_limit is not None:
                        limit = max(int(cpu), 1)
                        resource.setrlimit(cpu_limit, (limit, limit + 1))
            if mem is not None:
                with contextlib.suppress(OSError, ValueError):
                    # macOS 没有 RLIMIT_AS，只有 RLIMIT_RSS（近似）
                    mem_limit = getattr(
                        resource, "RLIMIT_AS", getattr(resource, "RLIMIT_RSS", None)
                    )
                    if mem_limit is not None:
                        resource.setrlimit(mem_limit, (int(mem), int(mem)))

        return preexec

    async def _communicate(
        self,
        argv: list,
        stdin_data: bytes,
        *,
        timeout: float,
        cwd: str = None,
    ) -> SandboxResult:
        """启动子进程、通信、处理超时 kill，返回统一结果。"""
        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                preexec_fn=self._make_preexec_fn(),
            )
        except OSError as e:
            return SandboxResult(
                success=False,
                error=f"无法启动子进程: {e}",
                exit_code=-1,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_data), timeout=timeout
            )
        except TimeoutError:
            # 超时：杀掉整个进程
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):  # pragma: no cover
                await process.wait()
            return SandboxResult(
                success=False,
                error=f"代码执行超时 ({timeout}s)，进程已被终止",
                exit_code=-1,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        exit_code = process.returncode if process.returncode is not None else -1
        return SandboxResult(
            success=exit_code == 0,
            output=stdout.decode(errors="replace"),
            error=stderr.decode(errors="replace"),
            exit_code=exit_code,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    # ---- SandboxDriver 协议 ----

    async def run_code(
        self,
        code: str,
        *,
        timeout: float = 30,
        files: dict[str, bytes] | None = None,
    ) -> SandboxResult:
        """用 ``python -I`` 子进程执行 Python 代码。

        ``files`` 会被写入一个一次性临时目录并作为工作目录（用完即删）。
        """
        workdir = None
        tmp = None
        try:
            if files:
                tmp = tempfile.TemporaryDirectory(prefix="wuwei_sandbox_")
                workdir = tmp.name
                for path, content in files.items():
                    full = os.path.join(workdir, path)
                    os.makedirs(os.path.dirname(full) or workdir, exist_ok=True)
                    with open(full, "wb") as f:
                        f.write(content)

            return await self._communicate(
                [self.python_executable, "-I", "-"],
                code.encode("utf-8"),
                timeout=timeout,
                cwd=workdir,
            )
        finally:
            if tmp is not None:
                tmp.cleanup()

    async def run_command(self, command: str, *, timeout: float = 30) -> SandboxResult:
        """执行 shell 命令（本地直接执行，无额外隔离）。"""
        return await run_shell(command, timeout=timeout)
