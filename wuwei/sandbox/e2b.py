"""E2B 云沙箱驱动器

使用 `e2b <https://e2b.dev>`_ 云端沙箱执行代码 / 命令。e2b SDK 为
可选依赖：缺失时抛出带安装提示的 ``ImportError``。

SDK 调用为同步 API，统一用 ``asyncio.to_thread`` 包装，避免阻塞事件循环；
Sandbox 实例懒创建并复用，``cleanup()`` / ``close()`` 可显式销毁。
"""

import asyncio
import contextlib
import os
import time

from wuwei.sandbox.base import SandboxResult, SandboxUnavailable

_INSTALL_HINT = (
    "e2b SDK 未安装。请安装：pip install 'e2b' "
    "（并设置 E2B_API_KEY 环境变量，见 https://e2b.dev/docs）"
)


def _to_text(value) -> str:
    """把 SDK 返回的 stdout/stderr（str 或 list[str]）统一为 str。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(str(v) for v in value)
    return str(value)


class E2BDriver:
    """E2B 云沙箱驱动器。

    Args:
        api_key: E2B API key；缺省时回退到环境变量 ``E2B_API_KEY``。
        timeout: 云沙箱会话保活时长（秒），默认 300。
    """

    def __init__(self, api_key: str = None, timeout: float = 300):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.timeout = timeout
        self._sandbox = None
        self._sdk = None

    # ---- SDK 懒加载 ----

    def _load_sdk(self):
        """懒加载 e2b SDK（缺失抛带安装提示的 ImportError）。"""
        if self._sdk is None:
            try:
                import e2b  # noqa: F401
            except ImportError as e:
                raise ImportError(_INSTALL_HINT) from e
            self._sdk = e2b
        return self._sdk

    def _resolve_api_key(self) -> str:
        if not self.api_key:
            raise SandboxUnavailable(
                "缺少 E2B API key：请传入 api_key=... 或设置 E2B_API_KEY 环境变量。"
            )
        return self.api_key

    async def _get_sandbox(self):
        """懒创建并缓存 Sandbox 实例（在线程池中执行同步构造）。"""
        if self._sandbox is None:
            sdk = self._load_sdk()
            api_key = self._resolve_api_key()

            def _create():
                return sdk.Sandbox(api_key=api_key, timeout=self.timeout)

            try:
                self._sandbox = await asyncio.to_thread(_create)
            except (ImportError, SandboxUnavailable):
                raise
            except Exception as e:
                raise SandboxUnavailable(f"创建 E2B 沙箱失败: {e}") from e
        return self._sandbox

    # ---- SandboxDriver 协议 ----

    async def run_code(
        self,
        code: str,
        *,
        timeout: float = 30,
        files: dict[str, bytes] | None = None,
    ) -> SandboxResult:
        """在 E2B 云沙箱中执行 Python 代码（files 先写入沙箱文件系统）。"""
        start = time.monotonic()
        sandbox = await self._get_sandbox()

        if files:
            def _write_files():
                for path, content in files.items():
                    sandbox.files.write(path, content)

            try:
                await asyncio.to_thread(_write_files)
            except Exception as e:
                return SandboxResult(
                    success=False,
                    error=f"写入沙箱文件失败: {e}",
                    exit_code=-1,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

        try:
            result = await asyncio.to_thread(
                sandbox.run_code, code, timeout=int(timeout)
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"E2B 执行失败: {e}",
                exit_code=-1,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # ExecutionResult 字段随 SDK 版本略有差异，统一防御性解析
        logs = getattr(result, "logs", None)
        stdout_text = _to_text(getattr(logs, "stdout", None) if logs else None)
        stderr_text = _to_text(getattr(logs, "stderr", None) if logs else None)
        if not stdout_text:
            stdout_text = _to_text(getattr(result, "stdout", None))
        if not stderr_text:
            stderr_text = _to_text(getattr(result, "stderr", None))

        error = getattr(result, "error", None)
        exit_code = getattr(result, "exit_code", None)
        if exit_code is None:
            exit_code = 0 if error is None else 1
        if error is not None and not stderr_text:
            stderr_text = str(error)

        return SandboxResult(
            success=error is None and exit_code == 0,
            output=stdout_text,
            error=stderr_text,
            exit_code=int(exit_code),
            duration_ms=(time.monotonic() - start) * 1000,
        )

    async def run_command(self, command: str, *, timeout: float = 30) -> SandboxResult:
        """在 E2B 云沙箱中执行 shell 命令。"""
        start = time.monotonic()
        sandbox = await self._get_sandbox()
        try:
            result = await asyncio.to_thread(
                sandbox.commands.run, command, timeout=int(timeout)
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"E2B 命令执行失败: {e}",
                exit_code=-1,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        exit_code = getattr(result, "exit_code", None)
        return SandboxResult(
            success=exit_code == 0,
            output=_to_text(getattr(result, "stdout", None)),
            error=_to_text(getattr(result, "stderr", None)),
            exit_code=int(exit_code) if exit_code is not None else 0,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    # ---- 生命周期 ----

    async def cleanup(self):
        """关闭并销毁云沙箱实例。"""
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - 清理失败不影响主流程
                await asyncio.to_thread(sandbox.kill)

    # SDK 新版叫 close()，旧版叫 close() 也可能不存在；提供两个别名
    async def close(self):
        await self.cleanup()

    async def __aenter__(self) -> "E2BDriver":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
