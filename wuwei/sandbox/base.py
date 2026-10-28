"""沙箱基类与公共类型

定义所有沙箱驱动器共享的：

- ``SandboxResult``：统一执行结果（pydantic 模型）。
- ``SandboxDriver``：驱动器 Protocol（run_code / run_command）。
- ``BaseSandbox``：旧版抽象基类（execute/cleanup），向后兼容保留。
- ``SandboxUnavailable``：沙箱环境不可用（无 docker、缺 SDK 等）时抛出。
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator


class SandboxUnavailable(Exception):  # noqa: N818 - 保持对外 API 名称稳定
    """沙箱环境不可用。

    例如：本机没有 docker CLI / 守护进程、e2b SDK 未安装、API key 缺失等。
    """


class SandboxResult(BaseModel):
    """沙箱执行结果。

    主字段：

    - ``success``：退出码是否为 0
    - ``output``：stdout 文本
    - ``error``：stderr 文本（或超时/异常说明）
    - ``exit_code``：进程退出码（超时/无法获取时为 -1）
    - ``duration_ms``：执行耗时（毫秒）

    兼容别名：旧代码使用 ``stdout`` / ``stderr`` 字段名，构造和读取均保留支持
    （``result.stdout`` 等价于 ``result.output``）。
    """

    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    duration_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("output", "error", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        if value is None:
            return ""
        return value

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_fields(cls, data: Any) -> Any:
        """兼容旧版 dataclass 字段名 stdout/stderr。"""
        if isinstance(data, dict):
            if "stdout" in data and "output" not in data:
                data = {**data, "output": data.pop("stdout")}
            if "stderr" in data and "error" not in data:
                data = {**data, "error": data.pop("stderr")}
        return data

    # ---- 旧版兼容别名 ----

    @property
    def stdout(self) -> str:
        return self.output

    @property
    def stderr(self) -> str:
        return self.error

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error (exit code {self.exit_code}):\n{self.error}"


@runtime_checkable
class SandboxDriver(Protocol):
    """沙箱驱动器协议。

    所有驱动器（本地 Python、docker、e2b…）实现同一接口：

    - ``run_code``：在沙箱内用 Python 执行一段代码
    - ``run_command``：在沙箱内执行一条 shell 命令
    """

    async def run_code(
        self,
        code: str,
        *,
        timeout: float = 30,
        files: dict[str, bytes] | None = None,
    ) -> SandboxResult:
        """在沙箱内执行 Python 代码。

        Args:
            code: 要执行的 Python 源码。
            timeout: 超时秒数，超时后进程被 kill。
            files: 注入沙箱的文件（相对路径 -> 内容）。
        """
        ...

    async def run_command(self, command: str, *, timeout: float = 30) -> SandboxResult:
        """在沙箱内执行 shell 命令。"""
        ...


class BaseSandbox(ABC):
    """旧版沙箱基类（向后兼容保留）。

    新代码请实现/依赖 :class:`SandboxDriver` 协议。
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: float = 30,
        cwd: str = None,
    ) -> SandboxResult:
        """执行命令（旧版接口）"""
        ...

    @abstractmethod
    async def cleanup(self):
        """清理资源"""
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# 供各驱动器复用的统一执行入口
async def run_shell(
    command: str,
    *,
    timeout: float = 30,
    cwd: str = None,
) -> SandboxResult:
    """用本地 shell 执行命令并返回 :class:`SandboxResult`。"""
    import asyncio
    import time

    start = time.monotonic()
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return SandboxResult(
            success=process.returncode == 0,
            output=stdout.decode(errors="replace"),
            error=stderr.decode(errors="replace"),
            exit_code=process.returncode if process.returncode is not None else -1,
            duration_ms=(time.monotonic() - start) * 1000,
        )
    except TimeoutError:
        return SandboxResult(
            success=False,
            output="",
            error=f"命令执行超时 ({timeout}s)",
            exit_code=-1,
            duration_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as e:  # pragma: no cover - 防御性兜底
        return SandboxResult(
            success=False,
            output="",
            error=str(e),
            exit_code=-1,
            duration_ms=(time.monotonic() - start) * 1000,
        )
