"""Docker 沙箱驱动器

用 docker CLI 子进程在容器中执行代码 / 命令：

- ``run_code``：``docker run --rm -i <limits> <image> python -I -c <bootstrap>``，
  代码与注入文件以 JSON+base64 经 stdin 传入（避免 ``docker cp`` 的
  临时容器管理），bootstrap 解码文件后 exec 用户代码。
- ``run_command``：``docker run --rm -i <limits> <image> sh -c <command>``。

docker CLI 缺失或守护进程不可达时抛出
:class:`~wuwei.sandbox.base.SandboxUnavailable`。
"""

import asyncio
import base64
import contextlib
import json
import shutil
import time

from wuwei.sandbox.base import SandboxResult, SandboxUnavailable

# 经 -c 传入容器内 python 的 bootstrap：从 stdin 读取 JSON 载荷
# （{"code": b64, "files": {path: b64}}），写入文件后执行用户代码。
_BOOTSTRAP = """\
import base64, json, os, sys

payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
for path, data in (payload.get("files") or {}).items():
    full = os.path.join(os.getcwd(), path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full, "wb") as f:
        f.write(base64.b64decode(data))
source = base64.b64decode(payload["code"]).decode("utf-8")
exec(compile(source, "<sandbox>", "exec"), {"__name__": "__main__"})
"""


class DockerDriver:
    """Docker 容器沙箱驱动器。

    Args:
        image: 基础镜像，默认 ``python:3.12-slim``。
        memory: 内存限制（docker ``--memory``），默认 ``256m``。
        cpus: CPU 限制（docker ``--cpus``），默认 ``0.5``。
        network_none: 是否以 ``--network none`` 禁网运行，默认 True。
        workdir: 容器内工作目录，默认 ``/sandbox``。
        docker_bin: docker CLI 可执行文件名/路径。
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        memory: str = "256m",
        cpus: str = "0.5",
        network_none: bool = True,
        workdir: str = "/sandbox",
        docker_bin: str = "docker",
    ):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network_none = network_none
        self.workdir = workdir
        self.docker_bin = docker_bin
        self._docker_available: bool | None = None

    # ---- 可用性检查 ----

    async def _ensure_available(self):
        """检查 docker CLI 与守护进程是否可用（结果缓存）。"""
        if self._docker_available:
            return
        if shutil.which(self.docker_bin) is None:
            raise SandboxUnavailable(
                f"未找到 docker CLI（{self.docker_bin}）。"
                "请安装 Docker Desktop / docker engine 后重试。"
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_bin, "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        except (TimeoutError, OSError) as e:
            self._docker_available = False
            raise SandboxUnavailable(f"docker 守护进程不可用: {e}") from e
        if proc.returncode != 0:
            self._docker_available = False
            detail = (stderr or b"").decode(errors="replace").strip()
            raise SandboxUnavailable(
                f"docker 守护进程不可用 (exit {proc.returncode}): {detail[:300]}"
            )
        self._docker_available = True

    # ---- 参数构造 ----

    def _base_args(self) -> list:
        args = [self.docker_bin, "run", "--rm", "-i"]
        if self.network_none:
            args.append("--network")
            args.append("none")
        if self.memory:
            args += ["--memory", self.memory]
        if self.cpus:
            args += ["--cpus", self.cpus]
        if self.workdir:
            args += ["-w", self.workdir]
        return args

    # ---- SandboxDriver 协议 ----

    async def run_code(
        self,
        code: str,
        *,
        timeout: float = 30,
        files: dict[str, bytes] | None = None,
    ) -> SandboxResult:
        """在一次性容器中执行 Python 代码（files 经 stdin+base64 注入）。"""
        await self._ensure_available()

        payload = json.dumps(
            {
                "code": base64.b64encode(code.encode("utf-8")).decode("ascii"),
                "files": {
                    path: base64.b64encode(content or b"").decode("ascii")
                    for path, content in (files or {}).items()
                },
            }
        ).encode("utf-8")

        argv = self._base_args() + [self.image, "python", "-I", "-c", _BOOTSTRAP]
        return await self._run(argv, stdin_data=payload, timeout=timeout)

    async def run_command(self, command: str, *, timeout: float = 30) -> SandboxResult:
        """在一次性容器中执行 shell 命令。"""
        await self._ensure_available()
        argv = self._base_args() + [self.image, "sh", "-c", command]
        return await self._run(argv, stdin_data=b"", timeout=timeout)

    # ---- 执行 ----

    async def _run(
        self,
        argv: list,
        *,
        stdin_data: bytes,
        timeout: float,
    ) -> SandboxResult:
        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            self._docker_available = False
            raise SandboxUnavailable(f"无法启动 docker 进程: {e}") from e

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_data), timeout=timeout
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):  # pragma: no cover
                await process.wait()
            return SandboxResult(
                success=False,
                error=f"沙箱执行超时 ({timeout}s)，容器已被终止",
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
