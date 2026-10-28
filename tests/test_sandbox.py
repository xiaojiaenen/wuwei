"""wuwei.sandbox 沙箱驱动器测试

覆盖：

- LocalSandbox 旧接口兼容（execute / run / success / stdout）
- LocalPythonDriver：正常执行、异常退出、超时 kill、files 注入、run_command、
  资源限制参数
- SandboxResult 新旧字段兼容（output/output 别名、duration_ms）
- DockerDriver：无 docker 时跳过实际执行；SandboxUnavailable 逻辑
- E2BDriver：SDK 缺失时的 ImportError 提示、缺 api_key 的 SandboxUnavailable
- get_sandbox 工厂
"""

import shutil
import sys
from importlib.util import find_spec

import pytest

from wuwei.sandbox import (
    DockerDriver,
    E2BDriver,
    LocalPythonDriver,
    LocalSandbox,
    SandboxDriver,
    SandboxResult,
    SandboxUnavailable,
    get_sandbox,
)
from wuwei.sandbox.base import BaseSandbox

# ---------------------------------------------------------------------------
# SandboxResult
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_new_fields(self):
        r = SandboxResult(success=True, output="hi", exit_code=0, duration_ms=1.5)
        assert r.success is True
        assert r.output == "hi"
        assert r.error == ""
        assert r.duration_ms == 1.5

    def test_legacy_aliases_read(self):
        r = SandboxResult(success=False, output="", error="boom", exit_code=1)
        assert r.stdout == ""
        assert r.stderr == "boom"

    def test_legacy_fields_construct(self):
        # 旧代码用 stdout/stderr 构造
        r = SandboxResult(stdout="hello", stderr="", exit_code=0, success=True)
        assert r.output == "hello"
        assert r.stdout == "hello"
        assert r.success is True

    def test_str_success_and_failure(self):
        ok = SandboxResult(success=True, output="out")
        assert str(ok) == "out"
        bad = SandboxResult(success=False, error="err text", exit_code=2)
        assert "exit code 2" in str(bad)
        assert "err text" in str(bad)


# ---------------------------------------------------------------------------
# LocalSandbox（旧接口兼容）
# ---------------------------------------------------------------------------


class TestLocalSandboxCompat:
    async def test_execute_echo(self):
        s = LocalSandbox()
        result = await s.execute("echo hello")
        assert result.success
        assert "hello" in result.stdout
        assert result.exit_code == 0

    async def test_run_and_run_command_aliases(self):
        s = LocalSandbox()
        r1 = await s.run("echo via-run")
        assert r1.success and "via-run" in r1.output
        r2 = await s.run_command("echo via-rc")
        assert r2.success and "via-rc" in r2.output

    async def test_error_command(self):
        s = LocalSandbox()
        result = await s.execute("nonexistent_command_xyz 2>&1")
        assert isinstance(result, SandboxResult)
        assert not result.success
        assert result.exit_code != 0

    async def test_timeout(self):
        s = LocalSandbox()
        result = await s.execute("sleep 5", timeout=0.5)
        assert not result.success
        assert result.exit_code == -1
        assert "超时" in result.stderr

    async def test_cwd_workspace(self, tmp_path):
        s = LocalSandbox(workspace=str(tmp_path))
        result = await s.execute("pwd")
        assert result.stdout.strip() == str(tmp_path)

    async def test_context_manager(self):
        async with LocalSandbox() as s:
            result = await s.execute("echo ctx")
            assert result.success
        assert isinstance(s, BaseSandbox)


# ---------------------------------------------------------------------------
# LocalPythonDriver
# ---------------------------------------------------------------------------


class TestLocalPythonDriver:
    async def test_run_code_normal(self):
        d = LocalPythonDriver()
        r = await d.run_code("print('hi from sandbox')")
        assert r.success, r.error
        assert "hi from sandbox" in r.output
        assert r.exit_code == 0
        assert r.duration_ms > 0

    async def test_run_code_stdout_multiline(self):
        d = LocalPythonDriver()
        r = await d.run_code("for i in range(3): print(i)")
        assert r.output.splitlines() == ["0", "1", "2"]

    async def test_run_code_exception(self):
        d = LocalPythonDriver()
        r = await d.run_code("raise ValueError('boom')")
        assert not r.success
        assert r.exit_code != 0
        assert "ValueError" in r.error
        assert "boom" in r.error

    async def test_run_code_timeout(self):
        d = LocalPythonDriver()
        r = await d.run_code("while True: pass", timeout=1.0)
        assert not r.success
        assert r.exit_code == -1
        assert "超时" in r.error
        assert r.duration_ms >= 1000

    async def test_run_code_files_injection(self):
        d = LocalPythonDriver()
        files = {
            "data.txt": b"hello file",
            "sub/dir/nested.json": b'{"a": 1}',
        }
        code = (
            "import json, os\n"
            "print(open('data.txt').read())\n"
            "print(os.path.exists('sub/dir/nested.json'))\n"
            "print(json.load(open('sub/dir/nested.json'))['a'])\n"
            "print(os.getcwd() != '')\n"
        )
        r = await d.run_code(code, files=files)
        assert r.success, r.error
        lines = r.output.splitlines()
        assert lines[0] == "hello file"
        assert lines[1] == "True"
        assert lines[2] == "1"

    async def test_files_are_temporary(self):
        d = LocalPythonDriver()
        r = await d.run_code("open('marker.txt','w').write('x')", files={"seed.txt": b"s"})
        assert r.success, r.error
        # 新的一次执行看不到上次注入目录里的文件
        r2 = await d.run_code(
            "import os; print(os.path.exists('marker.txt'))"
        )
        assert "False" in r2.output

    async def test_run_code_isolated_mode(self):
        # -I 模式下 sys.flags.isolated == 1
        d = LocalPythonDriver()
        code = "import sys; print(sys.flags.isolated)"
        r = await d.run_code(code)
        assert r.success, r.error
        assert r.output.splitlines()[0] == "1"

    async def test_run_command(self):
        d = LocalPythonDriver()
        r = await d.run_command("echo shell-ok")
        assert r.success
        assert "shell-ok" in r.output

    async def test_run_command_error(self):
        d = LocalPythonDriver()
        r = await d.run_command("exit 3")
        assert not r.success
        assert r.exit_code == 3

    def test_resource_limit_params_accepted(self):
        # 仅验证参数被接受且 preexec 构造不报错
        d = LocalPythonDriver(cpu_seconds=5, memory_bytes=256 * 1024 * 1024)
        preexec = d._make_preexec_fn()
        if sys.platform != "win32":
            assert callable(preexec)
        # 正常执行不受影响
        import asyncio

        r = asyncio.run(d.run_code("print('limited ok')"))
        assert r.success, r.error

    def test_no_limits_preexec_is_none(self):
        assert LocalPythonDriver()._make_preexec_fn() is None

    def test_protocol_conformance(self):
        # LocalPythonDriver 完整实现 SandboxDriver 协议
        assert isinstance(LocalPythonDriver(), SandboxDriver)
        # LocalSandbox 是旧版 shell 沙箱，只承诺 execute/run_command
        assert isinstance(LocalSandbox(), BaseSandbox)


# ---------------------------------------------------------------------------
# DockerDriver
# ---------------------------------------------------------------------------


def _docker_daemon_available() -> bool:
    if shutil.which("docker") is None:
        return False
    import subprocess

    try:
        proc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_daemon_available()

docker_needed = pytest.mark.skipif(
    not DOCKER_AVAILABLE, reason="docker 不可用（未安装或守护进程未启动）"
)


class TestDockerDriver:
    def test_defaults(self):
        d = DockerDriver()
        assert d.image == "python:3.12-slim"
        assert d.memory == "256m"
        assert d.cpus == "0.5"
        assert d.network_none is True

    @pytest.mark.skipif(
        shutil.which("docker") is not None,
        reason="已安装 docker CLI，无法验证缺失路径",
    )
    async def test_unavailable_raises_without_cli(self):
        d = DockerDriver(docker_bin="definitely-not-docker-xyz")
        with pytest.raises(SandboxUnavailable):
            await d.run_command("echo hi")

    async def test_unavailable_raises_with_bad_bin(self):
        # 即使本机有 docker，这个不存在的二进制也必然失败
        d = DockerDriver(docker_bin="definitely-not-docker-xyz")
        with pytest.raises(SandboxUnavailable):
            await d.run_command("echo hi")

    @docker_needed
    async def test_run_command(self):
        d = DockerDriver()
        r = await d.run_command("echo docker-ok")
        assert r.success, r.error
        assert "docker-ok" in r.output

    @docker_needed
    async def test_run_code(self):
        d = DockerDriver()
        r = await d.run_code("print(1 + 1)")
        assert r.success, r.error
        assert "2" in r.output

    @docker_needed
    async def test_run_code_files(self):
        d = DockerDriver()
        r = await d.run_code(
            "print(open('injected.txt').read())",
            files={"injected.txt": b"from-stdin-b64"},
        )
        assert r.success, r.error
        assert "from-stdin-b64" in r.output

    @docker_needed
    async def test_run_code_exception(self):
        d = DockerDriver()
        r = await d.run_code("raise RuntimeError('container boom')")
        assert not r.success
        assert "RuntimeError" in r.error


# ---------------------------------------------------------------------------
# E2BDriver
# ---------------------------------------------------------------------------


class TestE2BDriver:
    def test_lazy_import_no_sdk_side_effect(self):
        # 构造本身不应触发 SDK 导入
        d = E2BDriver(api_key="fake")
        assert d._sdk is None
        assert d._sandbox is None

    @pytest.mark.skipif(
        find_spec("e2b") is not None, reason="e2b 已安装，无法验证缺失提示"
    )
    async def test_missing_sdk_import_error(self):
        d = E2BDriver(api_key="fake")
        with pytest.raises(ImportError) as exc_info:
            await d.run_code("print(1)")
        assert "pip install" in str(exc_info.value)

    @pytest.mark.skipif(
        find_spec("e2b") is not None, reason="e2b 已安装，无法验证缺失提示"
    )
    async def test_missing_sdk_run_command(self):
        d = E2BDriver(api_key="fake")
        with pytest.raises(ImportError):
            await d.run_command("ls")

    @pytest.mark.skipif(
        find_spec("e2b") is None, reason="e2b 已安装时无法走 api_key 缺失路径"
    )
    async def test_missing_api_key(self):
        import os

        env_backup = os.environ.pop("E2B_API_KEY", None)
        try:
            d = E2BDriver()
            with pytest.raises(SandboxUnavailable):
                await d.run_code("print(1)")
        finally:
            if env_backup is not None:
                os.environ["E2B_API_KEY"] = env_backup


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


class TestFactory:
    def test_local(self):
        s = get_sandbox("local")
        assert isinstance(s, LocalSandbox)

    def test_local_sandbox_alias(self):
        assert isinstance(get_sandbox("LOCAL-SANDBOX"), LocalSandbox)

    def test_local_python(self):
        d = get_sandbox("local-python")
        assert isinstance(d, LocalPythonDriver)

    def test_python_alias(self):
        assert isinstance(get_sandbox("Python"), LocalPythonDriver)

    def test_docker(self):
        assert isinstance(get_sandbox("docker"), DockerDriver)

    def test_e2b(self):
        assert isinstance(get_sandbox("e2b"), E2BDriver)

    def test_kwargs_passthrough(self):
        d = get_sandbox("docker", image="my-image:latest")
        assert d.image == "my-image:latest"

    def test_unknown_name(self):
        with pytest.raises(ValueError) as exc_info:
            get_sandbox("no-such-driver")
        assert "no-such-driver" in str(exc_info.value)


# ---------------------------------------------------------------------------
# __init__ 兼容
# ---------------------------------------------------------------------------


def test_package_exports():
    from wuwei.sandbox import LocalSandbox as pkg_sandbox  # noqa: N813
    from wuwei.sandbox import base, local  # noqa: F401
    from wuwei.sandbox.local import LocalSandbox as mod_sandbox  # noqa: N813

    assert pkg_sandbox is mod_sandbox
    assert issubclass(mod_sandbox, BaseSandbox)
