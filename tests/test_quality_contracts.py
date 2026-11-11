import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wuwei.evals import EvalCase, EvalRunner
from wuwei.sandbox.local import LocalPythonDriver


def _quote_command(parts: list[str]) -> str:
    """跨平台把参数列表拼成 shell 命令（POSIX 单引号 / Windows cmd 引号规则）。"""
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    import shlex

    return " ".join(shlex.quote(p) for p in parts)


def _pid_alive(pid: int) -> bool:
    """探测进程是否（仍能执行代码）存活。

    Windows：``os.kill(pid, 0)`` 语义不同（sig=0 不是探测），用 OpenProcess +
    GetExitCodeProcess（终止后的进程对象在句柄释放前仍可打开）。
    POSIX：僵尸进程在无回收 init 的容器里会永久保留表项，``kill(pid, 0)`` 仍
    成功——必须查进程状态，只有非 Z 状态才算存活。
    """
    if os.name != "nt":
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                # stat 形如 "pid (comm) state ..."；comm 可能含空格与括号，
                # 以最后一个 ")" 之后的首字段为状态。
                state = f.read().rsplit(b")", 1)[-1].split()[0]
            return state != b"Z"
        except (OSError, IndexError):
            # 无 /proc（如 macOS）：退回 kill(pid, 0) 存在性探测
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True
    import ctypes
    import ctypes.wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        # 终止后的进程对象在句柄释放前 OpenProcess 仍会成功，必须查退出码。
        exit_code = ctypes.wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


class Agent:
    def create_session(self, **kw):
        return SimpleNamespace(context=SimpleNamespace(get_messages=lambda: []))

    async def run(self, text, session):
        return SimpleNamespace(
            content="ok", status="incomplete", termination_reason="max_steps", usage=None
        )


@pytest.mark.asyncio
async def test_incomplete_is_not_green():
    result = await EvalRunner(Agent).run_case(
        EvalCase(name="x", input="hi", response_contains=["ok"])
    )
    assert not result.passed
    assert result.status == "incomplete"
    assert not result.usage_available


@pytest.mark.asyncio
async def test_empty_and_missing_required_judge_fail():
    class Completed(Agent):
        async def run(self, text, session):
            return SimpleNamespace(content="ok", status="completed", usage={})

    runner = EvalRunner(Completed)
    assert not (await runner.run_case(EvalCase(name="empty", input="x"))).passed
    assert not (await runner.run_case(EvalCase(name="judge", input="x", rubric="good"))).passed


@pytest.mark.asyncio
async def test_cancel_local_process_is_reaped(tmp_path):
    path = tmp_path / "pid"
    task = asyncio.create_task(
        LocalPythonDriver().run_code(
            f'import os,time; open({str(path)!r},"w").write(str(os.getpid())); time.sleep(60)'
        )
    )
    for _ in range(100):
        if path.exists():
            break
        await asyncio.sleep(0.01)
    pid = int(path.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _pid_alive(pid), "cancelled subprocess remains alive"


@pytest.mark.asyncio
async def test_tracing_root_has_ids_and_content_optin():
    from wuwei.graph.state import State
    from wuwei.middleware.base import MiddlewareContext
    from wuwei.observability.tracing import TracingMiddleware

    middleware = TracingMiddleware()
    middleware._tracer = None
    ctx = MiddlewareContext(
        state=State(),
        config={"run_state": {"run_id": "r", "session_id": "s", "parent_run_id": "p"}},
    )
    await middleware.before_run(ctx)
    await middleware.after_run(
        ctx, SimpleNamespace(status="completed", usage={"total_tokens": 4}, termination_reason=None)
    )
    assert middleware.records[-1]["run_id"] == "r"
    assert middleware.records[-1]["parent_run_id"] == "p"
    assert "content" not in middleware.records[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "interrupted", "unknown", "cancelled", "error"])
async def test_expected_noncompleted_status_optin(status):
    class Stopped(Agent):
        async def run(self, text, session):
            return SimpleNamespace(content="ok", status=status, usage={"total_tokens": 5})

    runner = EvalRunner(Stopped)
    assert not (
        await runner.run_case(EvalCase(name="x", input="q", response_contains=["ok"]))
    ).passed
    assert (
        await runner.run_case(
            EvalCase(name="x", input="q", expected_status=status, response_contains=["ok"])
        )
    ).passed


@pytest.mark.asyncio
async def test_multiturn_limits_and_baseline(tmp_path):
    from wuwei.evals import EvalReport

    class Completed(Agent):
        def __init__(self):
            self.inputs = []

        async def run(self, text, session):
            self.inputs.append(text)
            return SimpleNamespace(
                content="ok", status="completed", usage={"total_tokens": 3, "cost": 0.1}
            )

    agent = Completed()
    result = await EvalRunner(lambda: agent).run_case(
        EvalCase(name="x", input="one", turns=["two"], max_tokens=5, max_cost=0.15)
    )
    assert agent.inputs == ["one", "two"]
    assert result.usage["total_tokens"] == 6
    assert not result.passed
    skipped = result.model_copy(update={"passed": True, "skipped": True})
    report = EvalReport(results=[skipped])
    xml = report.write_junit(tmp_path / "junit.xml").read_text()
    assert "<skipped" in xml and 'failures="0"' in xml
    baseline = EvalReport(results=[result.model_copy(update={"passed": True})])
    assert report.compare(baseline)["regressions"] == ["x"]


@pytest.mark.asyncio
async def test_tool_request_not_success_and_denial_not_execution():
    from wuwei.core.message import AIMessage, FunctionCall, ToolCall, ToolMessage
    from wuwei.evals.runner import extract_tool_trajectories

    calls = [ToolCall(id="a", function=FunctionCall(name="write", arguments={}))]
    messages = [AIMessage(content="", tool_calls=calls)]
    session = SimpleNamespace(context=SimpleNamespace(get_messages=lambda: messages))
    assert extract_tool_trajectories(session)["succeeded"] == []
    messages.append(
        ToolMessage(
            content="denied",
            tool_call_id="a",
            name="write",
            status="error",
            metadata={"tool_outcome": {"status": "denied"}},
        )
    )
    assert extract_tool_trajectories(session) == {
        "requested": ["write"],
        "executed": [],
        "succeeded": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", ["shell", "docker"])
async def test_cancel_other_owned_subprocesses(driver, tmp_path):
    from wuwei.sandbox.base import run_shell
    from wuwei.sandbox.docker import DockerDriver

    if driver == "docker" and not _docker_available():
        pytest.skip("docker daemon unavailable")

    path = tmp_path / "pid"
    code = f'import os,time; open({str(path)!r},"w").write(str(os.getpid())); time.sleep(60)'
    if driver == "shell":
        # Python worker 是 shell 的孙进程：直接 kill shell 不会传播，
        # 因此取消后整棵树都必须消失。
        command = _quote_command([sys.executable, "-c", code])
        operation = run_shell(command)
    else:
        operation = DockerDriver()._run([sys.executable, "-c", code], stdin_data=b"", timeout=60)
    task = asyncio.create_task(operation)
    for _ in range(100):
        if path.exists():
            break
        await asyncio.sleep(0.01)
    pid = int(path.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _pid_alive(pid), f"cancelled {driver} subprocess remains alive"


@pytest.mark.asyncio
async def test_shell_timeout_kills_descendant(tmp_path):
    from wuwei.sandbox.base import run_shell

    path = tmp_path / "pid"
    code = f'import os,time; open({str(path)!r}, "w").write(str(os.getpid())); time.sleep(60)'
    worker = _quote_command([sys.executable, "-c", code])
    if os.name == "nt":
        # cmd.exe：`&` 顺序执行，python worker 是 shell 的子进程且存活 60s，
        # 超时后整棵树必须被终止。
        command = f"{worker} & ping -n 60 127.0.0.1 >nul"
    else:
        command = f"{worker} & wait"
    result = await run_shell(command, timeout=0.1)
    assert not result.success
    for _ in range(100):
        if path.exists():
            break
        await asyncio.sleep(0.01)
    pid = int(path.read_text())
    assert not _pid_alive(pid), "timed-out shell descendant is still running"


@pytest.mark.asyncio
async def test_trace_facts_redact_and_bound():
    from wuwei.core.message import AIMessage
    from wuwei.graph.state import State
    from wuwei.middleware.base import MiddlewareContext
    from wuwei.observability.tracing import TracingMiddleware

    middleware = TracingMiddleware(capture_content=True, content_limit=60, record_limit=2)
    middleware._tracer = None
    ctx = MiddlewareContext(state=State(), config={"provider": "fake", "model": "fake-model"})
    await middleware.before_run(ctx)
    response = AIMessage(
        content="ok", usage={"total_tokens": 10, "cached_tokens": 4}, metadata={"retry_count": 2}
    )
    await middleware.after_llm(ctx, response)
    await middleware.after_run(
        ctx,
        SimpleNamespace(content="token=secret-value " + "x" * 100, status="completed", usage=None),
    )
    assert len(middleware.records) == 2
    assert middleware.records[0]["retry_count"] == 2
    assert middleware.records[0]["usage"]["cached_tokens"] == 4
    assert middleware.records[0]["provider"] == "fake"
    assert "secret-value" not in middleware.records[-1]["content"]
    assert len(middleware.records[-1]["content"]) <= 60


@pytest.mark.asyncio
async def test_eval_factory_owned_agent_is_closed_and_cleanup_failure_is_visible():
    class Closeable(Agent):
        closed = False

        async def aclose(self):
            self.closed = True
            raise OSError("close failed")

    agent = Closeable()
    result = await EvalRunner(lambda: agent).run_case(
        EvalCase(name="cleanup", input="hello", response_contains=["ok"])
    )
    assert agent.closed and not result.passed
    assert any(a.kind == "lifecycle" for a in result.assertions)
