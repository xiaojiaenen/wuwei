"""Owned subprocess communication and process-tree cleanup."""

import asyncio
import contextlib
import os
import signal
import subprocess


def process_options():
    return {"start_new_session": True} if os.name == "posix" else {}


def _kill_process_tree(pid: int) -> None:
    """终止整棵进程树（同步、快速）。

    POSIX 用进程组；Windows 上只 kill 直接子进程不够——shell 启动的命令
    （python worker 等）是孙进程，父进程死亡不会传播，因此用
    ``taskkill /F /T`` 终止整棵树。
    """
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
            timeout=10,
        )


async def communicate(process, data=None, *, timeout):
    # Shield the pipe readers so cancellation never abandons subprocess transports.
    task = asyncio.create_task(process.communicate(data))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except (TimeoutError, asyncio.CancelledError):
        with contextlib.suppress(ProcessLookupError):
            await asyncio.to_thread(_kill_process_tree, process.pid)
        await asyncio.shield(task)
        await process.wait()
        raise
