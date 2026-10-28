"""Owned subprocess communication and POSIX process-group cleanup."""

import asyncio
import contextlib
import os
import signal


def process_options():
    return {"start_new_session": True} if os.name == "posix" else {}


async def communicate(process, data=None, *, timeout):
    # Shield the pipe readers so cancellation never abandons subprocess transports.
    task = asyncio.create_task(process.communicate(data))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except (TimeoutError, asyncio.CancelledError):
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        await asyncio.shield(task)
        await process.wait()
        raise
