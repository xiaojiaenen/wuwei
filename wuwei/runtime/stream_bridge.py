"""Bounded bridge between a run and whoever consumes its events.

A run holds ``session.lock`` for its whole duration. If the lock is taken inside
an async generator that also yields events, the lock's lifetime is bound to the
*generator object*: a consumer that breaks out of ``async for`` and drops the
generator without ``aclose()`` keeps the lock forever, and every later run on
that session blocks with no trace in the logs.

Here the producer runs in its own task, so the lock is released when the run
ends or is cancelled — never because a consumer forgot to clean up. Events cross
a bounded queue, which gives back-pressure without dropping anything.

The consumer is declared gone when an event cannot be enqueued for
``idle_timeout`` seconds, i.e. the queue stayed full and nobody read. The timeout
deliberately does not guard the lock or the model call: those measure something
else, and expiring them would let a second run into a session whose tool is still
executing, which is worse than waiting.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress

logger = logging.getLogger(__name__)

_END = object()


class _ConsumerGoneError(Exception):
    """Internal signal: the queue stayed full past the idle timeout."""


async def _close_source(source) -> None:
    """Close the producer generator so its own cleanup always runs."""
    close = getattr(source, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except (asyncio.CancelledError, GeneratorExit):
        pass
    except Exception:
        logger.warning("stream producer cleanup failed", exc_info=True)


async def bridge(
    source: AsyncIterator,
    *,
    queue_size: int = 64,
    idle_timeout: float | None = 30.0,
    name: str | None = None,
) -> AsyncIterator:
    """Drive ``source`` in a task and re-publish its events through a queue.

    ``idle_timeout`` must be positive or ``None``. ``None`` restores the old
    behaviour (the producer waits forever) but keeps the lock owned by the task.
    """
    if idle_timeout is not None and idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive or None")

    queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(queue_size)))
    failure: list[BaseException] = []

    async def emit(item) -> None:
        if idle_timeout is None:
            await queue.put(item)
            return
        try:
            await asyncio.wait_for(queue.put(item), idle_timeout)
        except TimeoutError:
            raise _ConsumerGoneError from None

    async def pump() -> None:
        try:
            try:
                async for event in source:
                    await emit(event)
            except _ConsumerGoneError:
                logger.warning(
                    "event stream abandoned: consumer read nothing for %.1fs "
                    "(queue_size=%s); cancelling the run",
                    idle_timeout,
                    queue.maxsize,
                )
                return
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except BaseException as exc:  # surfaced to the consumer below
                failure.append(exc)
            with suppress(_ConsumerGoneError):
                await emit(_END)
        finally:
            await _close_source(source)

    producer = asyncio.create_task(pump(), name=name or "wuwei-event-bridge")
    try:
        while True:
            item = await queue.get()
            if item is _END:
                break
            yield item
    finally:
        if not producer.done():
            producer.cancel()
        # The consumer's own cancellation is re-raised by the enclosing try/finally;
        # suppressing here only hides the producer's exit.
        with suppress(BaseException):
            await producer
    if failure:
        raise failure[0]
