from contextlib import contextmanager
from contextvars import ContextVar

_governed_execution = ContextVar("wuwei_governed_execution", default=None)


@contextmanager
def bind_tool_execution(callback):
    token = _governed_execution.set(callback)
    try:
        yield
    finally:
        _governed_execution.reset(token)


def governed_tool_execution():
    return _governed_execution.get()


import asyncio
import json
from typing import Any

from pydantic import BaseModel

from wuwei.llm.types import Message, ToolCall
from wuwei.tools.registry import ToolRegistry
from wuwei.tools.tool import Tool

DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_MAX_OUTPUT_CHARS = 100_000


class ToolTimeout(TimeoutError):
    """The framework's deadline fired; the underlying call may still be running.

    Distinct from a ``TimeoutError`` raised by the tool itself: only this one means
    the framework stopped waiting, which is the case where the effect is in doubt.
    """


def _truncate(text: str, limit: int | None) -> str:
    if not limit or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def serialize_tool_output(output: Any, limit: int | None = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Serialize a tool result, bounding what one call can spend on context."""
    if isinstance(output, str):
        return _truncate(output, limit)
    if isinstance(output, BaseModel):
        return _truncate(output.model_dump_json(exclude_none=True), limit)
    try:
        return _truncate(json.dumps(output, ensure_ascii=False, default=str), limit)
    except TypeError:
        return _truncate(json.dumps({"value": str(output)}, ensure_ascii=False), limit)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_output_chars: int | None = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        self.registry = registry
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_output_chars = max_output_chars
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop = None

    def _concurrency_slot(self) -> asyncio.Semaphore:
        # Bind lazily so a shared executor stays usable across event loops (tests,
        # worker restarts) instead of raising "attached to a different loop".
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    async def execute(
        self,
        tool_calls: list[ToolCall],
        concurrent: bool = False,
    ) -> list[Message]:
        """执行工具调用列表

        当 concurrent=True 时：
        - 按 concurrency_safe 分组
        - 安全组并发执行（受 max_concurrency 限制）
        - 非安全组保持顺序串行执行
        """
        if not tool_calls:
            return []

        if concurrent and len(tool_calls) > 1:
            return await self._execute_batched(tool_calls)

        results: list[Message] = []
        for tool_call in tool_calls:
            results.append(await self.execute_one(tool_call))
        return results

    async def _execute_batched(
        self,
        tool_calls: list[ToolCall],
    ) -> list[Message]:
        """按 concurrency_safe 分批次并发执行

        策略：
        1. 扫描 tool_calls，将连续安全工具分组
        2. 遇到非安全工具时，等待前面批次完成后再执行
        3. 安全工具并发执行，但并发度受 max_concurrency 限制
        4. 结果按原始顺序排列
        """
        results: list[Message | None] = [None] * len(tool_calls)

        async def guarded(tool_call: ToolCall) -> Message:
            async with self._concurrency_slot():
                return await self.execute_one(tool_call)

        i = 0
        while i < len(tool_calls):
            tc = tool_calls[i]
            tool = self.registry.get(tc.function.name)
            is_safe = tool.concurrency_safe if tool else True

            if is_safe:
                batch = [tc]
                batch_indices = [i]
                j = i + 1
                while j < len(tool_calls):
                    next_tool = self.registry.get(tool_calls[j].function.name)
                    if next_tool and next_tool.concurrency_safe:
                        batch.append(tool_calls[j])
                        batch_indices.append(j)
                        j += 1
                    else:
                        break

                batch_results = await asyncio.gather(
                    *(guarded(tool) for tool in batch),
                    return_exceptions=True,
                )
                for idx, result in zip(batch_indices, batch_results):
                    if isinstance(result, BaseException):
                        # CancelledError and other BaseException subclasses must not be
                        # flattened into tool payloads: flattening loses the cancellation
                        # and puts a non-Message object into the result list.
                        if not isinstance(result, Exception):
                            raise result
                        results[idx] = self.build_error_message(
                            tool_calls[idx],
                            error_type=type(result).__name__,
                            message=str(result),
                        )
                    else:
                        results[idx] = result

                i = j
            else:
                results[i] = await self.execute_one(tc)
                i += 1

        missing = [index for index, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError(f"Tool batch produced no result for indices {missing}")
        return results

    async def execute_one(self, tool_call: ToolCall) -> Message:
        tool = self.registry.get(tool_call.function.name)
        if tool is None:
            return self.build_error_message(
                tool_call,
                error_type="ToolNotFound",
                message=f"Tool '{tool_call.function.name}' not found",
                metadata={"attempts": 0},
            )

        max_attempts = (
            1
            if tool.execution.side_effect
            else tool.execution.retry_policy.normalized_max_attempts()
        )
        backoff_seconds = max(0.0, tool.execution.retry_policy.backoff_seconds)

        attempt = 0
        while True:
            attempt += 1
            try:
                output = await self._invoke_tool(tool, tool_call.function.arguments)
            except ToolTimeout:
                # A timed-out call is never retried: the framework already knows it
                # could not finish, and a retry may duplicate the effect.
                return self._timeout_message(tool, tool_call, attempt)
            except Exception as exc:
                if attempt >= max_attempts or self._not_retryable(exc):
                    return self._failure_message(tool, tool_call, exc, attempt, max_attempts)
                if backoff_seconds:
                    await asyncio.sleep(backoff_seconds)
                continue
            return self._success_message(tool, tool_call, output)

    async def _invoke_tool(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        timeout_seconds = tool.execution.timeout_seconds
        if timeout_seconds is None:
            return await tool.invoke(arguments)
        call = asyncio.ensure_future(tool.invoke(arguments))
        try:
            done, _ = await asyncio.wait({call}, timeout=timeout_seconds)
        except BaseException:
            call.cancel()
            raise
        if not done:
            call.cancel()
            raise ToolTimeout(
                f"Tool '{tool.name}' timed out after {timeout_seconds} seconds"
            )
        return call.result()

    @staticmethod
    def _not_retryable(exc: Exception) -> bool:
        return isinstance(exc, (ValueError, TypeError, TimeoutError)) or type(
            exc
        ).__module__.startswith(("jsonschema", "pydantic"))

    def _success_message(self, tool: Tool, tool_call: ToolCall, output: Any) -> Message:
        from wuwei.tools.outcome import ToolOutcome

        if isinstance(output, ToolOutcome):
            return output.to_message(
                tool_call.id, tool_call.function.name, limit=self.max_output_chars
            )
        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=self.serialize_output(output),
            name=tool_call.function.name,
            metadata={"tool_outcome": {"status": "success", "executed": True}},
        )

    def _timeout_message(self, tool: Tool, tool_call: ToolCall, attempts: int) -> Message:
        policy = tool.execution
        # Unknown only when a second execution could compound a real effect.
        uncertain = policy.side_effect and not policy.idempotent
        if uncertain:
            detail = (
                "it may still be running on the provider side; do not retry blindly, "
                "reconcile the actual result first"
            )
        else:
            detail = "no lasting effect is expected, it can be retried safely"
        return self.build_error_message(
            tool_call,
            error_type="ToolTimeout",
            message=(
                f"Tool '{tool.name}' timed out after {policy.timeout_seconds} seconds; {detail}"
            ),
            metadata={
                "executed": True,
                "attempts": attempts,
                "status": "unknown" if uncertain else "failed",
                "execution_may_continue": True,
                "retryable": not uncertain,
            },
        )

    def _failure_message(
        self,
        tool: Tool,
        tool_call: ToolCall,
        exc: Exception,
        attempts: int,
        max_attempts: int,
    ) -> Message:
        return self.build_error_message(
            tool_call,
            error_type=type(exc).__name__,
            message=str(exc),
            metadata={
                "executed": bool(getattr(exc, "_wuwei_tool_executed", False)),
                "attempts": attempts,
                "status": "failed",
                "execution_may_continue": False,
                "retryable": attempts < max_attempts,
            },
        )

    def build_error_message(
        self,
        tool_call: ToolCall,
        *,
        error_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        if metadata:
            payload.update(metadata)
        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            content=json.dumps(payload, ensure_ascii=False),
            name=tool_call.function.name,
            metadata={
                "tool_outcome": {
                    "executed": False,
                    **(metadata or {}),
                    "status": "unknown" if (metadata or {}).get("status") == "unknown" else "error",
                }
            },
        )

    def serialize_output(self, output: Any) -> str:
        # getattr: this used to be called as ToolExecutor.serialize_output(None, x).
        limit = getattr(self, "max_output_chars", DEFAULT_MAX_OUTPUT_CHARS)
        return serialize_tool_output(output, limit)

    @staticmethod
    def extract_error_message(content: str | None) -> str | None:
        if not content:
            return None

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict) or payload.get("ok") is not False:
            return None

        error = payload.get("error")
        if not isinstance(error, dict):
            return None

        message = error.get("message")
        return message if isinstance(message, str) else None
