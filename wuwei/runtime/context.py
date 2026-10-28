"""Task-local execution context; configuration objects never carry mutable run state."""

from contextvars import ContextVar

current_context: ContextVar = ContextVar("wuwei_run_context", default=None)
