"""Versioned persistent state and explicitly transient business dependencies."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field, model_validator

Deps = TypeVar("Deps")


@dataclass
class RunContext(Generic[Deps]):
    deps: Deps
    run_id: str = ""
    session_id: str = ""
    parent_run_id: str | None = None


class RunState(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: int = 2
    run_id: str = ""
    session_id: str = ""
    step_index: int = 0
    tool_states: dict[str, str] = Field(default_factory=dict)
    approval_decisions: dict[str, Any] = Field(default_factory=dict)
    permission_state: dict[str, Any] = Field(default_factory=dict)
    status: str = "running"
    output_spec: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_invariants(self):
        allowed = {"preparing", "running", "unknown", "success", "error"}
        if any(value not in allowed for value in self.tool_states.values()):
            raise ValueError("Invalid persisted tool execution state")
        if any(
            self.tool_states.get(call_id) in {"success", "error"}
            for call_id in self.approval_decisions
        ):
            raise ValueError("Completed tool cannot retain a pending approval decision")
        return self

    @model_validator(mode="before")
    @classmethod
    def migrate(cls, value):
        value = dict(value)
        version = value.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported RunState schema version: {version}")
        value["schema_version"] = 2
        return value
