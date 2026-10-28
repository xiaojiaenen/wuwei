"""Typed, explicitly selected delegation inputs and outputs."""

from typing import Any
from pydantic import BaseModel, Field
from wuwei.artifacts import ArtifactRef


class DelegationInput(BaseModel):
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class DelegationOutput(BaseModel):
    sub_agent: str
    result: Any
    status: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    session_id: str
    interrupt: dict | None = None
    reconciliation: dict | None = None
    elapsed_ms: int = 0
    steps: int = 0
