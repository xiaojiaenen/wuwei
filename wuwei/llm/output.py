"""Structured output strategy shared by the runner and gateway."""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .profile import ModelProfile

STRUCTURED_TOOL_NAME = "__wuwei_structured_output"


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: dict = Field(alias="schema")
    name: str = Field(default="structured_output", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    mode: Literal["auto", "native", "tool", "prompted"] = "auto"

    def resolve(self, profile: ModelProfile) -> str:
        mode = (
            ("native" if profile.capabilities.native_schema else "prompted")
            if self.mode == "auto"
            else self.mode
        )
        if mode == "native" and not profile.capabilities.native_schema:
            raise ValueError(f"{profile.name} does not support native structured output")
        if mode == "tool" and not profile.capabilities.tool_calling:
            raise ValueError(f"{profile.name} does not support tool output")
        return mode


def strict_json_schema(schema: dict) -> dict:
    """Build the strict object subset without mutating or weakening open maps.

    Native strict output requires explicit fields (nullable when optional).
    Pydantic defaults are applied locally, not sent as unsupported JSON defaults.
    Dynamic dictionaries/extra=allow cannot be represented faithfully; callers
    can select prompted mode rather than silently losing those fields.
    """
    result = deepcopy(schema)
    if result.get("type") != "object" and "$ref" not in result:
        raise ValueError("Strict structured output requires an object root")

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                additional = node.get("additionalProperties", False)
                if additional is not False:
                    raise ValueError(
                        "Strict output cannot represent open additionalProperties; use prompted mode"
                    )
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for child in node.values():
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(result)
    return result


def apply_output(request: dict, spec: OutputSpec, provider: str):
    schema = (
        strict_json_schema(spec.schema_) if provider in {"openai", "responses"} else spec.schema_
    )
    if provider == "openai":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": spec.name, "strict": True, "schema": schema},
        }
    elif provider == "responses":
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": spec.name,
                "strict": True,
                "schema": schema,
            }
        }
    elif provider == "anthropic":
        request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    else:
        raise ValueError(f"Native structured output is unsupported by {provider}")
