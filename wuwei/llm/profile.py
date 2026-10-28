"""Explicit endpoint capabilities; custom endpoints default to conservative support."""

from pydantic import BaseModel, ConfigDict


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    native_schema: bool = False
    tool_calling: bool = True
    image_input: bool = False
    file_input: bool = False
    streaming: bool = True


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = "custom"
    capabilities: ModelCapabilities = ModelCapabilities()


def default_profile(provider: str, custom_endpoint: bool = False) -> ModelProfile:
    return ModelProfile(
        name=provider,
        capabilities=ModelCapabilities(
            native_schema=not custom_endpoint and provider in {"openai", "responses", "anthropic"},
            image_input=not custom_endpoint,
            file_input=not custom_endpoint,
        ),
    )
