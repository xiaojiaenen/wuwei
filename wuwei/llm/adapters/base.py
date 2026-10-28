from abc import ABC, abstractmethod
from typing import Any

from ..types import AdapterEvent, LLMResponse, Message

CLIENT_OPTIONS = {
    "timeout",
    "max_retries",
    "http_client",
    "default_headers",
    "default_query",
    "organization",
    "project",
    "webhook_secret",
}


def make_client(
    factory, api_key, base_url, options, client=None, client_factory=None, owns_client=None
):
    params = {k: options.pop(k) for k in list(options) if k in CLIENT_OPTIONS}
    params.setdefault("max_retries", 0)
    borrowed = client is not None
    result = (
        client
        if borrowed
        else (client_factory or factory)(api_key=api_key, base_url=base_url, **params)
    )
    return result, (not borrowed if owns_client is None else owns_client)


class BaseAdapter(ABC):
    async def aclose(self):
        if getattr(self, "owns_client", False):
            await self.client.close()

    @abstractmethod
    def build_request(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        stream: bool | None = False,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def call(self, request: Any) -> Any:
        pass

    @abstractmethod
    def parse_response(self, raw_response: Any) -> LLMResponse:
        pass

    @abstractmethod
    def parse_stream_chunk(self, chunk: Any) -> AdapterEvent | None:
        pass


def provider_dump(obj):
    """Lossless JSON-safe SDK output for forward-compatible consumers."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return vars(obj)


def output_parts(blocks):
    """Project known display fields while keeping the entire provider payload separately."""
    from ...core.message import CitationPart, TextPart

    parts = []
    for block in blocks:
        data = provider_dump(block)
        if data.get("type") in {"text", "output_text"}:
            parts.append(TextPart(text=data.get("text", "")))
        annotations = data.get("annotations", data.get("citations", [])) or []
        for annotation in annotations:
            a = provider_dump(annotation) if not isinstance(annotation, dict) else annotation
            value = a.get("url_citation", a)
            parts.append(
                CitationPart(
                    url=value.get("url"),
                    title=value.get("title"),
                    text=value.get("cited_text"),
                    provider_data=a,
                )
            )
    return parts
