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
