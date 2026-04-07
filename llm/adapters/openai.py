from typing import Any

from base import Adapter
from llm.types import LLMResponseChunk, LLMResponse, Message


class OpenAIAdapter(Adapter):
    def build_request(self, messages: list[Message], tools: list[dict] | None = None, stream: bool | None = False,
                      **kwargs) -> Any:
        openai_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                pass
            if msg.role=="tool":
                pass


    def call(self) -> Any:
        pass

    def parse_response(self, row_response: Any) -> LLMResponse:
        pass

    def parse_stream_chunk(self, raw_response: Any) -> LLMResponseChunk:
        pass