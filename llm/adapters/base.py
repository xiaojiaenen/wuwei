from abc import ABC,abstractmethod
from typing import Any

from llm.types import Message, LLMResponse,LLMResponseChunk


class Adapter(ABC):
    @abstractmethod
    def build_request(self,
                      messages:list[Message],
                      tools:list[dict]|None=None,
                      stream:bool|None=False,
                        **kwargs)-> Any:
        pass

    @abstractmethod
    def call(self,request:Any)-> Any:
        pass

    @abstractmethod
    def parse_response(self,raw_response:Any)->LLMResponse:
        pass

    @abstractmethod
    def parse_stream_chunk(self,chunk:Any)->LLMResponseChunk:
        pass