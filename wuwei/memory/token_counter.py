from typing import Protocol

from wuwei import Message, Tool


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_message(self, message: Message) -> int: ...

    def count_messages(self, messages: list[Message]) -> int: ...

    def count_tools(self, tools: list[Tool]) -> int: ...


class SimpleTokenCounter:
    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1,len(text.encode('utf-8'))//4)

    def count_message(self, message: Message) -> int:
        total = 4
        total += self.count_text(message.role)
        total += self.count_text(message.content or "")
        if message.tool_call_id:
            total += self.count_text(message.tool_call_id)
        if message.tool_calls:
            total += self.count_text(message.model_dump_json())
        return total

    def count_messages(self, messages: list[Message]) -> int:
        return sum(self.count_message(message) for message in messages)

    def count_tools(self, tools: list[Tool]) -> int:
        return sum(self.count_text(tool.model_dump_json()) for tool in tools)

