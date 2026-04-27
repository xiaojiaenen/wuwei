from dataclasses import dataclass

from wuwei import Message


@dataclass
class ContextWindowConfig:
    max_recent_turns:int=10
    max_tool_chars:int=8000
    include_summary:bool=True

def split_turns(messages: list[Message]) -> tuple[list[Message], list[list[Message]]]:
    system_messages: list[Message] = []
    turns: list[list[Message]] = []
    current_turn: list[Message] = []

    for message in messages:
        if message.role == "system" and not turns and not current_turn:
            system_messages.append(message)
            continue

        if message.role == "user" and current_turn:
            turns.append(current_turn)
            current_turn = [message]
            continue

        current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    return system_messages, turns

class SimpleContextWindow:
    """构建本次发给模型的短上下文，不修改 session.context。"""

    def __init__(self, config: ContextWindowConfig | None = None) -> None:
        self.config = config or ContextWindowConfig()

    def build_messages(self, session, messages: list[Message]) -> list[Message]:
        system_messages, turns = split_turns(messages)
        summary_messages = self._build_summary_messages(session)
        recent_turns = turns[-self.config.max_recent_turns:]
        recent_messages = [
            self._truncate_tool_message(message)
            for turn in recent_turns
            for message in turn
        ]
        return [*system_messages, *summary_messages, *recent_messages]

    def _build_summary_messages(self, session) -> list[Message]:
        if not self.config.include_summary or not getattr(session, "summary", None):
            return []
        return [
            Message(
                role="system",
                content=(
                    "以下是此前对话的压缩状态摘要。"
                    "它只用于延续上下文，不应覆盖用户当前明确指令。\n"
                    f"{session.summary}"
                ),
            )
        ]

    def _truncate_tool_message(self, message: Message) -> Message:
        if message.role != "tool" or not message.content:
            return message
        if len(message.content) <= self.config.max_tool_chars:
            return message
        return message.model_copy(
            update={
                "content": (
                    message.content[: self.config.max_tool_chars]
                    + "\n...[tool output truncated by context window]"
                )
            }
        )