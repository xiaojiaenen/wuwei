from wuwei.llm import Message, ToolCall


class Context:
    """维护单个会话的消息历史。"""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_user_message(self, content: str) -> None:
        """追加一条用户消息。"""
        self._messages.append(Message(role="user", content=content))

    def add_system_message(self, content: str) -> None:
        """追加一条 system 消息。"""
        self._messages.append(Message(role="system", content=content))

    def add_tool_message(self, content: str, tool_call_id: str | None) -> None:
        """追加一条工具返回消息。"""
        self._messages.append(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def add_ai_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """追加一条 assistant 消息。"""
        self._messages.append(
            Message(
                role="assistant",
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )
        )

    def get_messages(self) -> list[Message]:
        """返回完整消息历史。"""
        return self._messages

    def get_last_message(self) -> Message | None:
        """返回最后一条消息。"""
        return self._messages[-1] if self._messages else None

    def reset(self) -> None:
        """清空上下文。"""
        self._messages.clear()
