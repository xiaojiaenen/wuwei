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
    ) -> Message:
        """追加一条 assistant 消息。"""
        message = Message(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
        self._messages.append(message)
        return message

    def get_messages(self) -> list[Message]:
        """返回完整消息历史。"""
        return self._messages

    def get_last_message(self) -> Message | None:
        """返回最后一条消息。"""
        return self._messages[-1] if self._messages else None

    def reset(self) -> None:
        """清空上下文。"""
        self._messages.clear()

    def keep_last_turns(self, n: int) -> None:
        """只保留 system 消息和最近 n 轮对话，旧消息从内存中移除。"""
        from wuwei.memory.context_window import split_turns

        system_msgs, turns = split_turns(self._messages)
        if len(turns) <= n:
            return
        kept = list(system_msgs)
        for turn in turns[-n:]:
            kept.extend(turn)
        self._messages = kept

    def to_dict(self) -> dict:
        """序列化为 dict。"""
        return {"messages": [m.model_dump(exclude_none=True) for m in self._messages]}

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """从 dict 反序列化。"""
        ctx = cls()
        for m in data.get("messages", []):
            ctx._messages.append(Message.model_validate(m))
        return ctx
