"""状态定义"""

from dataclasses import dataclass, field
from typing import Any, Optional
from wuwei.core.message import BaseMessage, AIMessage, HumanMessage


@dataclass
class State:
    """图状态

    状态图执行过程中的数据容器。
    可以扩展以包含自定义字段。
    """
    messages: list[BaseMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    step: int = 0

    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)

    def get_last_message(self) -> Optional[BaseMessage]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None

    def get_last_ai_message(self) -> Optional[AIMessage]:
        """获取最后一条 AI 消息"""
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    def get_last_user_message(self) -> Optional[HumanMessage]:
        """获取最后一条用户消息"""
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                return msg
        return None

    def get_tool_messages(self) -> list[BaseMessage]:
        """获取所有工具消息"""
        return [msg for msg in self.messages if msg.role == "tool"]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "messages": [msg.model_dump() for msg in self.messages],
            "metadata": self.metadata,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        """从字典创建"""
        from wuwei.core.message import SystemMessage, ToolMessage

        messages = []
        for msg_data in data.get("messages", []):
            role = msg_data.get("role")
            if role == "system":
                messages.append(SystemMessage(**msg_data))
            elif role == "user":
                messages.append(HumanMessage(**msg_data))
            elif role == "assistant":
                messages.append(AIMessage(**msg_data))
            elif role == "tool":
                messages.append(ToolMessage(**msg_data))

        return cls(
            messages=messages,
            metadata=data.get("metadata", {}),
            step=data.get("step", 0),
        )
