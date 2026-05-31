"""Topic 通道"""

from typing import Any
from wuwei.graph.channels.base import BaseChannel


class Topic(BaseChannel):
    """Topic 通道

    累积多个值，每次更新都会添加新值。

    示例：
        channel = Topic(str)
        channel.update("message1")
        channel.update("message2")
        print(channel.get())  # ["message1", "message2"]
    """

    def __init__(self, value_type: type = Any):
        """
        Args:
            value_type: 值类型（用于类型提示）
        """
        self.value_type = value_type
        self._values: list[Any] = []

    def update(self, value: Any):
        """添加值"""
        self._values.append(value)

    def get(self) -> list[Any]:
        """获取所有值"""
        return self._values.copy()

    def reset(self):
        """重置"""
        self._values.clear()

    def get_and_clear(self) -> list[Any]:
        """获取并清空"""
        values = self._values.copy()
        self._values.clear()
        return values

    @property
    def count(self) -> int:
        """值的数量"""
        return len(self._values)
