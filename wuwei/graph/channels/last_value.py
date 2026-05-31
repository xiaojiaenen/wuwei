"""LastValue 通道"""

from typing import Any, Optional
from wuwei.graph.channels.base import BaseChannel


class LastValue(BaseChannel):
    """LastValue 通道

    只保留最后一个值，每次更新都会覆盖之前的值。

    示例：
        channel = LastValue(list)
        channel.update([1, 2, 3])
        print(channel.get())  # [1, 2, 3]
        channel.update([4, 5])
        print(channel.get())  # [4, 5]
    """

    def __init__(self, value_type: type = Any):
        """
        Args:
            value_type: 值类型（用于类型提示）
        """
        self.value_type = value_type
        self._value: Any = None
        self._updated = False

    def update(self, value: Any):
        """更新值"""
        self._value = value
        self._updated = True

    def get(self) -> Any:
        """获取值"""
        return self._value

    def reset(self):
        """重置"""
        self._value = None
        self._updated = False

    @property
    def is_updated(self) -> bool:
        """是否已更新"""
        return self._updated
