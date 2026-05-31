"""Aggregate 通道"""

from typing import Any, Callable
from wuwei.graph.channels.base import BaseChannel


class Aggregate(BaseChannel):
    """Aggregate 通道

    使用聚合函数合并多个值。

    示例：
        # 求和
        channel = Aggregate(sum)
        channel.update(1)
        channel.update(2)
        channel.update(3)
        print(channel.get())  # 6

        # 求最大值
        channel = Aggregate(max)
        channel.update(1)
        channel.update(5)
        channel.update(3)
        print(channel.get())  # 5
    """

    def __init__(self, aggregator: Callable, initial_value: Any = None):
        """
        Args:
            aggregator: 聚合函数
            initial_value: 初始值
        """
        self.aggregator = aggregator
        self._value = initial_value
        self._values: list[Any] = []

    def update(self, value: Any):
        """添加值并聚合"""
        self._values.append(value)
        self._value = self.aggregator(self._values)

    def get(self) -> Any:
        """获取聚合值"""
        return self._value

    def reset(self):
        """重置"""
        self._value = None
        self._values.clear()

    @property
    def values(self) -> list[Any]:
        """获取所有原始值"""
        return self._values.copy()
