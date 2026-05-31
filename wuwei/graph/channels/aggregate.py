"""Aggregate 通道

借鉴 LangGraph 的 BinaryOperatorAggregate，支持：
- 二元归约器模式：binary_op(current, update) 而非全量重算
- 每步只应用增量更新，效率更高
"""

from typing import Any, Callable

from wuwei.graph.channels.base import BaseChannel, EmptyChannelError


class Aggregate(BaseChannel):
    """Aggregate 通道

    使用二元归约函数逐步合并值。与全量聚合不同，
    每次 update() 直接用 binary_op(当前值, 新值) 更新，
    无需保留所有原始值。

    示例：
        # 求和（二元操作）
        channel = Aggregate(lambda a, b: a + b, initial_value=0)
        channel.update(1)  # 0 + 1 = 1
        channel.update(2)  # 1 + 2 = 3
        channel.update(3)  # 3 + 3 = 6
        print(channel.get())  # 6

        # 列表合并（Annotated[list, add] 模式）
        channel = Aggregate(lambda a, b: a + b, initial_value=[])
        channel.update([1, 2])
        channel.update([3, 4])
        print(channel.get())  # [1, 2, 3, 4]
    """

    def __init__(
        self,
        binary_op: Callable[[Any, Any], Any],
        initial_value: Any = None,
    ):
        """
        Args:
            binary_op: 二元归约函数 (accumulated, new_value) -> new_accumulated
            initial_value: 初始值
        """
        self.binary_op = binary_op
        self._initial_value = initial_value
        self._value = initial_value
        self._has_value = initial_value is not None
        self._updated = False

    def update(self, value: Any):
        """应用增量更新

        使用 binary_op(current_value, new_value) 合并。
        """
        if not self._has_value:
            self._value = value
            self._has_value = True
        else:
            self._value = self.binary_op(self._value, value)
        self._updated = True

    def get(self) -> Any:
        """获取聚合值

        Raises:
            EmptyChannelError: 通道尚未写入任何值
        """
        if not self._has_value:
            raise EmptyChannelError("Aggregate 通道尚未写入任何值")
        return self._value

    def checkpoint(self) -> Any:
        """序列化为可持久化的值"""
        return self._value

    def from_checkpoint(self, data: Any) -> None:
        """从持久化数据恢复"""
        self._value = data
        self._has_value = data is not None
        self._updated = False

    def reset(self):
        """重置通道"""
        self._value = self._initial_value
        self._has_value = self._initial_value is not None
        self._updated = False

    def consume(self) -> bool:
        """标记已被消费"""
        self._updated = False
        return True

    @property
    def is_updated(self) -> bool:
        """是否已更新"""
        return self._updated
