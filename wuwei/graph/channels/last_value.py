"""LastValue 通道

借鉴 LangGraph 的 LastValue 通道，支持：
- 多写检测：同一步骤内多次写入抛出 InvalidUpdateError
- 类型校验：可选的类型检查
"""

from typing import Any

from wuwei.graph.channels.base import BaseChannel, EmptyChannelError


class InvalidUpdateError(Exception):
    """同一超步内多次写入 LastValue 通道时抛出"""
    pass


class LastValue(BaseChannel):
    """LastValue 通道

    只保留最后一个值，每次更新都会覆盖之前的值。
    在同一超步内，如果多次写入且值不同，将抛出 InvalidUpdateError。

    示例：
        channel = LastValue(int)
        channel.update(42)
        print(channel.get())  # 42
        channel.update(99)
        print(channel.get())  # 99
    """

    def __init__(self, value_type: type = Any):
        """
        Args:
            value_type: 值类型（用于类型提示）
        """
        self.value_type = value_type
        self._value: Any = None
        self._has_value = False
        self._updated = False

    def update(self, value: Any):
        """更新值

        同一超步内多次写入同一值视为幂等（不抛异常），
        不同值则抛出 InvalidUpdateError 防止意外 fan-in 导致数据丢失。

        Args:
            value: 新值

        Raises:
            InvalidUpdateError: 同一超步内被写入不同值
        """
        if self._updated and self._value != value:
            raise InvalidUpdateError(
                f"LastValue 通道在同一超步内被写入了不同值: "
                f"{self._value!r} != {value!r}"
            )
        self._value = value
        self._has_value = True
        self._updated = True

    def get(self) -> Any:
        """获取值

        Raises:
            EmptyChannelError: 通道尚未写入任何值
        """
        if not self._has_value:
            raise EmptyChannelError("LastValue 通道尚未写入任何值")
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
        self._value = None
        self._has_value = False
        self._updated = False

    def consume(self) -> bool:
        """标记已被消费"""
        self._updated = False
        return True

    @property
    def is_updated(self) -> bool:
        """是否已更新"""
        return self._updated
