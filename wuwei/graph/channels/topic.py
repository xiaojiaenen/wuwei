"""Topic 通道

借鉴 LangGraph 的 Topic 通道，支持：
- 累积模式：accumulate=True 时保留所有历史值
- 清空模式：accumulate=False 时每次 get() 后清空（pub-sub 语义）
- 支持单值或列表写入：update(value) 或 update([v1, v2])
"""

from typing import Any

from wuwei.graph.channels.base import BaseChannel, EmptyChannelError


class Topic(BaseChannel):
    """Topic 通道

    发布-订阅模式的通道，支持累积或清空两种模式。

    示例：
        # 累积模式
        channel = Topic(str, accumulate=True)
        channel.update("message1")
        channel.update("message2")
        print(channel.get())  # ["message1", "message2"]

        # 清空模式（pub-sub）
        channel = Topic(str, accumulate=False)
        channel.update("event1")
        channel.update("event2")
        print(channel.get())  # ["event1", "event2"]
        print(channel.get())  # []  # 已清空
    """

    def __init__(
        self,
        value_type: type = Any,
        accumulate: bool = True,
    ):
        """
        Args:
            value_type: 值类型（用于类型提示）
            accumulate: True=累积所有值，False=每次 get() 后清空
        """
        self.value_type = value_type
        self.accumulate = accumulate
        self._values: list[Any] = []
        self._updated = False

    def update(self, value: Any):
        """添加一个或多个值

        Args:
            value: 单个值或值列表
        """
        if isinstance(value, list):
            self._values.extend(value)
        else:
            self._values.append(value)
        self._updated = True

    def get(self) -> list[Any]:
        """获取所有值

        在非累积模式下，获取后清空内部缓存。
        """
        if self.accumulate:
            return self._values.copy()
        else:
            values = self._values.copy()
            self._values.clear()
            self._updated = False
            return values

    def checkpoint(self) -> Any:
        """序列化为可持久化的值"""
        return list(self._values)

    def from_checkpoint(self, data: Any) -> None:
        """从持久化数据恢复"""
        self._values = list(data) if data else []
        self._updated = False

    def reset(self):
        """重置通道"""
        self._values.clear()
        self._updated = False

    def consume(self) -> bool:
        """标记已被消费"""
        self._updated = False
        return True

    def get_and_clear(self) -> list[Any]:
        """获取并清空（无论 accumulate 模式）"""
        values = self._values.copy()
        self._values.clear()
        self._updated = False
        return values

    @property
    def is_updated(self) -> bool:
        """是否已更新"""
        return self._updated

    @property
    def count(self) -> int:
        """值的数量"""
        return len(self._values)
