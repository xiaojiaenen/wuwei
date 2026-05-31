"""EphemeralValue 通道

借鉴 LangGraph 的 EphemeralValue 通道：
- 值只存在一个超步（superstep）
- 每次 update() 后，如果为空序列则自动清除
- 用于一次性信号（条件路由、分支信号等）
"""

from typing import Any

from wuwei.graph.channels.base import BaseChannel, EmptyChannelError


class EphemeralValue(BaseChannel):
    """EphemeralValue 通道

    值只在一个超步内有效。
    - update([]) / update(None) 会清除值
    - update(value) 设置值，下一个超步开始时自动清除
    - 与 LastValue 不同，它不抛出 InvalidUpdateError（允许多次写入）

    示例：
        channel = EphemeralValue(str)
        channel.update("route_to_tools")   # 当前步可见
        print(channel.get())  # "route_to_tools"
        channel.update([])                 # 清除
        print(channel.get())  # EmptyChannelError
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

        空序列或 None 会清除通道。
        """
        # 空序列 / None 表示清除
        if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
            self._value = None
            self._has_value = False
        else:
            self._value = value
            self._has_value = True
        self._updated = True

    def get(self) -> Any:
        """获取当前值

        Raises:
            EmptyChannelError: 通道为空
        """
        if not self._has_value:
            raise EmptyChannelError("EphemeralValue 通道为空")
        return self._value

    def checkpoint(self) -> Any:
        """序列化（短暂值不持久化）"""
        return self._value if self._has_value else None

    def from_checkpoint(self, data: Any) -> None:
        """从持久化恢复"""
        if data is not None:
            self._value = data
            self._has_value = True
        else:
            self._value = None
            self._has_value = False
        self._updated = False

    def reset(self):
        """重置通道"""
        self._value = None
        self._has_value = False
        self._updated = False

    def consume(self) -> bool:
        """消费后自动清除（短暂语义）"""
        self._value = None
        self._has_value = False
        self._updated = False
        return True

    @property
    def is_updated(self) -> bool:
        """是否已更新"""
        return self._updated
