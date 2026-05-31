"""通道基类

借鉴 LangGraph 的 Channel 系统，提供状态管理的抽象。
每个通道负责自身数据的序列化与反序列化。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseChannel(ABC):
    """状态通道基类

    借鉴 LangGraph 的 Channel 系统，提供状态管理的抽象。

    生命周期：
    - update(): 每步接收写入
    - get(): 读取当前值（可能抛出 EmptyChannelError）
    - checkpoint(): 序列化当前值用于持久化
    - from_checkpoint(): 从持久化数据恢复
    - consume(): 标记已被订阅者消费（可选）
    - reset(): 重置通道
    """

    @abstractmethod
    def update(self, value: Any):
        """更新通道值

        Args:
            value: 写入的值
        """
        ...

    @abstractmethod
    def get(self) -> Any:
        """获取通道值

        Returns:
            当前通道值

        Raises:
            EmptyChannelError: 通道为空时抛出
        """
        ...

    def checkpoint(self) -> Any:
        """序列化当前状态用于持久化

        默认实现返回 get() 的结果。
        子类可覆写以提供更高效的序列化。
        """
        try:
            return self.get()
        except EmptyChannelError:
            return None

    def from_checkpoint(self, data: Any) -> None:
        """从持久化数据恢复通道状态

        默认实现调用 update()。
        子类应覆写以正确处理空值标记。
        """
        if data is not None:
            self.reset()
            self.update(data)

    @abstractmethod
    def reset(self):
        """重置通道到初始状态"""
        ...

    def consume(self) -> bool:
        """标记通道已被订阅者消费

        默认空操作。发布-订阅通道应覆写此方法。
        """
        return True

    @property
    def is_updated(self) -> bool:
        """通道自上次消费后是否被更新过"""
        return True


class EmptyChannelError(Exception):
    """通道为空时抛出的异常"""
    pass
