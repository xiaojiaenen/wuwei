"""通道基类"""

from abc import ABC, abstractmethod
from typing import Any


class BaseChannel(ABC):
    """状态通道基类

    借鉴 LangGraph 的 Channel 系统，提供状态管理的抽象。
    """

    @abstractmethod
    def update(self, value: Any):
        """更新通道值"""
        ...

    @abstractmethod
    def get(self) -> Any:
        """获取通道值"""
        ...

    @abstractmethod
    def reset(self):
        """重置通道"""
        ...
