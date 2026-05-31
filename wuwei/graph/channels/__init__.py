"""状态通道系统"""

from wuwei.graph.channels.base import BaseChannel
from wuwei.graph.channels.last_value import LastValue
from wuwei.graph.channels.topic import Topic
from wuwei.graph.channels.aggregate import Aggregate

__all__ = [
    "BaseChannel",
    "LastValue",
    "Topic",
    "Aggregate",
]
