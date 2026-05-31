"""状态通道系统"""

from wuwei.graph.channels.base import BaseChannel, EmptyChannelError
from wuwei.graph.channels.last_value import LastValue, InvalidUpdateError
from wuwei.graph.channels.topic import Topic
from wuwei.graph.channels.aggregate import Aggregate
from wuwei.graph.channels.ephemeral_value import EphemeralValue

__all__ = [
    "BaseChannel",
    "EmptyChannelError",
    "LastValue",
    "InvalidUpdateError",
    "Topic",
    "Aggregate",
    "EphemeralValue",
]
