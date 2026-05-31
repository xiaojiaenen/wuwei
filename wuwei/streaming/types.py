"""流式模式类型定义"""

from typing import Literal, Any
from pydantic import BaseModel, Field


# 流式模式类型
StreamMode = Literal[
    "values",    # 完整状态
    "updates",   # 节点更新
    "messages",  # LLM 消息
    "custom",    # 自定义数据
    "debug",     # 调试信息
]


class StreamChunk(BaseModel):
    """流式数据块"""
    type: str  # 流式模式类型
    ns: tuple = ()  # 命名空间
    data: Any = None  # 数据内容
    metadata: dict = Field(default_factory=dict)  # 元数据

    class Config:
        arbitrary_types_allowed = True


class ValuesStreamChunk(StreamChunk):
    """完整状态流式数据"""
    type: Literal["values"] = "values"


class UpdatesStreamChunk(StreamChunk):
    """节点更新流式数据"""
    type: Literal["updates"] = "updates"


class MessagesStreamChunk(StreamChunk):
    """LLM 消息流式数据"""
    type: Literal["messages"] = "messages"


class CustomStreamChunk(StreamChunk):
    """自定义流式数据"""
    type: Literal["custom"] = "custom"


class DebugStreamChunk(StreamChunk):
    """调试流式数据"""
    type: Literal["debug"] = "debug"
