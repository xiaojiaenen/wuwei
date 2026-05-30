"""Wuwei Core - 核心抽象层

提供：
- Runnable: 统一的可执行接口
- Message: 统一的消息体系
- Types: 基础类型定义
- Errors: 错误类型
"""

from wuwei.core.runnable import Runnable, RunnableSequence, RunnableConfig
from wuwei.core.message import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    ToolCall,
    FunctionCall,
)
from wuwei.core.types import LLMResponse, Usage, ToolResult
from wuwei.core.errors import (
    WuweiError,
    ToolError,
    LLMError,
    TimeoutError,
    ValidationError,
    ConfigError,
    ConnectionError,
)

__all__ = [
    # Runnable
    "Runnable",
    "RunnableSequence",
    "RunnableConfig",
    # Messages
    "BaseMessage",
    "AIMessage",
    "ToolMessage",
    "SystemMessage",
    "HumanMessage",
    "ToolCall",
    "FunctionCall",
    # Types
    "LLMResponse",
    "Usage",
    "ToolResult",
    # Errors
    "WuweiError",
    "ToolError",
    "LLMError",
    "TimeoutError",
    "ValidationError",
    "ConfigError",
    "ConnectionError",
]
