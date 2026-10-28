"""Wuwei Core - 核心抽象层

提供：
- Runnable: 统一的可执行接口
- Message: 统一的消息体系
- Types: 基础类型定义
- Errors: 错误类型
"""

from wuwei.core.errors import (
    ConfigError,
    ConnectionError,
    LLMError,
    TimeoutError,
    ToolError,
    ValidationError,
    WuweiError,
)
from wuwei.core.message import (
    AIMessage,
    BaseMessage,
    FunctionCall,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from wuwei.core.runnable import Runnable, RunnableConfig, RunnableSequence
from wuwei.core.types import LLMResponse, ToolResult, Usage

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
