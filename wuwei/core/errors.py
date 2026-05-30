"""错误类型定义"""


class WuweiError(Exception):
    """Wuwei 基础异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ToolError(WuweiError):
    """工具执行错误"""
    pass


class LLMError(WuweiError):
    """LLM 调用错误"""
    pass


class TimeoutError(WuweiError):
    """超时错误"""
    pass


class ValidationError(WuweiError):
    """验证错误"""
    pass


class ConfigError(WuweiError):
    """配置错误"""
    pass


class ConnectionError(WuweiError):
    """连接错误"""
    pass
