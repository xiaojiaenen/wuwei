"""输出解析器基类"""

from abc import ABC, abstractmethod
from typing import Any


class BaseOutputParser(ABC):
    """输出解析器基类

    所有输出解析器都继承此类。
    """

    @abstractmethod
    def parse(self, output: str) -> Any:
        """解析 LLM 输出"""
        ...

    def get_format_instructions(self) -> str:
        """获取格式说明（可选）"""
        return ""

    def __or__(self, other: "BaseOutputParser") -> "CompositeOutputParser":
        """支持 | 操作符组合多个解析器"""
        return CompositeOutputParser(self, other)


class CompositeOutputParser(BaseOutputParser):
    """组合输出解析器"""

    def __init__(self, *parsers: BaseOutputParser):
        self.parsers = parsers

    def parse(self, output: str) -> Any:
        """依次执行所有解析器"""
        result = output
        for parser in self.parsers:
            result = parser.parse(result)
        return result
