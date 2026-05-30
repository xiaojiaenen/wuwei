"""Runnable 接口 - 统一的可执行接口

借鉴 LangChain 的 Runnable 接口，但更轻量。
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass


@dataclass
class RunnableConfig:
    """Runnable 配置"""
    tags: list[str] = None
    metadata: dict = None
    callbacks: list = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.callbacks is None:
            self.callbacks = []


class Runnable(ABC):
    """统一的可执行接口

    所有可执行组件都实现此接口，支持：
    - invoke: 同步执行
    - stream: 流式执行
    - | : 管道操作符组合
    """

    @abstractmethod
    async def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        """执行 runnable

        Args:
            input: 输入数据
            config: 可选配置

        Returns:
            执行结果
        """
        ...

    async def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Any]:
        """流式执行

        默认实现：调用 invoke 并 yield 结果。
        子类可以覆盖此方法以实现真正的流式输出。

        Args:
            input: 输入数据
            config: 可选配置

        Yields:
            执行结果片段
        """
        result = await self.invoke(input, config)
        yield result

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """管道操作符：self | other

        示例：
            chain = prompt | llm | output_parser
        """
        return RunnableSequence(self, other)

    def __ror__(self, other: "Runnable") -> "RunnableSequence":
        """反向管道操作符：other | self"""
        return RunnableSequence(other, self)


class RunnableSequence(Runnable):
    """管道序列：按顺序执行多个 Runnable

    示例：
        chain = RunnableSequence(prompt, llm, output_parser)
        result = await chain.invoke("hello")
    """

    def __init__(self, *runnables: Runnable):
        if len(runnables) < 1:
            raise ValueError("RunnableSequence 至少需要一个 Runnable")
        self.runnables = list(runnables)

    async def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        """按顺序执行所有 Runnable"""
        result = input
        for runnable in self.runnables:
            result = await runnable.invoke(result, config)
        return result

    async def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Any]:
        """流式执行管道"""
        result = input
        for i, runnable in enumerate(self.runnables):
            if i == len(self.runnables) - 1:
                # 最后一个 Runnable 使用流式
                async for chunk in runnable.stream(result, config):
                    yield chunk
            else:
                result = await runnable.invoke(result, config)

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """支持链式组合：a | b | c"""
        if isinstance(other, RunnableSequence):
            return RunnableSequence(*self.runnables, *other.runnables)
        return RunnableSequence(*self.runnables, other)

    def __repr__(self) -> str:
        runnable_names = " | ".join(
            r.__class__.__name__ for r in self.runnables
        )
        return f"RunnableSequence({runnable_names})"
