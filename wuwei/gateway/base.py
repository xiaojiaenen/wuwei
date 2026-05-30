"""网关基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Any
import asyncio


@dataclass
class GatewayMessage:
    """统一网关消息格式"""
    platform: str  # 平台标识
    message_id: str  # 平台消息 ID
    user_id: str  # 用户 ID
    user_name: str  # 用户名
    content: str  # 消息内容
    message_type: str = "text"  # text/image/file/...
    reply_to: str | None = None  # 回复的消息 ID
    metadata: dict = field(default_factory=dict)  # 平台特有数据


class BaseGateway(ABC):
    """网关基类

    所有平台网关都继承此类。
    """

    def __init__(self, agent_factory: Callable[[], Any]):
        """
        Args:
            agent_factory: Agent 工厂函数，每次调用返回一个新的 Agent 实例
        """
        self.agent_factory = agent_factory
        self._message_queue: asyncio.Queue = asyncio.Queue()

    @abstractmethod
    async def start(self):
        """启动网关"""
        ...

    @abstractmethod
    async def stop(self):
        """停止网关"""
        ...

    @abstractmethod
    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息"""
        ...

    async def receive_messages(self) -> AsyncIterator[GatewayMessage]:
        """接收消息流"""
        while True:
            message = await self._message_queue.get()
            yield message

    async def handle_message(self, message: GatewayMessage) -> str:
        """处理消息

        默认实现：调用 Agent 处理消息。
        """
        agent = self.agent_factory()
        result = await agent.run(message.content)
        return result

    async def run(self):
        """运行网关

        启动网关并处理所有收到的消息。
        """
        await self.start()

        async for message in self.receive_messages():
            try:
                response = await self.handle_message(message)
                await self.send_message(
                    message.user_id,
                    response,
                    reply_to=message.message_id,
                )
            except Exception as e:
                # 错误处理
                error_msg = f"处理消息时出错: {e}"
                await self.send_message(
                    message.user_id,
                    error_msg,
                    reply_to=message.message_id,
                )
