"""Telegram 网关"""

from typing import Any, Callable
from wuwei.gateway.base import BaseGateway, GatewayMessage


class TelegramGateway(BaseGateway):
    """Telegram 网关

    支持 Telegram Bot API。

    示例：
        async def agent_factory():
            return Agent(llm=llm, tools=tools)

        gateway = TelegramGateway(
            agent_factory=agent_factory,
            bot_token="your-bot-token",
        )
        await gateway.start()
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        bot_token: str,
    ):
        super().__init__(agent_factory)
        self.bot_token = bot_token
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    async def start(self):
        """启动网关"""
        self._running = True

    async def stop(self):
        """停止网关"""
        self._running = False

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息"""
        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError("使用 Telegram 网关需要安装 httpx：pip install httpx")

        async with AsyncClient() as client:
            await client.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": user_id,
                    "text": content,
                },
            )
