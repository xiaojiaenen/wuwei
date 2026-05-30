"""微信网关"""

import json
from typing import Any, Callable
from wuwei.gateway.base import BaseGateway, GatewayMessage


class WeChatGateway(BaseGateway):
    """微信网关

    支持企业微信群机器人 Webhook。

    示例：
        async def agent_factory():
            return Agent(llm=llm, tools=tools)

        gateway = WeChatGateway(
            agent_factory=agent_factory,
            webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
        )
        await gateway.start()
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        webhook_url: str,
    ):
        super().__init__(agent_factory)
        self.webhook_url = webhook_url

    async def start(self):
        """启动网关"""
        # Webhook 模式下不需要主动监听
        pass

    async def stop(self):
        """停止网关"""
        pass

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息"""
        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError(
                "使用微信网关需要安装 httpx：\n"
                "pip install httpx"
            )

        async with AsyncClient() as client:
            await client.post(
                self.webhook_url,
                json={
                    "msgtype": "text",
                    "text": {"content": content},
                },
            )
