"""飞书网关"""

import json
from typing import Any, Callable
from wuwei.gateway.base import BaseGateway, GatewayMessage


class FeishuGateway(BaseGateway):
    """飞书网关

    支持飞书机器人和应用消息。

    示例：
        async def agent_factory():
            return Agent(llm=llm, tools=tools)

        gateway = FeishuGateway(
            agent_factory=agent_factory,
            app_id="your-app-id",
            app_secret="your-app-secret",
        )
        await gateway.start()
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        app_id: str,
        app_secret: str,
        verification_token: str = None,
    ):
        super().__init__(agent_factory)
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self._tenant_access_token = None

    async def _get_tenant_access_token(self) -> str:
        """获取飞书 Tenant Access Token"""
        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError("使用飞书网关需要安装 httpx：pip install httpx")

        async with AsyncClient() as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.app_id,
                    "app_secret": self.app_secret,
                },
            )
            return resp.json()["tenant_access_token"]

    async def start(self):
        """启动网关"""
        pass

    async def stop(self):
        """停止网关"""
        pass

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息"""
        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError("使用飞书网关需要安装 httpx：pip install httpx")

        token = await self._get_tenant_access_token()

        async with AsyncClient() as client:
            await client.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                params={"receive_id_type": "open_id"},
                json={
                    "receive_id": user_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": content}),
                },
            )
