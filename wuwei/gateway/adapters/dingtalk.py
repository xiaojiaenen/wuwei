"""钉钉网关"""

import json
import time
from typing import Any, Callable
from wuwei.gateway.base import BaseGateway, GatewayMessage


class DingTalkGateway(BaseGateway):
    """钉钉网关

    支持钉钉机器人和工作通知。

    示例：
        async def agent_factory():
            return Agent(llm=llm, tools=tools)

        gateway = DingTalkGateway(
            agent_factory=agent_factory,
            app_key="your-app-key",
            app_secret="your-app-secret",
            robot_code="your-robot-code",
        )
        await gateway.start()
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        app_key: str,
        app_secret: str,
        robot_code: str = None,
    ):
        super().__init__(agent_factory)
        self.app_key = app_key
        self.app_secret = app_secret
        self.robot_code = robot_code
        self._access_token = None
        self._token_expires = 0

    async def _get_access_token(self) -> str:
        """获取钉钉 Access Token"""
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError("使用钉钉网关需要安装 httpx：pip install httpx")

        async with AsyncClient() as client:
            resp = await client.post(
                "https://api.dingtalk.com/v1.0/oauth2/accessToken",
                json={
                    "appKey": self.app_key,
                    "appSecret": self.app_secret,
                },
            )
            data = resp.json()
            self._access_token = data["accessToken"]
            self._token_expires = time.time() + data["expireIn"] - 60
            return self._access_token

    async def start(self):
        """启动网关 — 初始化 token 并标记为运行中"""
        await self._get_access_token()
        self._running = True

    async def stop(self):
        """停止网关 — 清理资源"""
        self._running = False
        self._access_token = None
        self._token_expires = 0

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息"""
        try:
            from httpx import AsyncClient
        except ImportError:
            raise ImportError("使用钉钉网关需要安装 httpx：pip install httpx")

        token = await self._get_access_token()

        async with AsyncClient() as client:
            await client.post(
                "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend",
                headers={"x-acs-dingtalk-access-token": token},
                json={
                    "robotCode": self.robot_code,
                    "userIds": [user_id],
                    "msgKey": "sampleText",
                    "msgParam": json.dumps({"content": content}),
                },
            )
