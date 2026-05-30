"""通用 Webhook 网关"""

import json
import uuid
from typing import Any, Callable
from wuwei.gateway.base import BaseGateway, GatewayMessage


class WebhookGateway(BaseGateway):
    """通用 Webhook 网关

    支持任何 HTTP 平台，通过 Webhook 接收和发送消息。

    示例：
        async def agent_factory():
            return Agent(llm=llm, tools=tools)

        gateway = WebhookGateway(
            agent_factory=agent_factory,
            host="0.0.0.0",
            port=8080,
        )
        await gateway.start()
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        super().__init__(agent_factory)
        self.host = host
        self.port = port
        self._app = None

    async def start(self):
        """启动 Webhook 服务器"""
        try:
            from fastapi import FastAPI, Request
            import uvicorn
        except ImportError:
            raise ImportError(
                "使用 Webhook 网关需要安装 fastapi 和 uvicorn：\n"
                "pip install fastapi uvicorn"
            )

        self._app = FastAPI()

        @self._app.post("/webhook")
        async def handle_webhook(request: Request):
            """处理 Webhook 请求"""
            body = await request.json()

            message = GatewayMessage(
                platform="webhook",
                message_id=body.get("message_id", str(uuid.uuid4())),
                user_id=body.get("user_id", "anonymous"),
                user_name=body.get("user_name", ""),
                content=body.get("content", ""),
                message_type=body.get("type", "text"),
                metadata=body,
            )

            # 放入消息队列
            await self._message_queue.put(message)

            # 处理消息
            response = await self.handle_message(message)

            return {"content": response}

        @self._app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "ok"}

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        """停止网关"""
        pass

    async def send_message(self, user_id: str, content: str, **kwargs):
        """发送消息

        Webhook 网关通过 HTTP 响应返回，不主动推送。
        """
        # Webhook 模式下，消息通过 HTTP 响应返回
        pass
