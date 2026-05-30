"""多平台网关模块"""

from wuwei.gateway.base import BaseGateway, GatewayMessage
from wuwei.gateway.webhook import WebhookGateway

__all__ = [
    "BaseGateway",
    "GatewayMessage",
    "WebhookGateway",
]
