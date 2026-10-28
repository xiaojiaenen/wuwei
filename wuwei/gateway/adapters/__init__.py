"""网关适配器"""

from wuwei.gateway.adapters.dingtalk import DingTalkGateway
from wuwei.gateway.adapters.feishu import FeishuGateway
from wuwei.gateway.adapters.telegram import TelegramGateway
from wuwei.gateway.adapters.wechat import WeChatGateway

__all__ = [
    "WeChatGateway",
    "DingTalkGateway",
    "FeishuGateway",
    "TelegramGateway",
]
