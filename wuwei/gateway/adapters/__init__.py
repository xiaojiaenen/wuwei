"""网关适配器"""

from wuwei.gateway.adapters.wechat import WeChatGateway
from wuwei.gateway.adapters.dingtalk import DingTalkGateway
from wuwei.gateway.adapters.feishu import FeishuGateway
from wuwei.gateway.adapters.telegram import TelegramGateway

__all__ = [
    "WeChatGateway",
    "DingTalkGateway",
    "FeishuGateway",
    "TelegramGateway",
]
