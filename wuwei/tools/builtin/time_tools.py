from datetime import datetime
from zoneinfo import ZoneInfo

from wuwei.tools import ToolRegistry


def register_time_tools(registry: ToolRegistry) -> ToolRegistry:
    @registry.tool(description="获取当前时间")
    def get_now(timezone: str = "Asia/Shanghai") -> dict:
        now = datetime.now(ZoneInfo(timezone))
        return {"timezone": timezone, "iso": now.isoformat()}

    return registry
