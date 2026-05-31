"""存储中间件"""

import json
import os
from datetime import datetime
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import AIMessage, ToolMessage


class StorageMiddleware(Middleware):
    """存储中间件

    增量持久化每条消息到文件。

    示例：
        middleware = StorageMiddleware(storage_path=".wuwei/sessions")
    """

    def __init__(self, storage_path: str = ".wuwei/sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """LLM 调用后保存消息"""
        session_id = ctx.metadata.get("session_id", "default")
        session_dir = os.path.join(self.storage_path, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 保存 AI 消息
        message_file = os.path.join(
            session_dir,
            f"message_{datetime.now().isoformat()}.json",
        )
        with open(message_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        tc.model_dump() for tc in response.tool_calls
                    ] if response.tool_calls else [],
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return ctx

    async def after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """工具执行后保存结果"""
        session_id = ctx.metadata.get("session_id", "default")
        session_dir = os.path.join(self.storage_path, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 保存工具消息
        message_file = os.path.join(
            session_dir,
            f"tool_{datetime.now().isoformat()}.json",
        )
        with open(message_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "role": "tool",
                    "content": tool_message.content,
                    "tool_call_id": tool_message.tool_call_id,
                    "name": tool_message.name,
                    "status": tool_message.status,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return tool_message
