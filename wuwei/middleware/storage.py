"""存储中间件

增量持久化每条消息到文件。

v3.1 修复：
- 文件 IO 用 asyncio.to_thread 包裹，不再阻塞事件循环
- 文件名加毫秒时间戳 + 短随机后缀，消除同秒覆盖
- session_id 优先从 ctx.state.metadata 读取（runner 在那里写入）
"""

import asyncio
import json
import os
from datetime import datetime
from uuid import uuid4

from wuwei.core.message import AIMessage, ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext


class StorageMiddleware(Middleware):
    """存储中间件

    示例：
        middleware = StorageMiddleware(storage_path=".wuwei/sessions")
    """

    def __init__(self, storage_path: str = ".wuwei/sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def _session_dir(self, ctx: MiddlewareContext) -> tuple[str, str]:
        """解析 session 目录；session_id 依次从 ctx.state.metadata / ctx.metadata 读取。"""
        session_id = "default"
        state_meta = getattr(ctx.state, "metadata", None)
        if isinstance(state_meta, dict):
            session_id = state_meta.get("session_id") or session_id
        session_id = ctx.metadata.get("session_id") or session_id
        session_dir = os.path.join(self.storage_path, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir, session_id

    @staticmethod
    def _message_file(session_dir: str, prefix: str) -> str:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
        return os.path.join(session_dir, f"{prefix}_{stamp}_{uuid4().hex[:6]}.json")

    @staticmethod
    async def _write_json(path: str, payload: dict) -> None:
        def _write() -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_write)

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """LLM 调用后保存消息"""
        session_dir, _ = self._session_dir(ctx)
        await self._write_json(
            self._message_file(session_dir, "message"),
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    tc.model_dump() for tc in response.tool_calls
                ] if response.tool_calls else [],
                "timestamp": datetime.now().isoformat(),
            },
        )
        return ctx

    async def after_tool(
        self,
        ctx: MiddlewareContext,
        tool_message: ToolMessage,
    ) -> ToolMessage:
        """工具执行后保存结果"""
        session_dir, _ = self._session_dir(ctx)
        await self._write_json(
            self._message_file(session_dir, "tool"),
            {
                "role": "tool",
                "content": tool_message.content,
                "tool_call_id": tool_message.tool_call_id,
                "name": tool_message.name,
                "status": tool_message.status,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return tool_message
