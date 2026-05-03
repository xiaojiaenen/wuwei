from wuwei.memory.storage import Storage
from wuwei.runtime.hooks import RuntimeHook


class StorageHook(RuntimeHook):
    """增量持久化：每条消息即时追加到存储，不依赖 context 完整性。"""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def before_llm(self, session, messages, tools, *, step, task=None):
        if step == 0:
            await self.storage.save_meta(session)
            user_msg = session.context.get_last_message()
            if user_msg and user_msg.role == "user":
                await self.storage.append_message(session.session_id, user_msg)
        return messages, tools

    async def after_llm(self, session, response, *, step, task=None):
        await self.storage.append_message(session.session_id, response.message)
        if session.summary:
            await self.storage.save_meta(session)

    async def after_ai_message(self, session, message, *, step, task=None):
        await self.storage.append_message(session.session_id, message)
        if session.summary:
            await self.storage.save_meta(session)

    async def after_tool(self, session, tool_call, tool_message, *, step, task=None, tool=None):
        await self.storage.append_message(session.session_id, tool_message)
