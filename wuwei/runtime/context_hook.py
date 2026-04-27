from wuwei import Message
from wuwei.memory.context_compressor import ContextCompressor
from wuwei.memory.context_window import SimpleContextWindow, ContextWindowConfig, split_turns
from wuwei.runtime import RuntimeHook


class ContextCompressionHook(RuntimeHook):
    """历史 turn 过多时生成滚动摘要。"""

    METADATA_KEY = "_context_compressed_until_turn"

    def __init__(
        self,
        compressor: ContextCompressor,
        context_window: SimpleContextWindow | None = None,
        *,
        compress_after_turns: int = 30,
        keep_recent_turns: int = 10,
    ) -> None:
        if keep_recent_turns >= compress_after_turns:
            raise ValueError("keep_recent_turns 必须小于 compress_after_turns")

        self.compressor = compressor
        self.compress_after_turns = compress_after_turns
        self.keep_recent_turns = keep_recent_turns
        self.context_window = context_window or SimpleContextWindow(
            ContextWindowConfig(max_recent_turns=keep_recent_turns)
        )

    async def before_llm(self, session, messages, tools, *, step: int, task=None):
        _, turns = split_turns(messages)

        if len(turns) > self.compress_after_turns:
            await self._compress_old_turns(session, turns)

        return self._build_window(session, messages, tools)

    async def _compress_old_turns(self, session, turns: list[list[Message]]) -> None:
        metadata = getattr(session, "metadata", None)
        if metadata is None:
            metadata = {}
            session.metadata = metadata

        compress_until = len(turns) - self.keep_recent_turns
        compressed_until = metadata.get(self.METADATA_KEY, 0)

        if compressed_until >= compress_until:
            return

        turns_to_compress = turns[compressed_until:compress_until]
        session.summary = await self.compressor.compress(
            previous_summary=getattr(session, "summary", None),
            messages=self._flatten(turns_to_compress),
        )
        metadata[self.METADATA_KEY] = compress_until

    def _build_window(self, session, messages, tools):
        windowed_messages = self.context_window.build_messages(session, messages)
        return windowed_messages, tools

    def _flatten(self, turns: list[list[Message]]) -> list[Message]:
        return [message for turn in turns for message in turn]