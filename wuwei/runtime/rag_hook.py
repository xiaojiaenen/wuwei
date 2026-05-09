from __future__ import annotations

from typing import TYPE_CHECKING

from wuwei.runtime.hooks import RuntimeHook

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.llm import Message
    from wuwei.memory.knowledge_store import KnowledgeStore
    from wuwei.tools import Tool


class RagRetrievalHook(RuntimeHook):
    """在每次 LLM 调用前，从知识库检索相关片段并注入 system prompt。"""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        top_k: int = 4,
        max_chars_per_chunk: int = 500,
        namespace: str = "default",
    ):
        self.store = store
        self.top_k = top_k
        self.max_chars = max_chars_per_chunk
        self.namespace = namespace

    async def before_llm(
        self,
        session: AgentSession,
        messages: list[Message],
        tools: list[Tool],
        *,
        step: int,
        task=None,
    ):
        user_texts = [
            m.content[:300]
            for m in messages[-4:]
            if m.role == "user" and m.content
        ]
        query = " ".join(user_texts)
        if not query.strip():
            return messages, tools

        chunks = await self.store.search(
            query, namespace=self.namespace, limit=self.top_k
        )
        if not chunks:
            return messages, tools

        lines = ["以下是检索到的参考资料，如不相关可忽略：", ""]
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text[: self.max_chars]
            header = f"[{i}] source={chunk.source}"
            if chunk.title:
                header += f" title={chunk.title}"
            lines.append(header)
            lines.append(text)
            lines.append("")

        injection = "\n".join(lines)

        from wuwei.llm.types import Message as LLMMessage

        rag_msg = LLMMessage(role="system", content=injection)

        new_messages: list[Message] = []
        system_inserted = False
        for msg in messages:
            new_messages.append(msg)
            if msg.role == "system" and not system_inserted:
                new_messages.append(rag_msg)
                system_inserted = True
        if not system_inserted:
            new_messages.insert(0, rag_msg)

        return new_messages, tools
