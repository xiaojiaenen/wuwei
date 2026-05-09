from __future__ import annotations

import json
from typing import TYPE_CHECKING

from wuwei.runtime.hooks import RuntimeHook

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.llm import Message
    from wuwei.memory.memory_store import MemoryStore
    from wuwei.tools import Tool

_EXTRACTION_PROMPT = """\
分析以下对话，提取值得长期记住的信息。

只提取以下类型的信息：
- preference: 用户明确表达的偏好
- fact: 已确认的稳定事实
- constraint: 长期约束
- summary: 对话中达成的重要结论

不要提取：
- 临时任务、一次性问答
- 模型的猜测或不确定的信息
- 已经是常识的信息

输出 JSON 数组，每条包含 type 和 content。最多 3 条。
如果没有值得记住的信息，输出空数组 []。

对话内容：
{conversation}

输出（纯 JSON，不要 markdown）："""


class MemoryRetrievalHook(RuntimeHook):
    """在每次 LLM 调用前，检索相关长期记忆并注入 system prompt。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        top_k: int = 3,
        namespace: str = "default",
    ):
        self.store = store
        self.top_k = top_k
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
        recent_texts = []
        for msg in messages[-6:]:
            if msg.role in ("user", "assistant") and msg.content:
                recent_texts.append(msg.content[:200])
        query = " ".join(recent_texts)
        if not query.strip():
            return messages, tools

        records = await self.store.search(
            query, namespace=self.namespace, limit=self.top_k
        )
        if not records:
            return messages, tools

        lines = ["以下是可能相关的长期记忆，仅在与当前任务相关时使用："]
        for r in records:
            lines.append(f"- [{r.memory_type}] {r.content}")
        injection = "\n".join(lines)

        from wuwei.llm.types import Message as LLMMessage

        memory_msg = LLMMessage(role="system", content=injection)

        new_messages: list[Message] = []
        system_inserted = False
        for msg in messages:
            new_messages.append(msg)
            if msg.role == "system" and not system_inserted:
                new_messages.append(memory_msg)
                system_inserted = True
        if not system_inserted:
            new_messages.insert(0, memory_msg)

        return new_messages, tools


class MemoryExtractionHook(RuntimeHook):
    """在一轮运行结束后，用 LLM 从对话中抽取值得记住的记忆。"""

    def __init__(
        self,
        llm,
        store: MemoryStore,
        *,
        namespace: str = "default",
        max_memories_per_run: int = 3,
    ):
        self.llm = llm
        self.store = store
        self.namespace = namespace
        self.max_memories = max_memories_per_run

    async def on_run_end(self, session, result, *, task=None) -> None:
        messages = session.context.get_messages()
        recent = messages[-10:]
        conversation = "\n".join(
            f"{m.role}: {m.content[:300]}" for m in recent if m.content
        )
        if len(conversation) < 50:
            return

        prompt = _EXTRACTION_PROMPT.format(conversation=conversation)
        from wuwei.llm.types import Message as LLMMessage

        response = await self.llm.generate(
            [LLMMessage(role="user", content=prompt)]
        )

        try:
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            items = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            return

        if not isinstance(items, list):
            return

        for item in items[: self.max_memories]:
            if not isinstance(item, dict) or "content" not in item:
                continue
            content = item["content"].strip()
            if not content:
                continue

            existing = await self.store.search(
                content, namespace=self.namespace, limit=1
            )
            if existing and existing[0].content.lower() == content.lower():
                continue

            await self.store.add(
                content=content,
                namespace=self.namespace,
                memory_type=item.get("type", "fact"),
                importance=item.get("importance", 0.5),
                confidence=0.7,
            )
