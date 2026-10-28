"""记忆中间件

结合记忆检索和记忆提取功能。
注入只作用于本轮 LLM 调用的消息视图，不写入会话历史，因此不会堆积。
"""

import json

from wuwei.core.message import AIMessage, BaseMessage, SystemMessage
from wuwei.llm.gateway import LLMGateway
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.middleware.rag import _injection_position


class MemoryMiddleware(Middleware):
    """记忆中间件

    示例：
        from wuwei.memory import InMemoryMemoryStore

        memory_store = InMemoryMemoryStore()
        middleware = MemoryMiddleware(llm=llm, memory_store=memory_store)
    """

    EXTRACTION_PROMPT = """从以下对话中提取用户的关键信息，用于后续个性化服务。

只提取以下类型的信息：
- 用户偏好（喜欢的主题、风格、习惯）
- 重要事实（职业、项目、需求）
- 明确指令（"我总是要用xx主题"、"我不喜欢xx"）

忽略：
- 临时性对话内容
- 工具调用细节
- 不重要的寒暄

输出 JSON 数组，每条包含 content（事实描述）和 importance（0.0-1.0）。
如果没有值得记忆的信息，输出空数组 []。

对话内容：
{conversation}

输出（纯 JSON，不要代码块）："""

    def __init__(
        self,
        llm: LLMGateway,
        memory_store,
        top_k: int = 5,
        min_importance: float = 0.3,
        extract_every_n: int = 5,
        namespace: str = "default",
    ):
        """
        Args:
            llm: LLM 网关
            memory_store: 记忆存储（实现 search(query, limit=...) 协议）
            top_k: 检索的最大记忆数
            min_importance: 最小重要性阈值
            extract_every_n: 每 N 次对话提取一次记忆（避免过度调用 LLM）
            namespace: 记忆命名空间
        """
        self.llm = llm
        self.memory_store = memory_store
        self.top_k = top_k
        self.min_importance = min_importance
        self.extract_every_n = extract_every_n
        self.namespace = namespace
        if extract_every_n <= 0:
            raise ValueError("extract_every_n must be positive")

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前检索相关记忆"""
        if not ctx.state or not ctx.state.messages:
            return ctx

        last_user_msg = None
        for msg in reversed(ctx.state.messages):
            if msg.role == "user":
                last_user_msg = msg
                break
        if not last_user_msg or not last_user_msg.content:
            return ctx

        memories = await self.memory_store.search(
            last_user_msg.content,
            namespace=self.namespace,
            limit=self.top_k,
        )
        if not memories:
            return ctx

        memories = [m for m in memories if m.importance >= self.min_importance]
        if not memories:
            return ctx

        memory_text = "\n".join(f"- {m.content}" for m in memories)
        ctx.state.messages.insert(
            _injection_position(ctx.state.messages),
            SystemMessage(content=f"以下是关于用户的已知记忆，供回答时参考：\n{memory_text}"),
        )
        return ctx

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """LLM 调用后提取值得记忆的信息（每 N 次对话提取一次）"""
        if not ctx.state or not ctx.state.messages:
            return ctx

        session = ctx.config.get("session")
        owner = (
            session.metadata.setdefault("middleware_state", {})
            if session is not None
            else ctx.metadata
        )
        memory_state = owner.setdefault(
            f"memory_extraction:{self.namespace}:{self.extract_every_n}", {}
        )
        count = memory_state["turn_count"] = memory_state.get("turn_count", 0) + 1
        if count % self.extract_every_n != 0:
            return ctx

        recent_messages = ctx.state.messages[-5:]
        conversation_parts = [
            f"{msg.role}: {msg.content}"
            for msg in recent_messages
            if msg.role in ("user", "assistant") and msg.content
        ]
        if not conversation_parts:
            return ctx

        conversation = "\n".join(conversation_parts)
        prompt = self.EXTRACTION_PROMPT.format(conversation=conversation)

        try:
            messages = [BaseMessage(role="user", content=prompt)]
            model_call = ctx.config.get("model_call")
            extraction = (
                await model_call(messages, tools=None)
                if model_call
                else await self.llm.generate(messages=messages)
            )

            content = extraction.message.content or ""
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            memories = json.loads(content.strip())
            if not isinstance(memories, list):
                return ctx

            for memory in memories:
                if isinstance(memory, dict) and memory.get("content"):
                    await self.memory_store.add(
                        memory["content"],
                        namespace=self.namespace,
                        importance=memory.get("importance", 0.5),
                    )
        except Exception as exc:
            from wuwei.middleware.budget import BudgetExhausted

            if isinstance(exc, BudgetExhausted):
                raise
            # Optional extraction failures do not abort an otherwise valid response.
            pass

        return ctx
