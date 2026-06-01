"""记忆中间件"""

from typing import Optional
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import AIMessage, BaseMessage, SystemMessage
from wuwei.llm.gateway import LLMGateway


class MemoryMiddleware(Middleware):
    """记忆中间件

    结合记忆检索和记忆提取功能。

    示例：
        from wuwei.memory import InMemoryMemoryStore

        memory_store = InMemoryMemoryStore()
        middleware = MemoryMiddleware(
            llm=llm,
            memory_store=memory_store,
        )
    """

    # 记忆提取提示词模板
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
            memory_store: 记忆存储
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
        self._turn_count = 0

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前检索相关记忆"""
        if not ctx.state or not ctx.state.messages:
            return ctx

        # 获取最后一条用户消息
        last_user_msg = None
        for msg in reversed(ctx.state.messages):
            if msg.role == "user":
                last_user_msg = msg
                break

        if not last_user_msg:
            return ctx

        # 搜索相关记忆
        memories = await self.memory_store.search(
            last_user_msg.content,
            top_k=self.top_k,
        )

        if memories:
            # 过滤低重要性记忆
            memories = [
                m for m in memories
                if m.importance >= self.min_importance
            ]

            if memories:
                # 构建记忆文本
                memory_texts = []
                for m in memories:
                    memory_texts.append(f"- {m.content}")

                memory_text = "\n".join(memory_texts)

                # 注入到消息开头
                ctx.state.messages.insert(
                    0,
                    SystemMessage(content=f"相关记忆：\n{memory_text}"),
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

        self._turn_count += 1
        if self._turn_count % self.extract_every_n != 0:
            return ctx

        # 获取最近的对话
        recent_messages = ctx.state.messages[-5:]
        conversation_parts = []
        for msg in recent_messages:
            if msg.role in ("user", "assistant"):
                conversation_parts.append(f"{msg.role}: {msg.content}")

        if not conversation_parts:
            return ctx

        conversation = "\n".join(conversation_parts)
        prompt = self.EXTRACTION_PROMPT.format(conversation=conversation)

        try:
            response = await self.llm.generate(
                messages=[BaseMessage(role="user", content=prompt)],
            )

            import json
            content = response.message.content or ""
            # 提取 JSON
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

        except Exception:
            # 记忆提取失败不影响主流程
            pass

        return ctx
