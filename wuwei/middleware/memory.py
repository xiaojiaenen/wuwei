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

    def __init__(
        self,
        llm: LLMGateway,
        memory_store,
        top_k: int = 5,
        min_importance: float = 0.3,
    ):
        """
        Args:
            llm: LLM 网关
            memory_store: 记忆存储
            top_k: 检索的最大记忆数
            min_importance: 最小重要性阈值
        """
        self.llm = llm
        self.memory_store = memory_store
        self.top_k = top_k
        self.min_importance = min_importance

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
        """LLM 调用后提取值得记忆的信息"""
        if not ctx.state or not ctx.state.messages:
            return ctx

        # 获取最近的对话
        recent_messages = ctx.state.messages[-5:]

        # 使用 LLM 提取记忆
        prompt = """分析以下对话，提取值得记住的信息（用户偏好、关键决策、重要事实）。
如果没有什么值得记住的，返回空列表。

对话：
"""
        for msg in recent_messages:
            if msg.role in ("user", "assistant"):
                prompt += f"{msg.role}: {msg.content}\n"

        prompt += "\n请以 JSON 格式返回值得记住的信息列表，每条包含 content 和 importance 字段。"

        try:
            response = await self.llm.generate(
                messages=[BaseMessage(role="user", content=prompt)],
            )

            # 解析 LLM 响应
            import json
            try:
                # 尝试从响应中提取 JSON
                start = response.content.find("[")
                end = response.content.rfind("]") + 1
                if start != -1 and end != 0:
                    memories = json.loads(response.content[start:end])

                    # 存储记忆
                    for memory in memories:
                        if isinstance(memory, dict) and "content" in memory:
                            await self.memory_store.add(
                                memory["content"],
                                importance=memory.get("importance", 0.5),
                            )
            except (json.JSONDecodeError, KeyError):
                pass

        except Exception:
            # 记忆提取失败不影响主流程
            pass

        return ctx
