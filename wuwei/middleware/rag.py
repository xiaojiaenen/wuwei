"""RAG 中间件"""

from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import BaseMessage, SystemMessage


class RagMiddleware(Middleware):
    """RAG 中间件

    从知识库检索相关文档并注入到上下文。

    示例：
        from wuwei.memory import InMemoryKnowledgeStore

        knowledge_store = InMemoryKnowledgeStore()
        middleware = RagMiddleware(knowledge_store=knowledge_store)
    """

    def __init__(
        self,
        knowledge_store,
        top_k: int = 5,
        min_score: float = 0.3,
    ):
        """
        Args:
            knowledge_store: 知识库存储
            top_k: 检索的最大文档数
            min_score: 最小相似度阈值
        """
        self.knowledge_store = knowledge_store
        self.top_k = top_k
        self.min_score = min_score

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前检索相关文档"""
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

        # 搜索相关文档
        results = await self.knowledge_store.search(
            last_user_msg.content,
            top_k=self.top_k,
        )

        if results:
            # 过滤低分结果
            results = [
                r for r in results
                if r.score >= self.min_score
            ]

            if results:
                # 构建文档文本
                doc_texts = []
                for r in results:
                    doc_texts.append(f"[来源: {r.source}]\n{r.text}")

                docs_text = "\n\n---\n\n".join(doc_texts)

                # 注入到消息开头
                ctx.state.messages.insert(
                    0,
                    SystemMessage(content=f"相关文档：\n{docs_text}"),
                )

        return ctx
