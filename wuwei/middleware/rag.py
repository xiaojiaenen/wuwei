"""RAG 中间件

从知识库检索相关文档并注入到上下文。
注入位置：紧随开头的 system 消息之后（不覆盖 system prompt）。
注入只作用于本轮 LLM 调用的消息视图，不写入会话历史，因此不会堆积。
"""

from wuwei.core.message import SystemMessage
from wuwei.middleware.base import Middleware, MiddlewareContext


def _injection_position(messages) -> int:
    """返回注入位置：跳过开头连续的 system 消息。"""
    pos = 0
    for msg in messages:
        if msg.role == "system":
            pos += 1
        else:
            break
    return pos


class RagMiddleware(Middleware):
    """RAG 中间件

    示例：
        from wuwei.memory import InMemoryKnowledgeStore

        knowledge_store = InMemoryKnowledgeStore()
        middleware = RagMiddleware(knowledge_store=knowledge_store)
    """

    def __init__(
        self,
        knowledge_store,
        top_k: int = 5,
        min_score: float | None = None,
        namespace: str = "default",
    ):
        """
        Args:
            knowledge_store: 知识库存储（实现 search(query, limit=...) 协议）
            top_k: 检索的最大文档数
            min_score: 最小相似度阈值（仅当检索结果自带 score 字段时生效）
            namespace: 知识库命名空间
        """
        self.knowledge_store = knowledge_store
        self.top_k = top_k
        self.min_score = min_score
        self.namespace = namespace

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前检索相关文档"""
        if not ctx.state or not ctx.state.messages:
            return ctx

        last_user_msg = None
        for msg in reversed(ctx.state.messages):
            if msg.role == "user":
                last_user_msg = msg
                break
        if not last_user_msg or not last_user_msg.content:
            return ctx

        results = await self.knowledge_store.search(
            last_user_msg.content,
            namespace=self.namespace,
            limit=self.top_k,
        )
        if not results:
            return ctx

        # 仅当结果自带 score 时按阈值过滤
        if self.min_score is not None and hasattr(results[0], "score"):
            results = [r for r in results if r.score >= self.min_score]
        if not results:
            return ctx

        doc_texts = []
        for r in results:
            title = getattr(r, "title", None)
            source = getattr(r, "source", "")
            header = f"[来源: {source}{' | ' + title if title else ''}]"
            doc_texts.append(f"{header}\n{r.text}")

        docs_text = "\n\n---\n\n".join(doc_texts)
        ctx.state.messages.insert(
            _injection_position(ctx.state.messages),
            SystemMessage(content=f"以下是知识库中与问题相关的参考资料：\n\n{docs_text}"),
        )
        return ctx
