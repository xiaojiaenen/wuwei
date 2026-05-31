"""上下文压缩中间件"""

from typing import Optional
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.core.message import BaseMessage, HumanMessage, AIMessage, SystemMessage
from wuwei.llm.gateway import LLMGateway


class ContextCompressionMiddleware(Middleware):
    """上下文压缩中间件

    借鉴 DeepAgents 的 SummarizationMiddleware，支持：
    - 自动检测 token 数量
    - 生成对话摘要
    - 保留最近消息
    - 卸载旧消息到文件

    示例：
        middleware = ContextCompressionMiddleware(
            llm=llm,
            trigger_tokens=4000,
            keep_recent=10,
        )
    """

    def __init__(
        self,
        llm: LLMGateway,
        trigger_tokens: int = 4000,
        keep_recent: int = 10,
        max_tokens: int = 8000,
        summary_model: str = None,
    ):
        """
        Args:
            llm: LLM 网关
            trigger_tokens: 触发压缩的 token 数量
            keep_recent: 保留最近的消息数
            max_tokens: 最大 token 数量
            summary_model: 用于摘要的模型（可选）
        """
        self.llm = llm
        self.trigger_tokens = trigger_tokens
        self.keep_recent = keep_recent
        self.max_tokens = max_tokens
        self.summary_model = summary_model
        self._summary_cache = {}

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前检查并压缩上下文"""
        if not ctx.state or not ctx.state.messages:
            return ctx

        messages = ctx.state.messages

        # 计算 token 数量（简化版：按字符数估算）
        total_chars = sum(len(msg.content) for msg in messages if msg.content)
        estimated_tokens = total_chars // 4  # 粗略估算

        # 检查是否需要压缩
        if estimated_tokens <= self.trigger_tokens:
            return ctx

        # 压缩上下文
        compressed_messages = await self._compress_context(messages)
        ctx.state.messages = compressed_messages

        return ctx

    async def _compress_context(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """压缩上下文"""
        if len(messages) <= self.keep_recent:
            return messages

        # 分离系统消息、旧消息、新消息
        system_messages = [msg for msg in messages if msg.role == "system"]
        old_messages = messages[len(system_messages):-self.keep_recent]
        new_messages = messages[-self.keep_recent:]

        # 生成摘要
        if old_messages:
            summary = await self._generate_summary(old_messages)
            summary_message = HumanMessage(
                content=f"对话摘要：\n{summary}"
            )
            return system_messages + [summary_message] + new_messages

        return messages

    async def _generate_summary(self, messages: list[BaseMessage]) -> str:
        """生成对话摘要"""
        # 构建摘要提示词
        conversation = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in messages
            if msg.content
        )

        prompt = f"""请为以下对话生成一个简洁的摘要，保留关键信息：

{conversation}

摘要："""

        try:
            response = await self.llm.generate(
                messages=[BaseMessage(role="user", content=prompt)],
            )
            return response.content
        except Exception:
            # 如果摘要生成失败，返回简单的统计信息
            return f"对话包含 {len(messages)} 条消息，{sum(len(m.content) for m in messages if m.content)} 个字符"
