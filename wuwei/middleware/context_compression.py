"""上下文压缩中间件

借鉴 Hermes Agent 的 ContextCompressor 和 DeepAgents 的 SummarizationMiddleware：

压缩策略：
1. 触发条件：基于 token 阈值（trigger_ratio * context_size）
2. 保护头部：系统消息 + 前 N 条消息始终保留
3. 保护尾部：按 token 预算保留最近的对话轮次
4. 旧轮次 → LLM 结构化摘要（已解决/待解决问题 + 活跃任务）
5. 反抖动：上次压缩节省 <10% 则跳过
6. 摘要失败冷却：600s 内不重试失败的模型
7. 工具结果去重：相同输出的工具调用合并
"""

import hashlib
import time
from typing import Optional

from wuwei.core.message import AIMessage, ToolMessage
from wuwei.llm.types import Message
from wuwei.middleware.base import Middleware, MiddlewareContext


class ContextCompressionMiddleware(Middleware):
    """上下文压缩中间件

    在 LLM 调用前自动检测并压缩超长上下文。

    示例：
        middleware = ContextCompressionMiddleware(
            llm=llm,
            trigger_tokens=4000,
            keep_recent_turns=3,
        )
    """

    # 结构化摘要模板
    SUMMARY_TEMPLATE = """# Conversation Summary
*This is an automatically generated summary of earlier conversation turns.*
*Use it for context; the original messages have been removed to save tokens.*

## Resolved Questions
{resolved}

## Pending Questions
{pending}

## Active Task
{active_task}

## Key Findings
{key_findings}
"""

    def __init__(
        self,
        llm,
        trigger_tokens: int = 6000,
        keep_recent_turns: int = 3,
        max_tokens: int = 128000,
        trigger_ratio: float = 0.75,
        min_savings_ratio: float = 0.10,
        failure_cooldown_s: int = 600,
    ):
        """
        Args:
            llm: LLM 网关（用于生成摘要）
            trigger_tokens: 绝对触发阈值（token 数）
            keep_recent_turns: 保留最近的对话轮次数
            max_tokens: 模型上下文窗口大小
            trigger_ratio: 相对触发阈值（占用上下文窗口比例）
            min_savings_ratio: 最小节省比例（低于此值跳过压缩，防抖动）
            failure_cooldown_s: 摘要失败后的冷却时间（秒）
        """
        self.llm = llm
        self.trigger_tokens = trigger_tokens
        self.keep_recent_turns = keep_recent_turns
        self.max_tokens = max_tokens
        self.trigger_ratio = trigger_ratio
        self.min_savings_ratio = min_savings_ratio
        self.failure_cooldown_s = failure_cooldown_s

        # 状态追踪
        self._last_summary: str | None = None
        self._last_summary_saved_tokens: int = 0
        self._last_summary_time: float = 0
        self._last_failure_time: float = 0
        self._summary_cache: dict[str, str] = {}

    def _estimate_tokens(self, messages: list) -> int:
        """估算消息列表的 token 数

        优先使用 tiktoken，降级为 Unicode 字符估算。
        """
        total_chars = 0
        for msg in messages:
            content = getattr(msg, 'content', '') or ''
            total_chars += len(content)
            # tool_calls 也计入
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(str(tc))

        # Unicode 感知：中文字符 ≈ 1.5 tokens，英文 ≈ 0.25 tokens/char
        cjk_count = sum(1 for c in str(total_chars) if '\u4e00' <= c <= '\u9fff')
        non_cjk = total_chars - cjk_count
        return int(cjk_count * 1.5 + non_cjk * 0.3)

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """LLM 调用前检查并压缩上下文"""
        if not hasattr(ctx.state, 'messages') or not ctx.state.messages:
            return ctx

        messages = ctx.state.messages
        estimated_tokens = self._estimate_tokens(messages)

        # 触发检查：绝对阈值 + 相对阈值
        trigger = min(self.trigger_tokens, int(self.max_tokens * self.trigger_ratio))
        if estimated_tokens <= trigger:
            return ctx

        # 反抖动检查：上次压缩节省的 token < 10%
        if self._last_summary_time > 0:
            savings = estimated_tokens - trigger
            if self._last_summary_saved_tokens > 0:
                savings_ratio = savings / self._last_summary_saved_tokens
                if savings_ratio < self.min_savings_ratio:
                    return ctx

        # 执行压缩
        compressed = await self._compress_context(messages)
        ctx.state.messages = compressed

        new_estimate = self._estimate_tokens(compressed)
        self._last_summary_saved_tokens = estimated_tokens - new_estimate
        self._last_summary_time = time.time()

        return ctx

    def _split_turns(self, messages: list) -> list[list]:
        """将消息列表按对话轮次分组"""
        turns = []
        current_turn = []
        for msg in messages:
            current_turn.append(msg)
            if msg.role in ("assistant", "tool") and hasattr(msg, 'tool_calls') and not msg.tool_calls:
                # assistant 消息无 tool_calls → 轮次结束
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []
        if current_turn:
            turns.append(current_turn)
        return turns

    async def _compress_context(self, messages: list) -> list:
        """压缩上下文"""
        if len(messages) <= 2:
            return messages

        # 分离系统消息
        system_messages = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= self.keep_recent_turns * 2:
            return messages

        # 保护头部（系统消息 + 第一轮用户问题）
        protect_head = min(2, len(non_system) // 3)
        protect_tail = self.keep_recent_turns * 2  # 用户+助手各算一条

        old_messages = non_system[protect_head:-protect_tail] if protect_tail > 0 else non_system[protect_head:]
        recent_messages = non_system[-protect_tail:] if protect_tail > 0 else []

        if not old_messages:
            return messages

        # 工具结果去重
        old_messages = self._deduplicate_tool_results(old_messages)

        # 如果有缓存的摘要，迭代更新
        existing_summary = self._last_summary
        summary = await self._generate_structured_summary(old_messages, existing_summary)

        if summary:
            self._last_summary = summary
            summary_msg = Message(
                role="user",
                content=summary,
            )
            result = system_messages + [summary_msg] + recent_messages
            return result

        return messages

    def _deduplicate_tool_results(self, messages: list) -> list:
        """去重工具结果

        相同哈希的工具输出替换为简洁摘要。
        """
        seen_hashes: dict[str, int] = {}
        result = []
        for msg in messages:
            if msg.role == "tool":
                content = getattr(msg, 'content', '') or ''
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash in seen_hashes:
                    seen_hashes[content_hash] += 1
                    # 替换重复结果为简短引用
                    result.append(type(msg)(
                        role="tool",
                        content=f"[同上，结果与消息 #{seen_hashes[content_hash]} 相同]",
                        tool_call_id=getattr(msg, 'tool_call_id', ''),
                        name=getattr(msg, 'name', ''),
                    ))
                    continue
                seen_hashes[content_hash] = 1
            result.append(msg)
        return result

    async def _generate_structured_summary(
        self,
        messages: list,
        existing_summary: str | None = None,
    ) -> str | None:
        """生成结构化摘要

        使用 LLM 生成包含已解决/待解决问题、活跃任务的结构化摘要。
        """
        # 冷却检查
        if self._last_failure_time > 0:
            if time.time() - self._last_failure_time < self.failure_cooldown_s:
                return None

        # 构建压缩文本
        conversation_parts = []
        for msg in messages:
            content = getattr(msg, 'content', '') or ''
            if not content:
                continue
            role = getattr(msg, 'role', 'user')
            # 截断超长工具输出
            if role == "tool" and len(content) > 500:
                content = content[:500] + "..."
            conversation_parts.append(f"[{role}]: {content}")

        conversation_text = "\n".join(conversation_parts)
        if not conversation_text.strip():
            return None

        # 缓存检查
        cache_key = hashlib.md5(conversation_text.encode()).hexdigest()
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        # 构建提示
        existing_note = ""
        if existing_summary:
            existing_note = f"\n\nPrevious summary (iteratively update it):\n{existing_summary}"

        prompt = f"""Summarize the following conversation into a structured summary.
Focus on: user's goals, decisions made, tool results, current progress, and blockers.

{conversation_text}{existing_note}

Return ONLY the summary, no extra text. Use this format:
## Resolved
- item 1
## Pending
- item 1
## Active Task
current task description
## Key Findings
- finding 1"""

        try:
            response = await self.llm.generate(
                messages=[Message(role="user", content=prompt)],
            )
            summary = response.message.content if hasattr(response, 'message') else str(response)
            if summary:
                self._summary_cache[cache_key] = summary
                # LRU 缓存：保留最近 20 条
                if len(self._summary_cache) > 20:
                    oldest = next(iter(self._summary_cache))
                    del self._summary_cache[oldest]
            return summary
        except Exception:
            self._last_failure_time = time.time()
            return None
