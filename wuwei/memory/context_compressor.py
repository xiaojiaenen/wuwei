from typing import Protocol

from wuwei.llm import LLMGateway, Message


class ContextCompressor(Protocol):
    async def compress(
        self,
        *,
        previous_summary: str | None,
        messages: list[Message],
    ) -> str:
        """把一段旧消息压缩成可延续任务的摘要。"""
        ...


class LLMContextCompressor:
    def __init__(self, llm: LLMGateway, system_prompt: str | None = None) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or (
            "你是一个 Agent 上下文压缩器。"
            "你的任务是把旧对话压缩成后续可继续工作的状态摘要。"
            "不要编造历史中没有的信息。"
        )

    async def compress(
        self,
        *,
        previous_summary: str | None,
        messages: list[Message],
    ) -> str:
        history_text = self._format_messages(messages)
        prompt = f"""
        请压缩以下 Agent 历史，输出简洁、结构化中文摘要。

        必须保留：
        - 用户目标
        - 已确认事实和约束
        - 用户偏好
        - 已执行工具及关键结果
        - 当前进度
        - 待办事项
        - 风险和阻塞

        已有摘要：
        {previous_summary or "无"}

        待压缩历史：
        {history_text}
        """.strip()

        response = await self.llm.generate(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt),
            ]
        )
        return response.message.content or ""

    def _format_messages(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.content or ""
            if message.tool_calls:
                content += f"\n工具调用: {message.model_dump_json()}"
            lines.append(f"[{message.role}] {content}")
        return "\n\n".join(lines)
