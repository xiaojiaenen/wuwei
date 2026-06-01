"""决策工具

通用的用户决策工具，任何 Agent 都可以使用。
LLM 调用工具提供问题和选项，用户做决策。
"""

from __future__ import annotations

import uuid
from typing import Any

from wuwei.tools.registry import ToolRegistry


def register_decision_tools(registry: ToolRegistry) -> None:
    """注册决策工具"""

    @registry.tool(
        name="ask_user_decision",
        display_name="请求用户决策",
        description="向用户提出一个决策问题，提供选项让用户选择。用于需要用户确认或选择的场景。每个选项都会展示给用户，用户也可以输入自定义答案。",
    )
    async def ask_user_decision(
        question: str,
        options: list[str],
        context: str = "",
    ) -> dict[str, Any]:
        """向用户提出决策问题。

        参数:
            question: 要问用户的问题
            options: 选项列表（至少 2 个）
            context: 补充上下文信息

        返回:
            包含 decision_id 和选项的字典
        """
        if len(options) < 2:
            return {"error": "至少需要 2 个选项"}

        decision_id = str(uuid.uuid4())[:8]

        return {
            "ok": True,
            "decision_id": decision_id,
            "type": "user_decision",
            "question": question,
            "options": options,
            "context": context,
            "allow_custom": True,
            "message": f"已向用户提出决策问题：{question}，等待用户选择。",
        }
