"""Jev（System One 决策模型）接入示例

Jev 是 TypeSafe AI 2026-09 发布的"系统一"决策模型：不生成文本，
只对输入 state 回答带校准概率的结构化判定，快而便宜。
本示例演示三种典型玩法 + 与 HitlMiddleware 的组合（fail-closed 自动审批）。

运行：python examples/decision_client_demo.py
（需要 TYPESAFE_API_KEY 环境变量；或改用 OpenRouter/本地 Intern-Decision，
 见 wuwei/decision/client.py 模块文档）
"""

from __future__ import annotations

import asyncio
import os

from wuwei.decision import DecisionClient, Noul, Choice, decision_approval_provider
from wuwei.middleware import HitlMiddleware


async def main() -> None:
    client = DecisionClient(
        api_key=os.getenv("TYPESAFE_API_KEY", ""),
        # 后端切换示例：
        # OpenRouter:  base_url="https://openrouter.ai/api/v1"
        # 本地开源版 Intern-Decision: base_url="http://localhost:8000/v1"
    )

    # ── 玩法 1：工单分诊（一次调用并行回答多个问题） ────────────────
    answers = await client.judge(
        state="Help! My payouts have been failing for 3 days.",
        questions={
            "is_urgent": Noul("Does this convey urgency?"),
            "department": Choice(
                "Which department should handle this?",
                {"billing": None, "technical": None, "sales": None},
            ),
        },
    )
    print("紧急概率:", answers["is_urgent"].value)
    print("分派部门:", answers["department"].value, "置信度:", answers["department"].confidence)

    # ── 玩法 2：路由（低置信抛 LowConfidence → 走默认大模型） ───────
    try:
        choice = await client.route(
            state={"task": "总结一篇 20 页的 PDF 报告"},
            options=["quick_model", "reasoning_model", " escalate_to_human"],
            min_confidence=0.7,
        )
        print("路由选择:", choice)
    except Exception as exc:
        print("路由降级:", exc)

    # ── 玩法 3：HITL 自动审批（fail-closed） ────────────────────────
    hitl = HitlMiddleware(
        approval_provider=decision_approval_provider(client, threshold=0.9),
        approval_policy="declared",  # 只对 requires_approval=True 的工具启用
    )
    print("HITL 中间件已就绪：高置信操作自动放行，低置信/服务异常一律升级人工。")

    # ── 兜底演示：无 API key 时服务不可达 → fail-closed ─────────────
    verdict = await client.approve({"action": "rm -rf /"}, threshold=0.9)
    print("危险操作审批判定:", verdict, "（异常或低置信都返回 escalate）")


if __name__ == "__main__":
    asyncio.run(main())
