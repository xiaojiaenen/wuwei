"""v3.2 高级能力综合示例：durable / 权限 / 预算 / 账本 / 计划

运行前在项目根目录准备 .env（OPENAI_* 或 ANTHROPIC_*），例如：
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://api.deepseek.com
    OPENAI_MODEL=deepseek-flash

    python examples/advanced_features.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from wuwei import (
    Agent,
    Budget,
    BudgetMiddleware,
    ContextLedger,
    LLMGateway,
    LedgerMiddleware,
    MiddlewareStack,
    PermissionConfig,
    PermissionMiddleware,
)
from wuwei.graph.checkpoint import SQLiteCheckpointer
from wuwei.middleware.ledger import ledger_tools


class Report(BaseModel):
    topic: str
    key_points: list[str]


async def main() -> None:
    llm = LLMGateway.from_env()

    # ── 1. 权限（deny-by-default）+ 预算 + 账本 组合中间件 ───────────
    ledger = ContextLedger()
    middleware = MiddlewareStack()
    middleware.add(
        PermissionMiddleware(
            PermissionConfig(
                filesystem_write=["./out/**"],
                network_allow=["api.github.com"],
                shell_enabled=False,
            )
        )
    )
    middleware.add(BudgetMiddleware(Budget(max_tokens=50_000, on_exhausted="warn")))
    middleware.add(LedgerMiddleware(ledger, archive_threshold=2000))

    agent = Agent(
        llm=llm,
        load_builtins="minimal",
        default_system_prompt="你是一个简洁可靠的助手。",
        middleware=middleware,
        checkpointer=SQLiteCheckpointer(".wuwei/checkpoints.db"),  # durable 基础
    )
    for tool in ledger_tools(ledger):
        agent.tool_registry.register(tool)

    # ── 2. 结构化输出（校验失败自动重试） ────────────────────────────
    model, _ = await agent.run_structured("给三个学习 Python 的理由", Report)
    print("结构化输出:", model.model_dump_json())

    # ── 3. Plan-and-Execute（DAG 并行调度） ──────────────────────────
    plan_result = await agent.run_plan(
        "写一份 100 字的 AI Agent 趋势简报",
        context="面向中文读者，语气务实。",
    )
    for task in plan_result.tasks:
        print(f"任务 {task.id}: {task.status} - {(task.result or task.error or '')[:60]}")
    print("计划用量:", plan_result.usage)

    # ── 4. 成本归因 ──────────────────────────────────────────────────
    from wuwei import estimate_cost

    print(f"本次会话估算成本: ${estimate_cost(agent._default_session.last_usage, llm.adapter.model):.4f}")

    # ── 5. durable interrupt / resume（HITL 挂起-恢复） ──────────────
    # 需要 requires_approval=True 的工具触发，例如：
    #   agent.middleware.add(HitlMiddleware(mode="interrupt", approval_policy="declared"))
    #   result = await agent.run("部署新版本")        # status == "interrupted"
    #   await agent.resume(result.interrupt, approved=True)
    # kill -9 后重启进程，检查点仍在 .wuwei/checkpoints.db，可继续恢复。

    # ── 6. Time-Travel：从检查点分叉重放 ─────────────────────────────
    from wuwei.debug import list_steps

    steps = await list_steps(agent.checkpointer, limit=5)
    print(f"\n检查点（最近 {len(steps)} 个）：")
    for s in steps:
        print(f"  {s['id']}  kind={s['kind']} step={s['step']}")


if __name__ == "__main__":
    asyncio.run(main())
