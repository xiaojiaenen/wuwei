"""Cost Governor — 预算断路器与成本归因

直击"token 成本超预估 10-20 倍、无法按任务归因"的痛点：

- 每次 run 设定 Budget（token / 估算成本 / 步数上限）
- 软阈值（默认 80%）发警告日志；硬阈值按策略执行：
  - "abort": 抛 BudgetExhausted，run 以 error 事件结束（已完成内容保留在会话里）
  - "warn": 仅记录，继续执行
- 成本按模型价格表估算（$/Mtok），可在 Budget.prices 覆盖

用法：
    budget = Budget(max_cost_usd=0.5, max_tokens=200_000, max_steps=40)
    agent = Agent(llm=llm, middleware=MiddlewareStack(BudgetMiddleware(budget, model="deepseek-flash")))
    result = await agent.run(task)
    cost = estimate_cost(result.usage, "deepseek-flash")   # 按任务归因
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from wuwei.core.errors import WuweiError
from wuwei.middleware.base import Middleware, MiddlewareContext

logger = logging.getLogger("wuwei.budget")


class BudgetExhausted(WuweiError):
    """预算耗尽，运行被断路。"""


# 常见模型价格（美元 / 百万 token）: (input, output)。未知模型用 DEFAULT。
DEFAULT_PRICE: tuple[float, float] = (0.5, 1.5)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "deepseek-chat": (0.27, 1.1),
    "deepseek-flash": (0.27, 1.1),
    "deepseek-reasoner": (0.55, 2.19),
    "claude-sonnet-4-6": (3.0, 15.0),
    "glm-4": (0.1, 0.1),
    "glm-4-flash": (0.01, 0.01),
    "qwen-max": (1.6, 6.4),
}


def estimate_cost(
    usage: dict[str, int] | None,
    model: str,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float:
    """按 usage 与模型价格表估算美元成本。"""
    if not usage:
        return 0.0
    table = prices or MODEL_PRICES
    price_in, price_out = DEFAULT_PRICE
    for key, value in sorted(table.items(), key=lambda item: len(item[0]), reverse=True):
        if model and model.startswith(key):
            price_in, price_out = value
            break
    prompt = usage.get("prompt_tokens", 0) / 1_000_000
    completion = usage.get("completion_tokens", 0) / 1_000_000
    return prompt * price_in + completion * price_out


@dataclass
class Budget:
    """一次 run 的预算约束（None 表示不限）。"""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_steps: int | None = None
    on_exhausted: str = "abort"  # abort | warn
    soft_ratio: float = 0.8
    prices: dict[str, tuple[float, float]] | None = None

    def __post_init__(self):
        if self.on_exhausted not in ("abort", "warn"):
            raise ValueError("on_exhausted 只支持 abort/warn")
        for value in (self.max_tokens, self.max_cost_usd, self.max_steps):
            if value is not None and value < 0:
                raise ValueError("Budget limits must be nonnegative")
        if not 0 < self.soft_ratio <= 1:
            raise ValueError("soft_ratio must be in (0,1]")


@dataclass
class _RunSpend:
    """单个 run 的累计消耗（挂在 ctx.metadata 上，跨步骤共享）。"""

    model: str
    warned: bool = False
    cost_usd: float = 0.0


class BudgetMiddleware(Middleware):
    """预算中间件：软阈值警告、硬阈值断路。

    依赖 AgentRunner 在 ctx.metadata 写入 usage_total（累计 usage dict 引用）
    与 step_index（当前步号）。
    """

    def __init__(self, budget: Budget, model: str = ""):
        self.budget = budget
        self.model = model

    @staticmethod
    def _scope(ctx):
        while ctx.config.get("parent_context") is not None:
            ctx = ctx.config["parent_context"]
        return ctx

    def _spend(self, ctx: MiddlewareContext) -> _RunSpend:
        ctx = self._scope(ctx)
        usage = ctx.metadata.get("usage_total") or {}
        records = ctx.metadata.get("usage_records") or []
        cost = (
            sum(
                estimate_cost(r.get("usage"), r.get("model") or self.model, self.budget.prices)
                for r in records
            )
            if records
            else estimate_cost(usage, self.model, self.budget.prices)
        )
        return _RunSpend(
            model=self.model, warned=ctx.metadata.get("_budget_warned", False), cost_usd=cost
        )

    def _check(
        self,
        ctx: MiddlewareContext,
        usage: dict[str, int],
        steps: int,
        spend: _RunSpend,
    ) -> None:
        b = self.budget
        if ctx.metadata.get("usage_available") is False and (
            b.max_tokens is not None or b.max_cost_usd is not None
        ):
            raise BudgetExhausted("模型未提供完整usage，无法可靠执行token/成本预算，已停止后续动作")
        exhausted_reasons: list[str] = []
        if b.max_tokens is not None and usage.get("total_tokens", 0) >= b.max_tokens:
            exhausted_reasons.append(f"tokens {usage.get('total_tokens', 0)} >= {b.max_tokens}")
        if b.max_cost_usd is not None and spend.cost_usd >= b.max_cost_usd:
            exhausted_reasons.append(f"cost ${spend.cost_usd:.4f} >= ${b.max_cost_usd}")
        if b.max_steps is not None and steps >= b.max_steps:
            exhausted_reasons.append(f"steps {steps} >= {b.max_steps}")

        if exhausted_reasons:
            msg = "预算耗尽: " + "; ".join(exhausted_reasons)
            if b.on_exhausted == "abort":
                raise BudgetExhausted(msg)
            logger.warning("[budget] %s（warn 模式继续执行）", msg)
            return

        # 软阈值警告（只警告一次）
        if not spend.warned:
            soft_hit = False
            if (
                b.max_tokens is not None
                and usage.get("total_tokens", 0) >= b.max_tokens * b.soft_ratio
            ):
                soft_hit = True
            if b.max_cost_usd is not None and spend.cost_usd >= b.max_cost_usd * b.soft_ratio:
                soft_hit = True
            if b.max_steps is not None and steps >= b.max_steps * b.soft_ratio:
                soft_hit = True
            if soft_hit:
                spend.warned = True
                self._scope(ctx).metadata["_budget_warned"] = True
                logger.warning(
                    "[budget] 接近预算上限（%.0f%%）：tokens=%s cost=$%.4f steps=%s",
                    b.soft_ratio * 100,
                    usage.get("total_tokens", 0),
                    spend.cost_usd,
                    steps,
                )

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        scope = self._scope(ctx)
        usage = scope.metadata.get("usage_total") or {}
        steps = scope.metadata.get("llm_calls", scope.metadata.get("step_index", 0))
        self._check(scope, usage, steps, self._spend(ctx))
        return ctx

    async def after_llm(self, ctx: MiddlewareContext, response) -> MiddlewareContext:
        scope = self._scope(ctx)
        usage = scope.metadata.get("usage_total") or {}
        steps = scope.metadata.get("llm_calls", scope.metadata.get("step_index", 0) + 1)
        self._check(scope, usage, steps, self._spend(ctx))
        return ctx
