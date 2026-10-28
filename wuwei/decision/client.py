"""Decision Client — System One 决策模型接入层

接入 TypeSafe AI 的 Jev（2026-09 发布的"系统一"决策模型）及同范式模型：
对输入 state 回答带校准概率的结构化判定（是/否、选择、打分），
不生成自由文本，延迟与成本比 chat 模型低 1-2 个数量级。

三种后端通过 base_url 切换（协议同构）：
- TypeSafe 云:      https://api.typesafe.ai/v1        (POST /v1/systemone)
- OpenRouter 托管:  https://openrouter.ai/api/v1      (POST /api/v1/systemone)
- Intern-Decision:  http://localhost:8000             (POST /v1/decisions，开源多模态判定模型)

在 agent 框架中的典型用法（不要用它做主循环模型——它只会输出判定）：
1. HITL 自动审批：decision_approval_provider(client) 作为 HitlMiddleware 的
   approval_provider，低置信自动升级人工（fail-closed）
2. 模型/工具/技能路由：client.route(state, options)
3. 意图分类 / 记忆相关性重排：client.classify(state, labels)

API 参考: https://docs.typesafe.ai/api
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

logger = logging.getLogger("wuwei.decision")

__all__ = [
    "Noul",
    "Choice",
    "Score",
    "Answer",
    "DecisionError",
    "DecisionUnavailable",
    "LowConfidence",
    "DecisionClient",
    "decision_approval_provider",
]


# ── 问题类型 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Noul:
    """是/否判定。judge 返回 yes 的概率（0-1）。"""

    instructions: str
    criteria: dict[str, str | None] | None = None  # {"true": ..., "false": ...}

    def to_api(self) -> dict:
        q: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            q["criteria"] = self.criteria
        return q


@dataclass(frozen=True)
class Choice:
    """单选判定（最多 255 个选项）。judge 返回 (选项, 概率表, confidence)。"""

    instructions: str
    options: dict[str, str | None]  # 选项名 -> 判定标准（rubric）

    def to_api(self) -> dict:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.options),
        }


@dataclass(frozen=True)
class Score:
    """有序量表打分（2-10 级）。judge 返回概率加权分数（可落在级间）。"""

    instructions: str
    levels: list[str]

    def __post_init__(self):
        if not 2 <= len(self.levels) <= 10:
            raise ValueError("Score 需要 2-10 个等级")

    def to_api(self) -> dict:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": dict.fromkeys(self.levels),
        }


@dataclass(frozen=True)
class Answer:
    """一条判定的结果。"""

    type: Literal["noul", "choice", "score"]
    value: float | str  # noul: 概率; choice: 选项名; score: 加权分数
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


class DecisionError(Exception):
    """决策 API 请求失败（重试耗尽后）。"""


class DecisionUnavailable(DecisionError):
    """决策服务不可用（网络错误/过载）。fail-closed 场景应按"不能自动判定"处理。"""


class LowConfidence(DecisionError):
    """判定置信度低于阈值。"""


# ── 客户端 ───────────────────────────────────────────────────────────


class DecisionClient:
    """System One 决策模型客户端（Jev / OpenRouter / Intern-Decision 通用）。

    示例：
        client = DecisionClient(api_key="...", model="jev-latest")
        answers = await client.judge(
            state="用户要求删除数据库",
            questions={
                "is_destructive": Noul("该请求是否具有破坏性？"),
                "handler": Choice("应交给谁处理？", {"human": None, "agent": None}),
            },
        )
        answers["is_destructive"].value   # 0.97（yes 概率）
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.typesafe.ai/v1",
        model: str = "jev-latest",
        timeout: float = 5.0,
        max_retries: int = 2,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport  # 测试可注入 MockTransport
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(headers or {}),
        }

    async def judge(
        self,
        state: str | dict | list,
        questions: dict[str, Noul | Choice | Score],
    ) -> dict[str, Answer]:
        """一次调用并行回答多个问题（省延迟省钱的最佳实践）。

        429/529 指数退避重试；网络/服务异常抛 DecisionUnavailable。
        """
        if not questions:
            return {}

        payload = {
            "state": state if isinstance(state, str) else json_dumps(state),
            "model": self.model,
            "questions": {k: q.to_api() for k, q in questions.items()},
        }
        url = f"{self.base_url}/systemone"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self._transport
                ) as client:
                    resp = await client.post(url, json=payload, headers=self._headers)
                if resp.status_code in (429, 529):
                    last_exc = DecisionError(f"决策服务限流/过载: {resp.status_code}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt * 0.5)
                        continue
                    break
                resp.raise_for_status()
                return self._parse_answers(resp.json(), questions)
            except httpx.HTTPStatusError as exc:
                raise DecisionError(
                    f"决策 API 错误 {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except httpx.HTTPError as exc:
                last_exc = DecisionUnavailable(f"决策服务不可达: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt * 0.5)
                    continue
                break

        raise last_exc or DecisionUnavailable("决策服务不可达")

    # ── 便捷封装 ─────────────────────────────────────────────────────

    async def route(
        self,
        state: str | dict,
        options: list[str],
        min_confidence: float = 0.7,
        instructions: str = "Which option is the best fit?",
    ) -> str:
        """路由判定（模型/工具/技能选择）。低于阈值抛 LowConfidence。"""
        answer = await self.judge(
            state,
            {"route": Choice(instructions, dict.fromkeys(options))},
        )
        result = answer["route"]
        if result.confidence < min_confidence:
            raise LowConfidence(
                f"路由置信度 {result.confidence:.2f} < {min_confidence}，"
                f"probabilities={result.probabilities}"
            )
        return str(result.value)

    async def approve(
        self,
        state: str | dict,
        criteria: dict[str, str | None] | None = None,
        threshold: float = 0.9,
    ) -> Literal["auto_approve", "escalate"]:
        """HITL 自动审批（fail-closed）：服务异常或置信度不足一律 escalate。"""
        try:
            answer = await self.judge(
                state,
                {"approve": Noul("该操作是否可以安全地自动执行？", criteria)},
            )
            prob = float(answer["approve"].value)
            if prob >= threshold:
                return "auto_approve"
            return "escalate"
        except DecisionError:
            logger.warning("决策服务异常，HITL 升级人工审批（fail-closed）")
            return "escalate"

    async def classify(
        self,
        state: str | dict,
        labels: list[str],
        instructions: str = "Which label best describes this?",
    ) -> tuple[str, float]:
        """分类判定，返回 (label, confidence)。"""
        answer = await self.judge(
            state,
            {"label": Choice(instructions, dict.fromkeys(labels))},
        )
        result = answer["label"]
        return str(result.value), result.confidence

    # ── 响应解析 ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_answers(
        raw: dict, questions: dict[str, Noul | Choice | Score]
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        raw_answers = raw.get("answers") or {}
        for key, question in questions.items():
            item = raw_answers.get(key) or {}
            if isinstance(question, Noul):
                answers[key] = Answer(
                    type="noul", value=float(item.get("noul", 0.0))
                )
            elif isinstance(question, Choice):
                answers[key] = Answer(
                    type="choice",
                    value=str(item.get("choice", "")),
                    probabilities=dict(item.get("probabilities") or {}),
                    confidence=float(item.get("confidence", 0.0)),
                )
            elif isinstance(question, Score):
                answers[key] = Answer(
                    type="score",
                    value=float(item.get("score", 0.0)),
                    probabilities=dict(item.get("probabilities") or {}),
                    confidence=float(item.get("confidence", 0.0)),
                )
        return answers


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


# ── HITL 集成 ────────────────────────────────────────────────────────


def decision_approval_provider(
    client: DecisionClient,
    *,
    threshold: float = 0.9,
    always_ask_tools: list[str] | None = None,
) -> Callable[[Any], Awaitable[bool]]:
    """构造一个可传入 HitlMiddleware 的 approval_provider。

    行为（fail-closed）：
    - Jev 以 >= threshold 的置信度判定"可自动执行" → 返回 True（免打扰）
    - 其余情况（低置信/服务异常/always_ask_tools 名单）→ 返回 False，
      触发 HitlMiddleware 的人工审批回喂流程

    示例：
        hitl = HitlMiddleware(
            approval_provider=decision_approval_provider(client, threshold=0.9),
            approval_policy="declared",   # 只对 requires_approval 工具启用
        )
    """
    always_ask = set(always_ask_tools or [])

    async def _provider(tool_call) -> bool:
        if tool_call.function.name in always_ask:
            return False
        state = {
            "tool": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
        verdict = await client.approve(state, threshold=threshold)
        return verdict == "auto_approve"

    return _provider
