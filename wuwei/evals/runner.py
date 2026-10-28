"""评估执行器：逐用例运行 agent 并评估断言。"""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from wuwei.evals.case import AssertionResult, EvalCase, EvalResult, import_from_string
from wuwei.evals.report import EvalReport

# judge 协议：async (input, output, rubric) -> (score, explanation)
JudgeFn = Callable[[str, str, str], Awaitable[tuple[float, str]]]


def _strip_code_fences(text: str) -> str:
    """剥掉 ```json ... ``` 围栏，便于解析模型输出。"""
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def extract_tool_calls(session: Any) -> list[str]:
    """从会话消息历史提取实际调用的工具名序列。

    优先取 assistant 消息的 tool_calls.function.name；
    若整个历史没有 assistant.tool_calls（部分适配器不回填），
    则回退到 role=tool 消息的 name 字段。
    """
    names: list[str] = []
    for m in session.context.get_messages():
        if getattr(m, "role", None) == "assistant":
            for tc in getattr(m, "tool_calls", None) or []:
                names.append(tc.function.name)
    if names:
        return names
    for m in session.context.get_messages():
        if getattr(m, "role", None) == "tool" and getattr(m, "name", None):
            names.append(m.name)
    return names


class EvalRunner:
    """评估运行器。

    Args:
        agent_factory: 返回 agent 的工厂（每个用例调用一次，保证全新隔离）。
            agent 需满足：`create_session(max_steps=...)` 与 `async run(input, session=...)`，
            与 wuwei Agent 一致即可（duck-type）。
        judge: 可选的 LLM-as-judge 可调用对象，
            签名 `async (input, output, rubric) -> (score, explanation)`，
            score 为 0-1 浮点。为 None 时 rubric 断言标记 skipped。
    """

    def __init__(self, agent_factory: Callable[[], Any], judge: JudgeFn | None = None) -> None:
        self.agent_factory = agent_factory
        self.judge = judge

    async def run(self, dataset: Any) -> EvalReport:
        """运行整个数据集（逐用例串行，避免相互干扰）。"""
        results = []
        for case in dataset.cases:
            results.append(await self.run_case(case))
        return EvalReport(dataset_name=dataset.name, results=results)

    async def run_case(self, case: EvalCase) -> EvalResult:
        agent = self.agent_factory()
        session = self._make_session(agent, case)

        start = time.monotonic()
        try:
            run_result = await agent.run(case.input, session=session)
        except Exception as e:  # noqa: BLE001 - 评估不应被单用例异常炸穿
            latency = int((time.monotonic() - start) * 1000)
            return EvalResult(
                case_name=case.name,
                passed=False,
                assertions=[
                    AssertionResult(
                        kind="run", name="agent.run", passed=False, message=f"运行异常: {e}"
                    )
                ],
                latency_ms=latency,
                error=repr(e),
            )

        latency = int((time.monotonic() - start) * 1000)
        content = getattr(run_result, "content", "") or ""
        usage = dict(getattr(run_result, "usage", None) or {})
        actual_tools = extract_tool_calls(session)

        assertions: list[AssertionResult] = []
        assertions.extend(self._check_tool_calls(case, actual_tools))
        assertions.extend(self._check_content(case, content))
        assertions.append(await self._check_response_model(case, content))
        assertions.append(await self._check_rubric(case, content))

        passed = all(a.passed for a in assertions if not a.skipped)
        return EvalResult(
            case_name=case.name,
            passed=passed,
            assertions=assertions,
            usage=usage,
            latency_ms=latency,
            output=content,
            tool_calls=actual_tools,
        )

    # ── 会话 ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_session(agent: Any, case: EvalCase) -> Any:
        """为用例创建全新会话；max_steps 覆盖失败时静默降级。"""
        if case.max_steps is not None:
            try:
                return agent.create_session(max_steps=case.max_steps)
            except TypeError:
                pass
        return agent.create_session()

    # ── 断言评估 ──────────────────────────────────────────────────────

    @staticmethod
    def _check_tool_calls(case: EvalCase, actual: list[str]) -> list[AssertionResult]:
        results: list[AssertionResult] = []
        actual_set = set(actual)

        for name in case.expected_tool_calls:
            results.append(
                AssertionResult(
                    kind="expected_tool_calls",
                    name=name,
                    passed=name in actual_set,
                    message=(
                        f"工具 '{name}' 已调用"
                        if name in actual_set
                        else f"缺少必需的工具调用 '{name}'，实际: {actual or '无'}"
                    ),
                )
            )

        for name in case.forbidden_tool_calls:
            results.append(
                AssertionResult(
                    kind="forbidden_tool_calls",
                    name=name,
                    passed=name not in actual_set,
                    message=(
                        f"禁用工具 '{name}' 未被调用"
                        if name not in actual_set
                        else f"出现了禁用的工具调用 '{name}'，实际: {actual}"
                    ),
                )
            )
        return results

    @staticmethod
    def _check_content(case: EvalCase, content: str) -> list[AssertionResult]:
        results: list[AssertionResult] = []
        for sub in case.response_contains:
            results.append(
                AssertionResult(
                    kind="contains",
                    name=sub,
                    passed=sub in content,
                    message=(
                        f"回复包含 '{sub}'"
                        if sub in content
                        else f"回复未包含 '{sub}'，回复: {content[:200]!r}"
                    ),
                )
            )
        for sub in case.response_not_contains:
            results.append(
                AssertionResult(
                    kind="not_contains",
                    name=sub,
                    passed=sub not in content,
                    message=(
                        f"回复未包含禁用内容 '{sub}'"
                        if sub not in content
                        else f"回复包含禁用内容 '{sub}'"
                    ),
                )
            )
        return results

    @staticmethod
    async def _check_response_model(case: EvalCase, content: str) -> AssertionResult:
        if not case.response_model:
            return AssertionResult(
                kind="response_model", name="-", passed=True, skipped=True, message="未配置"
            )
        try:
            model_cls = import_from_string(case.response_model)
        except Exception as e:  # noqa: BLE001
            return AssertionResult(
                kind="response_model",
                name=case.response_model,
                passed=False,
                message=f"导入 response_model 失败: {e}",
            )
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as e:
            return AssertionResult(
                kind="response_model",
                name=case.response_model,
                passed=False,
                message=f"输出不是合法 JSON: {e}，输出: {content[:200]!r}",
            )
        try:
            model_cls(**data)
        except Exception as e:  # noqa: BLE001 - pydantic ValidationError 等
            return AssertionResult(
                kind="response_model",
                name=case.response_model,
                passed=False,
                message=f"输出不符合模型 {case.response_model}: {e}",
            )
        return AssertionResult(
            kind="response_model", name=case.response_model, passed=True, message="结构校验通过"
        )

    async def _check_rubric(self, case: EvalCase, content: str) -> AssertionResult:
        if not case.rubric:
            return AssertionResult(
                kind="rubric", name="-", passed=True, skipped=True, message="未配置"
            )
        if self.judge is None:
            return AssertionResult(
                kind="rubric",
                name=case.rubric[:80],
                passed=True,
                skipped=True,
                message="未提供 judge，跳过 rubric 评估",
            )
        try:
            score, explanation = await self.judge(case.input, content, case.rubric)
        except Exception as e:  # noqa: BLE001
            return AssertionResult(
                kind="rubric",
                name=case.rubric[:80],
                passed=False,
                message=f"judge 调用异常: {e}",
            )
        passed = float(score) >= case.threshold
        return AssertionResult(
            kind="rubric",
            name=case.rubric[:80],
            passed=passed,
            message=f"score={score} threshold={case.threshold} - {explanation}",
        )
