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


def extract_tool_trajectories(session: Any) -> dict[str, list[str]]:
    """Requests are intentions; committed tool messages are execution evidence."""
    result = {"requested": [], "executed": [], "succeeded": []}
    calls = {}
    messages = session.context.get_messages()
    for message in messages:
        if getattr(message, "role", None) == "assistant":
            for call in getattr(message, "tool_calls", None) or []:
                calls[call.id] = call.function.name
                result["requested"].append(call.function.name)
    for message in messages:
        if getattr(message, "role", None) != "tool":
            continue
        name = getattr(message, "name", None) or calls.get(getattr(message, "tool_call_id", None))
        if not name:
            continue
        metadata = getattr(message, "metadata", {}) or {}
        outcome = metadata.get("outcome") or metadata.get("tool_outcome") or {}
        if hasattr(outcome, "model_dump"):
            outcome = outcome.model_dump()
        status = outcome.get("status", getattr(message, "status", "unknown"))
        if outcome.get("executed", status not in {"denied", "rejected", "blocked", "skipped"}):
            result["executed"].append(name)
        if status == "success":
            result["succeeded"].append(name)
    return result


def extract_tool_calls(session: Any) -> list[str]:
    """Return committed successful calls (not assistant intentions)."""
    return extract_tool_trajectories(session)["succeeded"]


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
        agent, result = None, None
        try:
            agent = self.agent_factory()
            result = await self._run_case(agent, case)
        except Exception as exc:
            result = EvalResult(
                case_name=case.name,
                status="error",
                passed=False,
                error=repr(exc),
                termination_reason="evaluation_setup_error",
            )
        finally:
            close = getattr(agent, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception as exc:
                    if result is not None:
                        result.passed = False
                        result.assertions.append(
                            AssertionResult(
                                kind="lifecycle",
                                name="agent.aclose",
                                passed=False,
                                message=f"Agent cleanup failed: {type(exc).__name__}",
                            )
                        )
        return result

    async def _run_case(self, agent, case: EvalCase) -> EvalResult:
        session = self._make_session(agent, case)

        start = time.monotonic()
        try:
            usage = {}
            usage_available = True
            for turn in [case.input, *case.turns]:
                run_result = await agent.run(turn, session=session)
                turn_usage = getattr(run_result, "usage", None)
                usage_available = usage_available and getattr(
                    run_result, "usage_available", bool(turn_usage)
                )
                for key, value in (turn_usage or {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
                if getattr(run_result, "status", "unknown") != "completed":
                    break
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
                status="error",
                termination_reason="agent_exception",
                usage_available=False,
            )

        latency = int((time.monotonic() - start) * 1000)
        content = getattr(run_result, "content", "") or ""
        trajectories = extract_tool_trajectories(session)
        actual_tools = trajectories[case.tool_trajectory]
        status = getattr(run_result, "status", "unknown")

        assertions: list[AssertionResult] = []
        configured = any(
            (
                case.expected_tool_calls,
                case.forbidden_tool_calls,
                case.response_contains,
                case.response_not_contains,
                case.response_model,
                case.rubric,
                case.max_tokens is not None,
                case.max_cost is not None,
                case.max_latency_ms is not None,
            )
        )
        if not configured and not case.allow_empty_assertions:
            assertions.append(
                AssertionResult(
                    kind="configuration", name="assertions", message="No assertions configured"
                )
            )
        for name, limit, value in (
            ("tokens", case.max_tokens, usage.get("total_tokens") if usage_available else None),
            ("cost", case.max_cost, usage.get("cost", usage.get("estimated_cost"))),
            ("latency_ms", case.max_latency_ms, latency),
        ):
            if limit is not None:
                assertions.append(
                    AssertionResult(
                        kind="limit",
                        name=name,
                        passed=value is not None and value <= limit,
                        message=f"{name}: {value!r} <= {limit}",
                    )
                )
        assertions.extend(self._check_tool_calls(case, actual_tools))
        assertions.extend(self._check_content(case, content))
        assertions.append(await self._check_response_model(case, content))
        assertions.append(await self._check_rubric(case, content))

        assertions.append(
            AssertionResult(
                kind="status",
                name=case.expected_status,
                passed=status == case.expected_status,
                message=f"expected {case.expected_status}, actual {status}",
            )
        )
        passed = all(a.passed for a in assertions if not a.skipped)
        return EvalResult(
            case_name=case.name,
            passed=passed,
            assertions=assertions,
            usage=usage,
            latency_ms=latency,
            output=content,
            tool_calls=actual_tools,
            tool_trajectories=trajectories,
            status=status,
            termination_reason=getattr(run_result, "termination_reason", None),
            usage_available=usage_available,
            skipped=any(a.skipped and a.kind == "rubric" and a.name != "-" for a in assertions)
            and passed,
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
                passed=not case.judge_required,
                skipped=not case.judge_required,
                message="未提供 judge",
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
