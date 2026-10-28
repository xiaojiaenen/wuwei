"""评估用例与结果模型。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def import_from_string(path: str) -> Any:
    """按 "module:Attr" 形式导入对象（支持 module 内多级属性，如 module:Class.attr）。

    也兼容 "module.Attr" 写法（当不含冒号时，从右往左尝试拆分）。
    """
    if not path or not isinstance(path, str):
        raise ValueError(f"无效的导入路径: {path!r}")

    module_path: str
    attr_path: str
    if ":" in path:
        module_path, _, attr_path = path.partition(":")
    else:
        # 无冒号：从最右侧拆出属性
        parts = path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"无效的导入路径（缺少 module:Attr 形式）: {path!r}")
        module_path, attr_path = parts

    import importlib

    module = importlib.import_module(module_path)
    obj: Any = module
    for attr in attr_path.split("."):
        if not attr:
            raise ValueError(f"无效的导入路径: {path!r}")
        obj = getattr(obj, attr)
    return obj


class AssertionResult(BaseModel):
    """单条断言的结果。"""

    kind: str  # status / trajectory / content / schema / rubric / limits
    name: str  # 断言的具体内容（如工具名、子串、模型路径）
    passed: bool = False
    message: str = ""
    skipped: bool = False


class EvalCase(BaseModel):
    """一条评估用例。

    所有断言字段都是可选的；一条用例可以只测轨迹、只测内容，或全部组合。
    """

    name: str
    input: str

    turns: list[str] = Field(default_factory=list)
    expected_status: str = "completed"
    allow_empty_assertions: bool = False
    judge_required: bool = True
    tool_trajectory: Literal["requested", "executed", "succeeded"] = "succeeded"
    max_tokens: int | None = Field(default=None, ge=0)
    max_cost: float | None = Field(default=None, ge=0)
    max_latency_ms: int | None = Field(default=None, ge=0)

    # ── 轨迹断言 ──────────────────────────────────────────────────────
    expected_tool_calls: list[str] = Field(default_factory=list)  # 必须都出现
    forbidden_tool_calls: list[str] = Field(default_factory=list)  # 都不得出现

    # ── 内容断言 ──────────────────────────────────────────────────────
    response_contains: list[str] = Field(default_factory=list)
    response_not_contains: list[str] = Field(default_factory=list)

    # ── 结构断言 ──────────────────────────────────────────────────────
    response_model: str | None = None  # "module:ClassName"，校验输出 JSON

    # ── rubric（LLM-as-judge） ────────────────────────────────────────
    rubric: str | None = None
    threshold: float = 0.7  # rubric 得分及分线

    # ── 运行参数 ──────────────────────────────────────────────────────
    max_steps: int | None = None  # 覆盖 agent 默认步数
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold 必须在 [0, 1] 内，收到 {v}")
        return v


class EvalResult(BaseModel):
    """一条用例的评估结果。"""

    case_name: str
    passed: bool = False
    assertions: list[AssertionResult] = Field(default_factory=list)
    status: str = "unknown"
    termination_reason: str | None = None
    usage_available: bool = False
    skipped: bool = False
    tool_trajectories: dict[str, list[str]] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    latency_ms: int = 0
    # 运行期错误（agent.run 抛异常等）；为 None 表示正常跑完
    error: str | None = None
    output: str = ""  # agent 最终回复（便于报告排查）
    tool_calls: list[str] = Field(default_factory=list)  # 实际调用的工具名序列

    @property
    def has_rubric_skipped(self) -> bool:
        return any(a.skipped for a in self.assertions)
