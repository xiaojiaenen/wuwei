"""评估报告：汇总统计、text/json 输出、junit xml（CI 用）。"""

import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from wuwei.evals.case import EvalResult


class EvalReport(BaseModel):
    """一次评估运行的汇总报告。"""

    dataset_name: str = "dataset"
    results: list[EvalResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_case_names(self):
        names = [r.case_name for r in self.results]
        if len(set(names)) != len(names):
            raise ValueError(
                "Evaluation case names must be unique for reliable baseline comparison"
            )
        return self

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count - self.skipped_count

    @property
    def skipped_count(self) -> int:
        return sum(r.skipped for r in self.results)

    def compare(self, baseline: "EvalReport") -> dict:
        """Compare case identity, not aggregate pass rates that hide regressions."""
        previous = {r.case_name: r for r in baseline.results}
        current = {r.case_name: r for r in self.results}
        return {
            "regressions": [
                name
                for name, old in previous.items()
                if old.passed
                and not old.skipped
                and (name not in current or not current[name].passed or current[name].skipped)
            ],
            "added": sorted(current.keys() - previous.keys()),
            "removed": sorted(previous.keys() - current.keys()),
            "token_delta": {
                name: (
                    result.usage.get("total_tokens", 0)
                    - previous[name].usage.get("total_tokens", 0)
                )
                if result.usage_available and previous[name].usage_available
                else None
                for name, result in current.items()
                if name in previous
            },
            "tool_trajectory_changed": [
                name
                for name, result in current.items()
                if name in previous and result.tool_trajectories != previous[name].tool_trajectories
            ],
            "latency_delta_ms": {
                name: result.latency_ms - previous[name].latency_ms
                for name, result in current.items()
                if name in previous
            },
        }

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    # ── 输出 ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "total": self.total,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "pass_rate": round(self.pass_rate, 4),
            "results": [r.model_dump() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        """人类可读的 text 摘要。"""
        lines = [
            f"评估数据集: {self.dataset_name}",
            f"用例总数: {self.total}  通过: {self.passed_count}  "
            f"失败: {self.failed_count}  通过率: {self.pass_rate:.1%}",
            "-" * 60,
        ]
        for r in self.results:
            mark = "SKIP" if r.skipped else "PASS" if r.passed else "FAIL"
            lines.append(
                f"[{mark}] {r.case_name}  ({r.latency_ms}ms, "
                f"{r.usage.get('total_tokens', 0)} tokens)"
            )
            if r.passed:
                continue
            if r.error:
                lines.append(f"       运行错误: {r.error}")
            for a in r.assertions:
                if a.skipped:
                    continue
                if not a.passed:
                    lines.append(f"       [{a.kind}] {a.message}")
        lines.append("-" * 60)
        return "\n".join(lines)

    # ── JUnit XML ────────────────────────────────────────────────────

    def write_junit(self, path: str | Path) -> Path:
        """生成 junit xml（每个用例一个 testcase，失败写 <failure>）。"""
        testsuite = ET.Element(
            "testsuite",
            {
                "name": f"wuwei.evals.{self.dataset_name}",
                "tests": str(self.total),
                "failures": str(self.failed_count),
                "errors": "0",
                "skipped": str(self.skipped_count),
                "time": f"{sum(r.latency_ms for r in self.results) / 1000:.3f}",
            },
        )
        for r in self.results:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {
                    "classname": self.dataset_name,
                    "name": r.case_name,
                    "time": f"{r.latency_ms / 1000:.3f}",
                },
            )
            if r.skipped:
                ET.SubElement(testcase, "skipped", {"message": "Optional judge unavailable"})
                continue
            if r.passed:
                continue
            failure_msg = r.error or "; ".join(
                a.message for a in r.assertions if not a.passed and not a.skipped
            )
            failure = ET.SubElement(
                testcase, "failure", {"message": failure_msg[:200] or "eval failed"}
            )
            failure.text = failure_msg

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(testsuite).write(p, encoding="utf-8", xml_declaration=True)
        return p
