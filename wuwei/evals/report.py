"""评估报告：汇总统计、text/json 输出、junit xml（CI 用）。"""

import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, Field

from wuwei.evals.case import EvalResult


class EvalReport(BaseModel):
    """一次评估运行的汇总报告。"""

    dataset_name: str = "dataset"
    results: list[EvalResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count

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
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"[{mark}] {r.case_name}  ({r.latency_ms}ms, {r.usage.get('total_tokens', 0)} tokens)")
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
                "skipped": "0",
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
            if r.passed:
                continue
            failure_msg = r.error or "; ".join(
                a.message for a in r.assertions if not a.passed and not a.skipped
            )
            failure = ET.SubElement(testcase, "failure", {"message": failure_msg[:200] or "eval failed"})
            failure.text = failure_msg

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(testsuite).write(p, encoding="utf-8", xml_declaration=True)
        return p
