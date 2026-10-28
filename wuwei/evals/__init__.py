"""wuwei 内建评估框架（evals）。

用法::

    from wuwei.evals import EvalCase, EvalRunner, load_dataset

    runner = EvalRunner(agent_factory=lambda: Agent.from_env(tools=[...]), judge=my_judge)
    report = await runner.run(load_dataset("evals/smoke.yaml"))
    print(report.summary())
"""

from wuwei.evals.case import (
    AssertionResult,
    EvalCase,
    EvalResult,
    import_from_string,
)
from wuwei.evals.cli import main as main
from wuwei.evals.dataset import Dataset, load_dataset
from wuwei.evals.report import EvalReport
from wuwei.evals.runner import EvalRunner, extract_tool_calls

__all__ = [
    "AssertionResult",
    "Dataset",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "extract_tool_calls",
    "import_from_string",
    "load_dataset",
    "main",
]
