"""命令行入口：python -m wuwei.evals run <dataset.yaml|目录> [选项]"""

import argparse
import sys

from wuwei.evals.case import import_from_string
from wuwei.evals.dataset import load_dataset
from wuwei.evals.runner import EvalRunner

DEFAULT_AGENT = "wuwei.agent:Agent.from_env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wuwei.evals",
        description="wuwei 内建评估框架：对 agent 数据集运行评估并输出报告",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行评估数据集")
    run.add_argument("dataset", help="数据集 yaml 文件或目录")
    run.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        help="agent 工厂，格式 module:func（调用 func() 返回 Agent），默认 wuwei.agent:Agent.from_env",
    )
    run.add_argument("--judge", default=None, help="LLM-as-judge，格式 module:async_func")
    run.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="通过率及格线，低于该值退出码为 1（默认 1.0）",
    )
    run.add_argument("--json", dest="json_out", default=None, help="将 json 报告写入该路径")
    run.add_argument("--junit", dest="junit_out", default=None, help="将 junit xml 写入该路径（CI 用）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
    except Exception as e:  # noqa: BLE001 - CLI 层把加载错误变成可读退出码
        print(f"加载数据集失败: {args.dataset} -> {e}", file=sys.stderr)
        return 2
    if not dataset.cases:
        print(f"数据集为空: {args.dataset}", file=sys.stderr)
        return 2

    try:
        agent_factory = import_from_string(args.agent)
    except Exception as e:  # noqa: BLE001
        print(f"导入 agent 工厂失败: {args.agent} -> {e}", file=sys.stderr)
        return 2

    judge = None
    if args.judge:
        try:
            judge = import_from_string(args.judge)
        except Exception as e:  # noqa: BLE001
            print(f"导入 judge 失败: {args.judge} -> {e}", file=sys.stderr)
            return 2

    import asyncio

    runner = EvalRunner(agent_factory=agent_factory, judge=judge)
    report = asyncio.run(runner.run(dataset))

    print(report.summary())

    if args.json_out:
        from pathlib import Path

        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report.to_json(), encoding="utf-8")
        print(f"json 报告已写入: {p}")

    if args.junit_out:
        p = report.write_junit(args.junit_out)
        print(f"junit xml 已写入: {p}")

    if report.pass_rate < args.fail_under:
        print(
            f"通过率 {report.pass_rate:.1%} 低于及格线 {args.fail_under:.0%}，评估失败",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
