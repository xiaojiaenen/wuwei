"""数据集加载：YAML 文件（支持 include 其它文件）与目录聚合。"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from wuwei.evals.case import EvalCase

VALID_KEYS = {"name", "cases", "include"}


class Dataset(BaseModel):
    """一组评估用例的聚合。"""

    name: str = "dataset"
    cases: list[EvalCase] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # 来源文件（调试用）

    def __len__(self) -> int:
        return len(self.cases)


def _load_yaml_file(path: Path, _visited: set[Path]) -> tuple[str, list[EvalCase]]:
    """加载单个 yaml 文件，递归处理 include。返回 (数据集名, 用例列表)。"""
    real = path.resolve()
    if real in _visited:
        raise ValueError(f"数据集 include 存在循环引用: {path}")
    _visited.add(real)

    if not path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"数据集文件顶层必须是映射: {path}")

    unknown = set(data) - VALID_KEYS
    if unknown:
        raise ValueError(f"数据集文件含未知字段 {sorted(unknown)}: {path}")

    name = str(data.get("name", path.stem))
    cases: list[EvalCase] = []

    # include：相对当前文件的其它数据集文件（或目录）
    for inc in data.get("include", []) or []:
        inc_path = (path.parent / inc).resolve()
        sub = load_dataset(inc_path, _visited=_visited)
        cases.extend(sub.cases)

    for raw in data.get("cases", []) or []:
        if not isinstance(raw, dict):
            raise ValueError(f"用例必须是映射: {path} -> {raw!r}")
        if not raw.get("name"):
            raise ValueError(f"用例缺少 name 字段: {path} -> {raw!r}")
        cases.append(EvalCase(**raw))

    return name, cases


def load_dataset(path: str | Path, _visited: set[Path] | None = None) -> Dataset:
    """加载数据集。

    Args:
        path: yaml 文件或目录（目录时递归收集 *.yaml / *.yml，按文件名排序合并）。
    """
    if _visited is None:
        _visited = set()

    p = Path(path)
    if p.is_dir():
        files = sorted(
            [*p.glob("*.yaml"), *p.glob("*.yml")],
            key=lambda f: f.name,
        )
        if not files:
            raise FileNotFoundError(f"目录中没有 yaml 数据集文件: {p}")
        name = p.name
        cases: list[EvalCase] = []
        sources: list[str] = []
        for f in files:
            sub_name, sub_cases = _load_yaml_file(f, _visited)
            cases.extend(sub_cases)
            sources.append(str(f))
        return Dataset(name=name, cases=cases, sources=sources)

    name, cases = _load_yaml_file(p, _visited)
    return Dataset(name=name, cases=cases, sources=[str(p.resolve())])
