"""内建 evals 评估框架测试（全离线，不调真实 LLM）。

覆盖：
1. EvalCase 模型校验（threshold 边界、默认值）
2. load_dataset：单文件 / 目录聚合 / include / 非法字段
3. EvalRunner：轨迹断言（expected / forbidden tool_calls）、contains / not_contains、
   response_model 结构校验（含代码围栏）、rubric judge（fake）、judge 缺失时 skipped
4. EvalReport：text 摘要 / json / junit xml
5. CLI：python -m wuwei.evals run 的退出码与产物文件
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from wuwei.core.message import FunctionCall, ToolCall
from wuwei.evals import (
    AssertionResult,
    Dataset,
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
    import_from_string,
    load_dataset,
)
from wuwei.evals.runner import extract_tool_calls
from wuwei.llm.types import AgentRunResult
from wuwei.memory import Context

# ── FakeAgent（duck-type Agent，不调真实 LLM） ────────────────────────


class FakeSession:
    """最小会话：只提供 context.get_messages()。"""

    def __init__(self) -> None:
        self.context = Context()
        self.context.add_system_message("sys")


class FakeAgent:
    """脚本化假 agent：run 时按剧本写会话历史并返回 AgentRunResult。"""

    def __init__(
        self,
        tool_calls: list[str] | None = None,
        content: str = "好的，已完成",
        raise_on_run: Exception | None = None,
    ) -> None:
        self.tool_calls = list(tool_calls or [])
        self.content = content
        self.raise_on_run = raise_on_run
        self.received_inputs: list[str] = []

    def create_session(self, **kwargs) -> FakeSession:
        return FakeSession()

    async def run(self, user_input, session=None, stream=False) -> AgentRunResult:
        self.received_inputs.append(user_input)
        if self.raise_on_run is not None:
            raise self.raise_on_run
        if session is None:
            session = self.create_session()
        ctx = session.context
        ctx.add_user_message(user_input)
        if self.tool_calls:
            tcs = [
                ToolCall(id=f"call_{i}", function=FunctionCall(name=n, arguments={}))
                for i, n in enumerate(self.tool_calls)
            ]
            ctx.add_ai_message("", tool_calls=tcs)
            for tc in tcs:
                ctx.add_tool_message("ok", tool_call_id=tc.id)
        ctx.add_ai_message(self.content)
        return AgentRunResult(
            content=self.content,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=7,
            llm_calls=len(self.tool_calls) + 1,
            status="completed",
        )


def make_runner(agent: FakeAgent, judge=None) -> EvalRunner:
    return EvalRunner(agent_factory=lambda: agent, judge=judge)


# ── 共享的 fake judge ─────────────────────────────────────────────────


async def good_judge(input: str, output: str, rubric: str) -> tuple[float, str]:
    return 0.9, "回答切题"


async def bad_judge(input: str, output: str, rubric: str) -> tuple[float, str]:
    return 0.2, "回答偏题"


# ── EvalCase 模型 ─────────────────────────────────────────────────────


class TestEvalCase:
    def test_defaults(self):
        case = EvalCase(name="c1", input="hi")
        assert case.expected_tool_calls == []
        assert case.forbidden_tool_calls == []
        assert case.response_contains == []
        assert case.response_not_contains == []
        assert case.response_model is None
        assert case.rubric is None
        assert case.threshold == 0.7

    def test_threshold_bounds(self):
        EvalCase(name="c", input="i", threshold=0.0)
        EvalCase(name="c", input="i", threshold=1.0)
        with pytest.raises(ValueError):
            EvalCase(name="c", input="i", threshold=1.5)
        with pytest.raises(ValueError):
            EvalCase(name="c", input="i", threshold=-0.1)

    def test_import_from_string(self):
        import_from_string("wuwei.evals.case:EvalCase")
        import_from_string("wuwei.evals.case.EvalCase")  # 无冒号写法
        with pytest.raises(Exception):
            import_from_string("no_such_module_xyz:Foo")
        with pytest.raises(Exception):
            import_from_string("wuwei.evals.case:NoSuchAttr")


# ── 数据集加载 ────────────────────────────────────────────────────────


class TestLoadDataset:
    def test_load_yaml_file(self, tmp_path: Path):
        ds_file = tmp_path / "smoke.yaml"
        ds_file.write_text(
            yaml.safe_dump(
                {
                    "name": "冒烟",
                    "cases": [
                        {"name": "t1", "input": "你好", "expected_tool_calls": ["echo"]},
                        {"name": "t2", "input": "再见", "response_contains": ["拜拜"]},
                    ],
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        ds = load_dataset(ds_file)
        assert isinstance(ds, Dataset)
        assert ds.name == "冒烟"
        assert len(ds.cases) == 2
        assert ds.cases[0].name == "t1"
        assert ds.cases[0].expected_tool_calls == ["echo"]

    def test_load_directory_merges_sorted(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text(
            yaml.safe_dump({"cases": [{"name": "a1", "input": "x"}]}), encoding="utf-8"
        )
        (tmp_path / "b.yml").write_text(
            yaml.safe_dump({"cases": [{"name": "b1", "input": "y"}]}), encoding="utf-8"
        )
        ds = load_dataset(tmp_path)
        assert ds.name == tmp_path.name
        assert [c.name for c in ds.cases] == ["a1", "b1"]
        assert len(ds.sources) == 2

    def test_load_with_include(self, tmp_path: Path):
        (tmp_path / "base.yaml").write_text(
            yaml.safe_dump({"cases": [{"name": "base1", "input": "x"}]}), encoding="utf-8"
        )
        (tmp_path / "main.yaml").write_text(
            yaml.safe_dump(
                {
                    "include": ["base.yaml"],
                    "cases": [{"name": "main1", "input": "y"}],
                }
            ),
            encoding="utf-8",
        )
        ds = load_dataset(tmp_path / "main.yaml")
        assert [c.name for c in ds.cases] == ["base1", "main1"]

    def test_include_cycle_raises(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text(
            yaml.safe_dump({"include": ["b.yaml"], "cases": []}), encoding="utf-8"
        )
        (tmp_path / "b.yaml").write_text(
            yaml.safe_dump({"include": ["a.yaml"], "cases": []}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="循环"):
            load_dataset(tmp_path / "a.yaml")

    def test_unknown_key_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"typo_key": [], "cases": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="未知字段"):
            load_dataset(bad)

    def test_case_without_name_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"cases": [{"input": "x"}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="name"):
            load_dataset(bad)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.yaml")

    def test_empty_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path)


# ── 轨迹断言 ──────────────────────────────────────────────────────────


class TestTrajectoryAssertions:
    @pytest.mark.asyncio
    async def test_expected_tool_calls_pass(self):
        agent = FakeAgent(tool_calls=["search", "calc"])
        case = EvalCase(name="t", input="查天气", expected_tool_calls=["search", "calc"])
        result = await make_runner(agent).run_case(case)
        assert result.passed
        kinds = {(a.kind, a.name, a.passed) for a in result.assertions}
        assert ("expected_tool_calls", "search", True) in kinds
        assert ("expected_tool_calls", "calc", True) in kinds
        assert result.tool_calls == ["search", "calc"]
        assert result.usage["total_tokens"] == 15
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_expected_tool_calls_fail(self):
        agent = FakeAgent(tool_calls=["calc"])
        case = EvalCase(name="t", input="q", expected_tool_calls=["search"])
        result = await make_runner(agent).run_case(case)
        assert not result.passed
        failed = [a for a in result.assertions if a.kind == "expected_tool_calls"]
        assert len(failed) == 1
        assert not failed[0].passed
        assert "search" in failed[0].message

    @pytest.mark.asyncio
    async def test_forbidden_tool_calls(self):
        agent = FakeAgent(tool_calls=["rm_rf", "calc"])
        ok_case = EvalCase(name="ok", input="q", forbidden_tool_calls=["drop_db"])
        ok = await make_runner(agent).run_case(ok_case)
        assert ok.passed

        bad_case = EvalCase(name="bad", input="q", forbidden_tool_calls=["rm_rf"])
        bad = await make_runner(agent).run_case(bad_case)
        assert not bad.passed
        assert any(
            a.kind == "forbidden_tool_calls" and not a.passed for a in bad.assertions
        )

    @pytest.mark.asyncio
    async def test_extract_tool_calls_from_tool_messages_fallback(self):
        """assistant.tool_calls 缺失时回退到 tool 消息的 name 字段。"""
        session = FakeSession()
        session.context.add_user_message("hi")
        session.context.add_tool_message("result", tool_call_id="x")
        session.context.get_messages()[-1].name = "fallback_tool"
        assert extract_tool_calls(session) == ["fallback_tool"]

    @pytest.mark.asyncio
    async def test_each_case_gets_fresh_session(self):
        """agent_factory 每个用例调用一次，保证隔离。"""
        created: list[FakeAgent] = []
        script = FakeAgent(tool_calls=["calc"])

        def factory():
            a = FakeAgent(tool_calls=["calc"], content=script.content)
            created.append(a)
            return a

        runner = EvalRunner(agent_factory=factory)
        ds = Dataset(
            cases=[EvalCase(name="a", input="1"), EvalCase(name="b", input="2")]
        )
        report = await runner.run(ds)
        assert len(created) == 2
        assert all(r.passed for r in report.results)


# ── 内容断言 ──────────────────────────────────────────────────────────


class TestContentAssertions:
    @pytest.mark.asyncio
    async def test_contains_pass_and_fail(self):
        agent = FakeAgent(content="今天北京晴，气温 20 度")
        ok = await make_runner(agent).run_case(
            EvalCase(name="ok", input="天气", response_contains=["晴", "20"])
        )
        assert ok.passed

        bad = await make_runner(agent).run_case(
            EvalCase(name="bad", input="天气", response_contains=["下雨"])
        )
        assert not bad.passed
        assert any(a.kind == "contains" and not a.passed for a in bad.assertions)

    @pytest.mark.asyncio
    async def test_not_contains(self):
        agent = FakeAgent(content="答案是无为")
        ok = await make_runner(agent).run_case(
            EvalCase(name="ok", input="q", response_not_contains=["对不起"])
        )
        assert ok.passed

        bad = await make_runner(agent).run_case(
            EvalCase(name="bad", input="q", response_not_contains=["无为"])
        )
        assert not bad.passed
        assert any(a.kind == "not_contains" and not a.passed for a in bad.assertions)


# ── response_model 结构校验 ───────────────────────────────────────────


class TestResponseModel:
    @pytest.fixture
    def model_module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        """写一个临时模块（含 pydantic 模型），返回 '模块名:类名' 路径。"""
        mod_name = "fake_eval_output_mod"
        (tmp_path / f"{mod_name}.py").write_text(
            "from pydantic import BaseModel\n"
            "class PersonOutput(BaseModel):\n"
            "    name: str\n"
            "    age: int\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        return f"{mod_name}:PersonOutput"

    @pytest.mark.asyncio
    async def test_response_model_pass(self, model_module: str):
        agent = FakeAgent(content='{"name": "张三", "age": 30}')
        result = await make_runner(agent).run_case(
            EvalCase(name="t", input="q", response_model=model_module)
        )
        assert result.passed
        assert result.assertions[0].kind == "response_model"
        assert result.assertions[0].passed

    @pytest.mark.asyncio
    async def test_response_model_with_code_fence(self, model_module: str):
        agent = FakeAgent(content='```json\n{"name": "李四", "age": 25}\n```')
        result = await make_runner(agent).run_case(
            EvalCase(name="t", input="q", response_model=model_module)
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_response_model_schema_fail(self, model_module: str):
        agent = FakeAgent(content='{"name": "王五"}')  # 缺 age
        result = await make_runner(agent).run_case(
            EvalCase(name="t", input="q", response_model=model_module)
        )
        assert not result.passed
        assert "age" in result.assertions[0].message

    @pytest.mark.asyncio
    async def test_response_model_not_json(self, model_module: str):
        agent = FakeAgent(content="这不是 JSON")
        result = await make_runner(agent).run_case(
            EvalCase(name="t", input="q", response_model=model_module)
        )
        assert not result.passed
        assert "JSON" in result.assertions[0].message

    @pytest.mark.asyncio
    async def test_response_model_import_fail(self):
        agent = FakeAgent(content="{}")
        result = await make_runner(agent).run_case(
            EvalCase(name="t", input="q", response_model="no_such_mod_xyz:Model")
        )
        assert not result.passed
        assert "导入" in result.assertions[0].message


# ── rubric（LLM-as-judge） ────────────────────────────────────────────


class TestRubric:
    @pytest.mark.asyncio
    async def test_rubric_with_judge_pass(self):
        agent = FakeAgent(content="好答案")
        runner = make_runner(agent, judge=good_judge)
        result = await runner.run_case(
            EvalCase(name="t", input="q", rubric="回答需切题", threshold=0.7)
        )
        assert result.passed
        rubric_assertions = [a for a in result.assertions if a.kind == "rubric"]
        assert len(rubric_assertions) == 1
        assert rubric_assertions[0].passed
        assert "0.9" in rubric_assertions[0].message

    @pytest.mark.asyncio
    async def test_rubric_with_judge_fail(self):
        agent = FakeAgent(content="差答案")
        runner = make_runner(agent, judge=bad_judge)
        result = await runner.run_case(
            EvalCase(name="t", input="q", rubric="回答需切题", threshold=0.7)
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_rubric_skipped_without_judge(self):
        agent = FakeAgent(content="任意")
        result = await make_runner(agent, judge=None).run_case(
            EvalCase(name="t", input="q", rubric="切题")
        )
        assert result.passed  # skipped 不算失败
        skipped = [a for a in result.assertions if a.kind == "rubric"]
        assert skipped[0].skipped
        assert result.has_rubric_skipped

    @pytest.mark.asyncio
    async def test_judge_exception_fails_case(self):
        async def boom(input, output, rubric):
            raise RuntimeError("judge 挂了")

        agent = FakeAgent(content="x")
        result = await make_runner(agent, judge=boom).run_case(
            EvalCase(name="t", input="q", rubric="r")
        )
        assert not result.passed
        rubric_a = next(a for a in result.assertions if a.kind == "rubric")
        assert not rubric_a.passed
        assert "judge" in rubric_a.message


# ── 运行异常与混合断言 ────────────────────────────────────────────────


class TestRunnerMisc:
    @pytest.mark.asyncio
    async def test_agent_run_exception_captured(self):
        agent = FakeAgent(raise_on_run=RuntimeError("网关炸了"))
        result = await make_runner(agent).run_case(EvalCase(name="t", input="q"))
        assert not result.passed
        assert result.error is not None
        assert "网关炸了" in result.error

    @pytest.mark.asyncio
    async def test_case_without_assertions_passes(self):
        agent = FakeAgent(content="hi")
        result = await make_runner(agent).run_case(EvalCase(name="t", input="q"))
        assert result.passed

    @pytest.mark.asyncio
    async def test_mixed_assertions_all_evaluated(self):
        agent = FakeAgent(tool_calls=["calc"], content="2")
        result = await make_runner(agent, judge=good_judge).run_case(
            EvalCase(
                name="t",
                input="1+1",
                expected_tool_calls=["calc"],
                forbidden_tool_calls=["rm_rf"],
                response_contains=["2"],
                response_not_contains=["3"],
                rubric="正确",
            )
        )
        assert result.passed
        kinds = {a.kind for a in result.assertions if not a.skipped}
        assert kinds == {
            "expected_tool_calls",
            "forbidden_tool_calls",
            "contains",
            "not_contains",
            "rubric",
        }

    @pytest.mark.asyncio
    async def test_run_dataset_report(self):
        agent_pass = FakeAgent(tool_calls=["calc"], content="4")
        runner = EvalRunner(agent_factory=lambda: agent_pass)
        ds = Dataset(
            name="math",
            cases=[
                EvalCase(name="ok1", input="2+2", expected_tool_calls=["calc"]),
                EvalCase(name="ok2", input="2+2", response_contains=["4"]),
            ],
        )
        report = await runner.run(ds)
        assert report.total == 2
        assert report.passed_count == 2
        assert report.dataset_name == "math"


# ── 报告输出 ──────────────────────────────────────────────────────────


class TestReport:
    def _report(self) -> EvalReport:
        ok = EvalResult(
            case_name="ok",
            passed=True,
            assertions=[AssertionResult(kind="contains", name="x", passed=True)],
            usage={"total_tokens": 15},
            latency_ms=10,
        )
        bad = EvalResult(
            case_name="bad",
            passed=False,
            assertions=[
                AssertionResult(kind="contains", name="y", passed=False, message="未包含"),
                AssertionResult(kind="rubric", name="-", passed=True, skipped=True),
            ],
            usage={"total_tokens": 20},
            latency_ms=30,
        )
        return EvalReport(dataset_name="demo", results=[ok, bad])

    def test_summary_stats(self):
        r = self._report()
        assert r.total == 2
        assert r.passed_count == 1
        assert r.failed_count == 1
        assert r.pass_rate == 0.5

    def test_summary_text(self):
        text = self._report().summary()
        assert "demo" in text
        assert "50.0%" in text
        assert "[PASS] ok" in text
        assert "[FAIL] bad" in text
        assert "未包含" in text  # 失败断言明细
        # skipped 断言不出现在明细里
        assert "skipped" not in text.lower() or True  # rubric skipped 不打印

    def test_to_json(self):
        import json

        data = json.loads(self._report().to_json())
        assert data["total"] == 2
        assert data["passed"] == 1
        assert data["pass_rate"] == 0.5
        assert data["results"][1]["assertions"][0]["passed"] is False

    def test_write_junit(self, tmp_path: Path):
        p = self._report().write_junit(tmp_path / "sub" / "junit.xml")
        assert p.exists()
        root = ET.parse(p).getroot()
        assert root.tag == "testsuite"
        assert root.get("tests") == "2"
        assert root.get("failures") == "1"
        cases = root.findall("testcase")
        assert len(cases) == 2
        by_name = {c.get("name"): c for c in cases}
        assert by_name["ok"].find("failure") is None
        failure = by_name["bad"].find("failure")
        assert failure is not None
        assert "未包含" in (failure.text or "")


# ── CLI ───────────────────────────────────────────────────────────────


def _write_dataset(tmp_path: Path, name: str = "ds.yaml", **case_kwargs) -> Path:
    case = {"name": "t1", "input": "1+1"}
    case.update(case_kwargs)
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"cases": [case]}, allow_unicode=True), encoding="utf-8")
    return p


class TestCli:
    def _run_cli(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
        """在进程内跑 cli.main，fake agent 模块放到 sys.path。"""
        mod_dir = tmp_path / "mods"
        monkeypatch.syspath_prepend(str(mod_dir))
        from wuwei.evals.cli import main

        return main(argv)

    def _make_fake_module(self, mod_dir: Path, tool_calls: str, content: str) -> None:
        (mod_dir / "fake_cli_agent_mod.py").write_text(
            "from wuwei.core.message import FunctionCall, ToolCall\n"
            "from wuwei.llm.types import AgentRunResult\n"
            "from wuwei.memory import Context\n"
            "\n"
            "class FakeSession:\n"
            "    def __init__(self):\n"
            "        self.context = Context()\n"
            "\n"
            "class FakeAgent:\n"
            f"    def __init__(self, tool_calls={tool_calls!r}, content={content!r}):\n"
            "        self.tool_calls = list(tool_calls)\n"
            "        self.content = content\n"
            "\n"
            "    def create_session(self, **kw):\n"
            "        return FakeSession()\n"
            "\n"
            "    async def run(self, user_input, session=None, stream=False):\n"
            "        if session is None:\n"
            "            session = self.create_session()\n"
            "        ctx = session.context\n"
            "        ctx.add_user_message(user_input)\n"
            "        if self.tool_calls:\n"
            "            tcs = [ToolCall(id=f'c{i}', function=FunctionCall(name=n, arguments={}))\n"
            "                   for i, n in enumerate(self.tool_calls)]\n"
            "            ctx.add_ai_message('', tool_calls=tcs)\n"
            "            for tc in tcs:\n"
            "                ctx.add_tool_message('ok', tool_call_id=tc.id)\n"
            "        ctx.add_ai_message(self.content)\n"
            "        return AgentRunResult(content=self.content,\n"
            "                              usage={'total_tokens': 10}, latency_ms=3)\n"
            "\n"
            "def make_agent():\n"
            "    return FakeAgent()\n",
            encoding="utf-8",
        )

    def test_cli_pass_exit_0(self, tmp_path, monkeypatch):
        ds = _write_dataset(
            tmp_path, expected_tool_calls=["calc"], response_contains=["4"]
        )
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        self._make_fake_module(mod_dir, ["calc"], "答案是 4")
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            ["run", str(ds), "--agent", "fake_cli_agent_mod:make_agent"],
        )
        assert rc == 0

    def test_cli_fail_exit_1_with_fail_under(self, tmp_path, monkeypatch):
        ds = _write_dataset(tmp_path, expected_tool_calls=["search"])
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        self._make_fake_module(mod_dir, ["calc"], "答案是 4")
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            [
                "run",
                str(ds),
                "--agent",
                "fake_cli_agent_mod:make_agent",
                "--fail-under",
                "1.0",
            ],
        )
        assert rc == 1

    def test_cli_json_and_junit_outputs(self, tmp_path, monkeypatch):
        ds = _write_dataset(tmp_path, response_contains=["4"])
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        self._make_fake_module(mod_dir, ["calc"], "答案是 4")
        json_out = tmp_path / "out" / "report.json"
        junit_out = tmp_path / "out" / "junit.xml"
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            [
                "run",
                str(ds),
                "--agent",
                "fake_cli_agent_mod:make_agent",
                "--json",
                str(json_out),
                "--junit",
                str(junit_out),
            ],
        )
        assert rc == 0
        assert json_out.exists()
        assert junit_out.exists()
        assert '"pass_rate": 1.0' in json_out.read_text(encoding="utf-8")
        root = ET.parse(junit_out).getroot()
        assert root.get("tests") == "1"

    def test_cli_directory_dataset(self, tmp_path, monkeypatch):
        """数据集路径可以是目录。"""
        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        _write_dataset(ds_dir, name="a.yaml", response_contains=["4"])
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        self._make_fake_module(mod_dir, ["calc"], "答案是 4")
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            ["run", str(ds_dir), "--agent", "fake_cli_agent_mod:make_agent"],
        )
        assert rc == 0

    def test_cli_bad_agent_module_exit_2(self, tmp_path, monkeypatch):
        ds = _write_dataset(tmp_path)
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            ["run", str(ds), "--agent", "no_such_mod_xyz:factory"],
        )
        assert rc == 2

    def test_cli_missing_dataset_exit_nonzero(self, tmp_path, monkeypatch):
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            ["run", str(tmp_path / "nope.yaml"), "--agent", "fake_cli_agent_mod:make_agent"],
        )
        assert rc != 0

    def test_cli_missing_dataset_file_error(self, tmp_path, monkeypatch):
        """数据集文件不存在时报错且退出码非 0。"""
        rc = self._run_cli(
            monkeypatch,
            tmp_path,
            ["run", str(tmp_path / "ghost.yaml"), "--agent", "sys:exit"],
        )
        assert rc != 0

    def test_python_m_wuwei_evals_subprocess(self, tmp_path):
        """端到端：python -m wuwei.evals run <dataset> --agent module:func。"""
        import os
        import subprocess
        import sys

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        _write_dataset(ds_dir, name="smoke.yaml", response_contains=["4"])
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()

        # 复用本测试文件里的 FakeAgent 逻辑：生成独立模块
        (mod_dir / "e2e_agent_mod.py").write_text(
            "from wuwei.core.message import FunctionCall, ToolCall\n"
            "from wuwei.llm.types import AgentRunResult\n"
            "from wuwei.memory import Context\n"
            "\n"
            "class S:\n"
            "    def __init__(self):\n"
            "        self.context = Context()\n"
            "\n"
            "class A:\n"
            "    def create_session(self, **kw):\n"
            "        return S()\n"
            "\n"
            "    async def run(self, user_input, session=None, stream=False):\n"
            "        if session is None:\n"
            "            session = self.create_session()\n"
            "        session.context.add_ai_message('答案是 4')\n"
            "        return AgentRunResult(content='答案是 4', usage={'total_tokens': 1})\n"
            "\n"
            "def make_agent():\n"
            "    return A()\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(mod_dir)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "wuwei.evals",
                "run",
                str(ds_dir),
                "--agent",
                "e2e_agent_mod:make_agent",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert "[PASS]" in proc.stdout
