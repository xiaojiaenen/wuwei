import json

import pytest

from wuwei.agent.session import AgentSession
from wuwei.runtime.skill_hook import DEFAULT_SKILL_INSTRUCTION, SkillHook
from wuwei.skill.fs_provider import FileSystemSkillProvider
from wuwei.skill.skill import SkillManager
from wuwei.tools import ToolRegistry
from wuwei.tools.builtin import register_skill_tools
from wuwei.tools.builtin import skill_tools as skill_tools_module


@pytest.mark.asyncio
async def test_skill_hook_appends_generic_skill_guidance_to_system_prompt() -> None:
    session = AgentSession(
        session_id="session-1",
        system_prompt="你是一个有用的助手",
    )
    messages = [message.model_copy(deep=True) for message in session.context.get_messages()]

    updated_messages, updated_tools = await SkillHook().before_llm(
        session,
        messages,
        [],
        step=0,
        task=None,
    )

    assert updated_tools == []
    assert updated_messages[0].role == "system"
    assert updated_messages[0].content == f"你是一个有用的助手\n\n{DEFAULT_SKILL_INSTRUCTION}"
    assert session.context.get_messages()[0].content == "你是一个有用的助手"


@pytest.mark.asyncio
async def test_skill_tools_list_load_and_run_python_script(tmp_path) -> None:
    skill_dir = tmp_path / "weather"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_file = scripts_dir / "echo.py"
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    reference_file = references_dir / "weather.md"

    skill_file.write_text(
        "---\n"
        "name: weather_analyst\n"
        "description: 天气分析\n"
        "---\n\n"
        "优先调用天气工具，并输出城市、天气、温度。\n",
        encoding="utf-8",
    )
    script_file.write_text(
        "import json\n"
        "import sys\n"
        "\n"
        "print(json.dumps({'args': sys.argv[1:]}, ensure_ascii=False))\n",
        encoding="utf-8",
    )
    reference_file.write_text("天气分析参考资料\n第二行", encoding="utf-8")

    provider = FileSystemSkillProvider(str(tmp_path))
    manager = SkillManager([provider])
    registry = ToolRegistry()
    register_skill_tools(registry, manager)

    list_result = await registry.get("list_skills").invoke({})
    load_result = await registry.get("load_skill").invoke({"skill_name": "weather_analyst"})
    reference_result = await registry.get("load_skill_reference").invoke(
        {
            "skill_name": "weather_analyst",
            "reference_path": "references/weather.md",
            "load_token": load_result["load_token"],
            "max_chars": 6,
        }
    )
    run_result = await registry.get("run_skill_python_script").invoke(
        {
            "skill_name": "weather_analyst",
            "script_path": "scripts/echo.py",
            "load_token": load_result["load_token"],
            "args_json": json.dumps(["北京", "晴"]),
        }
    )

    assert len(list_result) == 1
    assert list_result[0]["name"] == "weather_analyst"
    assert list_result[0]["has_python_scripts"] is True
    assert list_result[0]["python_scripts"] == ["scripts/echo.py"]
    assert list_result[0]["has_references"] is True
    assert list_result[0]["references"] == ["references/weather.md"]
    assert load_result["name"] == "weather_analyst"
    assert load_result["instruction"] == "优先调用天气工具，并输出城市、天气、温度。"
    assert load_result["python_scripts"] == ["scripts/echo.py"]
    assert load_result["references"] == ["references/weather.md"]
    assert isinstance(load_result["load_token"], str)
    assert load_result["load_token"]
    assert reference_result["ok"] is True
    assert reference_result["content"] == "天气分析参考"
    assert reference_result["truncated"] is True
    assert reference_result["size_chars"] == len("天气分析参考资料\n第二行")
    assert run_result["ok"] is True
    assert run_result["returncode"] == 0
    assert json.loads(run_result["stdout"]) == {"args": ["北京", "晴"]}


@pytest.mark.asyncio
async def test_run_skill_python_script_requires_load_token(tmp_path) -> None:
    skill_dir = tmp_path / "weather"
    skill_dir.mkdir()
    (skill_dir / "scripts").mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: weather_analyst\n"
        "description: 天气分析\n"
        "---\n\n"
        "优先调用天气工具，并输出城市、天气、温度。\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "echo.py").write_text("print('ok')\n", encoding="utf-8")

    provider = FileSystemSkillProvider(str(tmp_path))
    manager = SkillManager([provider])
    registry = ToolRegistry()
    register_skill_tools(registry, manager)

    with pytest.raises(ValueError, match="必须先调用 load_skill"):
        await registry.get("run_skill_python_script").invoke(
            {
                "skill_name": "weather_analyst",
                "script_path": "scripts/echo.py",
                "load_token": "missing-token",
            }
        )


@pytest.mark.asyncio
async def test_skill_tools_timeout_and_output_truncation(tmp_path, monkeypatch) -> None:
    skill_dir = tmp_path / "weather"
    skill_dir.mkdir()
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: weather_analyst\n"
        "description: 天气分析\n"
        "---\n\n"
        "优先调用天气工具，并输出城市、天气、温度。\n",
        encoding="utf-8",
    )
    (scripts_dir / "big.py").write_text(f"print({'x' * 5000!r})\n", encoding="utf-8")
    (scripts_dir / "sleep.py").write_text(
        "import time\n" "time.sleep(1)\n" "print('done')\n",
        encoding="utf-8",
    )

    provider = FileSystemSkillProvider(str(tmp_path))
    manager = SkillManager([provider])
    registry = ToolRegistry()
    register_skill_tools(registry, manager)

    load_result = await registry.get("load_skill").invoke({"skill_name": "weather_analyst"})
    load_token = load_result["load_token"]

    monkeypatch.setattr(skill_tools_module, "SKILL_SCRIPT_OUTPUT_LIMIT", 100)
    big_result = await registry.get("run_skill_python_script").invoke(
        {
            "skill_name": "weather_analyst",
            "script_path": "scripts/big.py",
            "load_token": load_token,
        }
    )

    assert big_result["ok"] is True
    assert "... [truncated " in big_result["stdout"]
    assert len(big_result["stdout"]) < 160

    monkeypatch.setattr(skill_tools_module, "SKILL_SCRIPT_TIMEOUT_SECONDS", 0.1)
    with pytest.raises(TimeoutError, match="Skill 脚本执行超时"):
        await registry.get("run_skill_python_script").invoke(
            {
                "skill_name": "weather_analyst",
                "script_path": "scripts/sleep.py",
                "load_token": load_token,
            }
        )


def test_skill_manager_raises_clear_error_for_missing_skill() -> None:
    manager = SkillManager([])

    with pytest.raises(ValueError, match="Skill 'missing' not found"):
        manager.get_skill("missing")

    with pytest.raises(ValueError, match="Skill 'missing' not found"):
        manager.load_skill_instruction("missing")


def test_filesystem_skill_provider_loads_skill_instruction_and_path(tmp_path) -> None:
    skill_dir = tmp_path / "weather"
    skill_dir.mkdir()
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: weather_analyst\n"
        "description: 天气分析\n"
        "---\n\n"
        "优先调用天气工具，并输出城市、天气、温度。\n",
        encoding="utf-8",
    )
    (scripts_dir / "echo.py").write_text("print('ok')\n", encoding="utf-8")
    (references_dir / "guide.md").write_text("参考资料\n", encoding="utf-8")

    provider = FileSystemSkillProvider(str(tmp_path))
    skills = provider.list_skills()

    assert len(skills) == 1
    assert skills[0].name == "weather_analyst"
    assert skills[0].path == str(skill_dir)
    assert skills[0].scripts == ["scripts/echo.py"]
    assert skills[0].references == ["references/guide.md"]
    assert provider.load_skill_instruction("weather_analyst") == (
        "优先调用天气工具，并输出城市、天气、温度。"
    )
    assert provider.load_skill_instruction("missing") is None


def test_filesystem_skill_provider_uses_cache_until_refresh(tmp_path) -> None:
    first_skill_dir = tmp_path / "first"
    first_skill_dir.mkdir()
    (first_skill_dir / "SKILL.md").write_text(
        "---\n" "name: first_skill\n" "description: 第一个技能\n" "---\n\n" "第一个技能正文。\n",
        encoding="utf-8",
    )

    provider = FileSystemSkillProvider(str(tmp_path))
    assert [skill.name for skill in provider.list_skills()] == ["first_skill"]

    second_skill_dir = tmp_path / "second"
    second_skill_dir.mkdir()
    (second_skill_dir / "SKILL.md").write_text(
        "---\n" "name: second_skill\n" "description: 第二个技能\n" "---\n\n" "第二个技能正文。\n",
        encoding="utf-8",
    )

    assert [skill.name for skill in provider.list_skills()] == ["first_skill"]

    provider.refresh()

    assert [skill.name for skill in provider.list_skills()] == ["first_skill", "second_skill"]
