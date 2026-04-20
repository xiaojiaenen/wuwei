from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from wuwei.skill.skill import SkillManager
from wuwei.tools import ToolRegistry

SKILL_SCRIPT_TIMEOUT_SECONDS = 10
SKILL_SCRIPT_OUTPUT_LIMIT = 4000


def _normalize_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _truncate_output(value: str | bytes | None, *, limit: int | None = None) -> str:
    if limit is None:
        limit = SKILL_SCRIPT_OUTPUT_LIMIT
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [truncated {omitted} chars]"


def register_skill_tools(registry: ToolRegistry, skill_manager: SkillManager) -> None:
    loaded_skill_tokens: dict[str, str] = {}

    @registry.tool(
        name="list_skills",
        description="列出当前可用技能的摘要。先查看技能列表，再自行判断是否需要某个 skill。",
    )
    def list_skills() -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for skill in skill_manager.list_skills():
            scripts: list[str] = []
            if skill.path is not None:
                scripts_dir = Path(skill.path) / "scripts"
                if scripts_dir.is_dir():
                    scripts = sorted(
                        str(path.relative_to(Path(skill.path)))
                        for path in scripts_dir.rglob("*.py")
                    )

            items.append(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "has_python_scripts": bool(scripts),
                    "python_scripts": scripts,
                }
            )
        return items

    @registry.tool(
        name="load_skill",
        description="根据技能名称加载技能正文。选中 skill 后再调用这个工具。",
    )
    def load_skill(skill_name: str) -> dict[str, str]:
        skill = skill_manager.get_skill(skill_name)
        instruction = skill_manager.load_skill_instruction(skill_name)
        if not instruction:
            raise ValueError(f"Skill '{skill_name}' 不存在或没有正文")

        load_token = uuid4().hex
        loaded_skill_tokens[load_token] = skill.name
        return {
            "name": skill.name,
            "description": skill.description,
            "instruction": instruction,
            "load_token": load_token,
        }

    @registry.tool(
        name="run_skill_python_script",
        description=(
            "执行 skill 目录下 scripts/ 中的 Python 脚本。"
            "只有在已经加载该 skill 且正文明确要求执行脚本时才使用。"
            "必须传入 load_skill 返回的 load_token。"
        ),
    )
    def run_skill_python_script(
        skill_name: str,
        script_path: str,
        load_token: str,
        args_json: str = "[]",
    ) -> dict[str, object]:
        skill = skill_manager.get_skill(skill_name)
        if not skill.path:
            raise ValueError(f"Skill '{skill_name}' 没有关联目录")
        if loaded_skill_tokens.get(load_token) != skill.name:
            raise ValueError(
                "运行 skill 脚本前必须先调用 load_skill，并使用 load_skill 返回的 load_token。"
            )

        root_dir = Path(skill.path).resolve()
        target_path = (root_dir / script_path).resolve()

        if root_dir not in target_path.parents:
            raise ValueError("脚本路径必须位于 skill 目录内")
        if not target_path.is_file():
            raise ValueError(f"脚本不存在: {script_path}")
        if target_path.suffix != ".py":
            raise ValueError("当前仅允许执行 Python 脚本")

        relative_path = target_path.relative_to(root_dir)
        if not relative_path.parts or relative_path.parts[0] != "scripts":
            raise ValueError("当前仅允许执行 skill 目录下 scripts/ 中的脚本")

        try:
            args = json.loads(args_json)
        except json.JSONDecodeError as exc:
            raise ValueError("args_json 必须是合法的 JSON 数组") from exc

        if not isinstance(args, list):
            raise ValueError("args_json 必须是 JSON 数组")

        command = [sys.executable, str(target_path), *[str(item) for item in args]]
        try:
            result = subprocess.run(
                command,
                cwd=str(root_dir),
                capture_output=True,
                text=True,
                timeout=SKILL_SCRIPT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Skill 脚本执行超时（>{SKILL_SCRIPT_TIMEOUT_SECONDS} 秒）\n"
                f"stdout:\n{_truncate_output(exc.stdout)}\n"
                f"stderr:\n{_truncate_output(exc.stderr)}"
            ) from exc

        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": _truncate_output(result.stdout),
            "stderr": _truncate_output(result.stderr),
        }
