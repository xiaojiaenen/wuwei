import json
import subprocess

import pytest

from wuwei.tools import ToolRegistry


@pytest.mark.asyncio
async def test_file_tools_read_write_replace_delete(tmp_path) -> None:
    registry = ToolRegistry.from_builtin(["file"])

    write_result = await registry.get("write_text_file").invoke(
        {
            "path": "notes/todo.txt",
            "content": "hello world",
            "workspace": str(tmp_path),
        }
    )
    assert write_result["ok"] is True

    read_result = await registry.get("read_text_file").invoke(
        {"path": "notes/todo.txt", "workspace": str(tmp_path)}
    )
    assert read_result["content"] == "hello world"

    replace_result = await registry.get("replace_text_in_file").invoke(
        {
            "path": "notes/todo.txt",
            "old_text": "world",
            "new_text": "wuwei",
            "workspace": str(tmp_path),
        }
    )
    assert replace_result["replacements"] == 1
    assert (tmp_path / "notes" / "todo.txt").read_text(encoding="utf-8") == "hello wuwei"

    delete_result = await registry.get("delete_file").invoke(
        {"path": "notes/todo.txt", "workspace": str(tmp_path)}
    )
    assert delete_result["deleted"] is True
    assert not (tmp_path / "notes" / "todo.txt").exists()


@pytest.mark.asyncio
async def test_file_tools_reject_path_outside_workspace(tmp_path) -> None:
    registry = ToolRegistry.from_builtin(["file"])

    with pytest.raises(ValueError, match="workspace"):
        await registry.get("read_text_file").invoke(
            {"path": "../outside.txt", "workspace": str(tmp_path)}
        )


@pytest.mark.asyncio
async def test_python_tool_runs_workspace_script(tmp_path) -> None:
    registry = ToolRegistry.from_builtin(["python"])
    script = tmp_path / "echo.py"
    script.write_text(
        "import json\n"
        "import sys\n"
        "print(json.dumps({'args': sys.argv[1:]}, ensure_ascii=False))\n",
        encoding="utf-8",
    )

    result = await registry.get("run_python_script").invoke(
        {
            "script_path": "echo.py",
            "args_json": json.dumps(["北京", "晴"]),
            "workspace": str(tmp_path),
        }
    )

    assert result["ok"] is True
    assert json.loads(result["stdout"]) == {"args": ["北京", "晴"]}


@pytest.mark.asyncio
async def test_calculate_tool() -> None:
    registry = ToolRegistry.from_builtin(["calc"])

    result = await registry.get("calculate").invoke({"expression": "sqrt(16) + sin(pi / 2)"})
    assert result["ok"] is True
    assert result["result"] == 5

    with pytest.raises(ValueError, match="不允许的名称"):
        await registry.get("calculate").invoke({"expression": "__import__('os')"})


@pytest.mark.asyncio
async def test_git_tools_status_and_diff(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")

    registry = ToolRegistry.from_builtin(["git"])

    status = await registry.get("git_status").invoke({"workspace": str(tmp_path)})
    assert status["ok"] is True
    assert "README.md" in status["stdout"]

    diff = await registry.get("git_diff").invoke(
        {"path": "README.md", "workspace": str(tmp_path)}
    )
    assert diff["ok"] is True

    with pytest.raises(ValueError, match="workspace"):
        await registry.get("git_diff").invoke(
            {"path": "../outside.txt", "workspace": str(tmp_path)}
        )


@pytest.mark.asyncio
async def test_npm_tools_list_scripts_and_validate_script(tmp_path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "test": "echo ok",
                    "build": "echo build",
                }
            }
        ),
        encoding="utf-8",
    )

    registry = ToolRegistry.from_builtin(["npm"])

    scripts = await registry.get("npm_list_scripts").invoke({"workspace": str(tmp_path)})
    assert scripts["ok"] is True
    assert scripts["scripts"]["test"] == "echo ok"

    with pytest.raises(ValueError, match="不存在 script"):
        await registry.get("npm_run_script").invoke(
            {"script_name": "missing", "workspace": str(tmp_path)}
        )
