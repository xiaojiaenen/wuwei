"""tests for wuwei.fs — 虚拟文件系统（InMemoryFS / LocalFS / VFS 工具与插件）"""

import json
from pathlib import Path

import pytest

from wuwei.fs import (
    FSFile,
    InMemoryFS,
    LocalFS,
    make_vfs_plugin,
)
from wuwei.llm.types import FunctionCall, ToolCall
from wuwei.plugin import PluginContext
from wuwei.tools import ToolExecutor, ToolRegistry

# ----------------------------------------------------------------------
# 测试辅助
# ----------------------------------------------------------------------


def _make_vfs_registry(backend) -> ToolRegistry:
    """参考 tests/test_builtin_tools.py 的写法：注册 VFS 插件工具到 ToolRegistry。"""
    registry = ToolRegistry()
    ctx = PluginContext(tool_registry=registry)
    make_vfs_plugin(backend, name="vfs").setup(ctx)
    return registry


# ----------------------------------------------------------------------
# InMemoryFS
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_read_write_delete_roundtrip() -> None:
    fs = InMemoryFS()

    await fs.write_file("notes/todo.txt", "hello 世界".encode())
    assert await fs.read_file("notes/todo.txt") == "hello 世界".encode()

    await fs.append_file("notes/todo.txt", b"!")
    assert await fs.read_file("notes/todo.txt") == "hello 世界!".encode()

    await fs.delete_file("notes/todo.txt")
    with pytest.raises(FileNotFoundError):
        await fs.read_file("notes/todo.txt")

    with pytest.raises(FileNotFoundError):
        await fs.delete_file("notes/todo.txt")


@pytest.mark.asyncio
async def test_in_memory_binary_safe() -> None:
    fs = InMemoryFS()
    payload = bytes(range(256)) + b"\x00\x00tail"

    await fs.write_file("blob.bin", payload)
    assert await fs.read_file("blob.bin") == payload

    await fs.append_file("blob.bin", b"\x01")
    assert await fs.read_file("blob.bin") == payload + b"\x01"


@pytest.mark.asyncio
async def test_in_memory_rejects_path_escape() -> None:
    fs = InMemoryFS()

    for bad in ("../outside.txt", "a/../../b.txt", "/abs/path.txt", "C:\\evil.txt"):
        with pytest.raises(ValueError):
            await fs.write_file(bad, b"x")
        with pytest.raises(ValueError):
            await fs.read_file(bad)
        with pytest.raises(ValueError):
            await fs.delete_file(bad)


@pytest.mark.asyncio
async def test_in_memory_ls_recursive() -> None:
    fs = InMemoryFS()
    await fs.write_file("a.txt", b"aaa")
    await fs.write_file("sub/b.txt", b"bbbb")
    await fs.write_file("sub/deep/c.txt", b"c")

    entries = await fs.ls(".")
    assert [e.path for e in entries] == ["a.txt", "sub/b.txt", "sub/deep/c.txt"]
    assert entries[0].size == 3
    assert isinstance(entries[0], FSFile)
    assert entries[0].updated_at > 0

    sub_entries = await fs.ls("sub")
    assert [e.path for e in sub_entries] == ["sub/b.txt", "sub/deep/c.txt"]


@pytest.mark.asyncio
async def test_in_memory_glob() -> None:
    fs = InMemoryFS()
    await fs.write_file("a.txt", b"1")
    await fs.write_file("sub/b.md", b"2")
    await fs.write_file("sub/c.txt", b"3")

    assert await fs.glob("*.txt") == ["a.txt", "sub/c.txt"]
    assert await fs.glob("sub/*") == ["sub/b.md", "sub/c.txt"]


# ----------------------------------------------------------------------
# grep
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fs_factory", [InMemoryFS, LocalFS])
async def test_grep_substring_and_regex_with_line_numbers(tmp_path, fs_factory) -> None:
    fs: InMemoryFS | LocalFS = (
        fs_factory(tmp_path) if fs_factory is LocalFS else fs_factory()
    )
    await fs.write_file("code/main.py", b"import os\n# TODO: fix me\nprint(1)\n")
    await fs.write_file("notes.txt", "待办：买咖啡\nTODO: 写文档\n".encode())

    matches, truncated = await fs.grep("TODO")
    assert truncated is False
    assert [(m.path, m.line) for m in matches] == [
        ("code/main.py", 2),
        ("notes.txt", 2),
    ]
    assert matches[0].content == "# TODO: fix me"

    regex_matches, _ = await fs.grep(r"^\s*import \w+", regex=True)
    assert [(m.path, m.line) for m in regex_matches] == [("code/main.py", 1)]

    scoped, _ = await fs.grep("TODO", path="notes.txt")
    assert [m.path for m in scoped] == ["notes.txt"]

    with pytest.raises(ValueError):
        await fs.grep("([bad", regex=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fs_factory", [InMemoryFS, LocalFS])
async def test_grep_truncates_results(tmp_path, fs_factory) -> None:
    fs: InMemoryFS | LocalFS = (
        fs_factory(tmp_path) if fs_factory is LocalFS else fs_factory()
    )
    await fs.write_file("many.txt", "\n".join(f"line {i} hit" for i in range(1, 11)).encode())

    matches, truncated = await fs.grep("hit", max_results=5)
    assert truncated is True
    assert len(matches) == 5
    assert matches[0].line == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fs_factory", [InMemoryFS, LocalFS])
async def test_read_missing_file_raises(tmp_path, fs_factory) -> None:
    fs: InMemoryFS | LocalFS = (
        fs_factory(tmp_path) if fs_factory is LocalFS else fs_factory()
    )
    with pytest.raises(FileNotFoundError):
        await fs.read_file("no/such/file.txt")
    with pytest.raises(FileNotFoundError):
        await fs.delete_file("no/such/file.txt")


# ----------------------------------------------------------------------
# LocalFS
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_fs_roundtrip_and_disk_layout(tmp_path: Path) -> None:
    fs = LocalFS(tmp_path)

    await fs.write_file("logs/app.log", b"line1\n")
    await fs.append_file("logs/app.log", b"line2\n")
    assert await fs.read_file("logs/app.log") == b"line1\nline2\n"

    on_disk = tmp_path / "logs" / "app.log"
    assert on_disk.read_bytes() == b"line1\nline2\n"

    entries = await fs.ls(".")
    assert [e.path for e in entries] == ["logs/app.log"]
    assert entries[0].size == 12
    assert entries[0].updated_at > 0

    await fs.delete_file("logs/app.log")
    assert not on_disk.exists()
    assert not (tmp_path / "logs").exists() or list((tmp_path / "logs").iterdir()) == []


@pytest.mark.asyncio
async def test_local_fs_rejects_escape_outside_root(tmp_path: Path) -> None:
    fs = LocalFS(tmp_path)

    with pytest.raises(ValueError):
        await fs.write_file("../outside.txt", b"evil")
    with pytest.raises(ValueError):
        await fs.read_file("../../etc/passwd")
    with pytest.raises(ValueError):
        await fs.read_file("/etc/passwd")
    with pytest.raises(ValueError):
        await fs.grep("x", path="..")

    # root 外文件确实没有被创建
    assert not (tmp_path.parent / "outside.txt").exists()


@pytest.mark.asyncio
async def test_local_fs_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    fs = LocalFS(tmp_path / "root")
    (tmp_path / "root" / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        await fs.write_file("link/evil.txt", b"escape")


@pytest.mark.asyncio
async def test_local_fs_write_to_root_rejected(tmp_path: Path) -> None:
    fs = LocalFS(tmp_path)
    with pytest.raises(IsADirectoryError):
        await fs.write_file(".", b"x")
    with pytest.raises((IsADirectoryError, ValueError)):
        await fs.read_file(".")


@pytest.mark.asyncio
async def test_local_fs_glob(tmp_path: Path) -> None:
    fs = LocalFS(tmp_path)
    await fs.write_file("a.txt", b"1")
    await fs.write_file("sub/b.md", b"2")

    assert await fs.glob("**/*.md") == ["sub/b.md"]
    assert "a.txt" in await fs.glob("*.txt")


# ----------------------------------------------------------------------
# VFS 工具（ToolRegistry + ToolExecutor 执行路径）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vfs_tools_registered_with_prefixed_names() -> None:
    registry = _make_vfs_registry(InMemoryFS())

    names = {t.name for t in registry.list_tools()}
    assert names == {
        "vfs_write_file",
        "vfs_read_file",
        "vfs_append_file",
        "vfs_delete_file",
        "vfs_ls",
        "vfs_grep",
    }
    for tool in registry.list_tools():
        assert tool.description.startswith("[vfs]")
    assert registry.get("vfs_write_file").parameters.required == ["path", "content"]
    assert registry.get("vfs_write_file").execution.side_effect is True
    assert registry.get("vfs_read_file").execution.side_effect is False


@pytest.mark.asyncio
async def test_vfs_tools_invoke_via_registry() -> None:
    registry = _make_vfs_registry(InMemoryFS())

    write_result = await registry.get("vfs_write_file").invoke(
        {"path": "docs/readme.md", "content": "第一行\n第二行 TODO\n"}
    )
    assert write_result["ok"] is True
    assert write_result["bytes"] == len("第一行\n第二行 TODO\n".encode())

    read_result = await registry.get("vfs_read_file").invoke({"path": "docs/readme.md"})
    assert read_result["content"] == "第一行\n第二行 TODO\n"
    assert read_result["truncated"] is False

    append_result = await registry.get("vfs_append_file").invoke(
        {"path": "docs/readme.md", "content": "第三行\n"}
    )
    assert append_result["ok"] is True

    grep_result = await registry.get("vfs_grep").invoke({"pattern": "TODO", "regex": False})
    assert grep_result["count"] == 1
    assert grep_result["matches"][0]["line"] == 2

    ls_result = await registry.get("vfs_ls").invoke({"path": "."})
    assert ls_result["count"] == 1
    assert ls_result["files"][0]["path"] == "docs/readme.md"

    delete_result = await registry.get("vfs_delete_file").invoke({"path": "docs/readme.md"})
    assert delete_result["deleted"] is True


@pytest.mark.asyncio
async def test_vfs_read_file_truncates_at_limit() -> None:
    registry = _make_vfs_registry(InMemoryFS())
    big_text = "x" * 30_000
    await registry.get("vfs_write_file").invoke({"path": "big.txt", "content": big_text})

    result = await registry.get("vfs_read_file").invoke(
        {"path": "big.txt", "max_chars": 20_000}
    )
    assert result["truncated"] is True
    assert result["size_chars"] == 30_000
    assert len(result["content"]) == 20_000

    full = await registry.get("vfs_read_file").invoke({"path": "big.txt", "max_chars": 50_000})
    assert full["truncated"] is False
    assert full["size_chars"] == 30_000


@pytest.mark.asyncio
async def test_vfs_write_without_overwrite_conflicts() -> None:
    registry = _make_vfs_registry(InMemoryFS())
    await registry.get("vfs_write_file").invoke({"path": "a.txt", "content": "v1"})

    with pytest.raises(FileExistsError):
        await registry.get("vfs_write_file").invoke(
            {"path": "a.txt", "content": "v2", "overwrite": False}
        )

    result = await registry.get("vfs_write_file").invoke(
        {"path": "a.txt", "content": "v2", "overwrite": True}
    )
    assert result["ok"] is True
    read_result = await registry.get("vfs_read_file").invoke({"path": "a.txt"})
    assert read_result["content"] == "v2"


@pytest.mark.asyncio
async def test_vfs_tools_reject_escape_via_registry() -> None:
    registry = _make_vfs_registry(LocalFS(Path("/tmp/wuwei_fs_test_root").absolute()))

    with pytest.raises(ValueError, match="逃逸|绝对路径"):
        await registry.get("vfs_read_file").invoke({"path": "../outside.txt"})

    with pytest.raises(ValueError):
        await registry.get("vfs_write_file").invoke({"path": "/etc/x", "content": "y"})


@pytest.mark.asyncio
async def test_vfs_tools_via_tool_executor() -> None:
    """端到端：ToolCall → ToolExecutor.execute → Message（错误也被包装成 JSON）。"""
    registry = _make_vfs_registry(InMemoryFS())
    executor = ToolExecutor(registry)

    tool_calls = [
        ToolCall(
            id="call_1",
            function=FunctionCall(
                name="vfs_write_file",
                arguments={"path": "notes/todo.txt", "content": "buy milk\nTODO: nap"},
            ),
        ),
        ToolCall(
            id="call_2",
            function=FunctionCall(name="vfs_grep", arguments={"pattern": "TODO"}),
        ),
        ToolCall(
            id="call_3",
            function=FunctionCall(name="vfs_read_file", arguments={"path": "../evil"}),
        ),
    ]
    messages = await executor.execute(tool_calls)

    assert len(messages) == 3

    write_payload = json.loads(messages[0].content)
    assert write_payload["ok"] is True
    assert write_payload["path"] == "notes/todo.txt"

    grep_payload = json.loads(messages[1].content)
    assert grep_payload["ok"] is True
    assert grep_payload["matches"] == [
        {"path": "notes/todo.txt", "line": 2, "content": "TODO: nap"}
    ]

    error_payload = json.loads(messages[2].content)
    assert error_payload["ok"] is False
    assert error_payload["error"]["type"] == "ValueError"


@pytest.mark.asyncio
async def test_vfs_plugin_registers_all_tools() -> None:
    backend = InMemoryFS()
    plugin = make_vfs_plugin(backend, name="scratch")

    assert plugin.name == "scratch"
    registry = ToolRegistry()
    plugin.setup(PluginContext(tool_registry=registry))
    assert {t.name for t in registry.list_tools()} == {
        "scratch_write_file",
        "scratch_read_file",
        "scratch_append_file",
        "scratch_delete_file",
        "scratch_ls",
        "scratch_grep",
    }

    # 插件工具与后端联通
    await registry.get("scratch_write_file").invoke({"path": "x.txt", "content": "hi"})
    assert await backend.read_file("x.txt") == b"hi"

    # 同名工具重复注册应报错（Plugin.setup 幂等性由调用方保证）
    with pytest.raises(ValueError, match="已经注册"):
        plugin.setup(PluginContext(tool_registry=registry))
