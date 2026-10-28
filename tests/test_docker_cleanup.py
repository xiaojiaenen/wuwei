import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from wuwei.sandbox.docker import DockerDriver


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_owned_container_cleanup(monkeypatch, cancel):
    calls = []

    async def spawn(*argv, **kw):
        calls.append(argv)
        if argv[1] == "run":
            path = Path(argv[argv.index("--cidfile") + 1])
            assert not path.exists()
            path.write_text("a" * 64)
        return SimpleNamespace(returncode=0)

    async def communicate(proc, data=None, timeout=None):
        if len(calls) == 1:
            if cancel:
                raise asyncio.CancelledError
            raise TimeoutError
        return b"", b""

    monkeypatch.setattr("wuwei.sandbox.docker.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("wuwei.sandbox.docker.communicate", communicate)
    driver = DockerDriver()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await driver._run(["docker", "run", "--rm", "image"], stdin_data=b"", timeout=1)
    else:
        result = await driver._run(["docker", "run", "--rm", "image"], stdin_data=b"", timeout=1)
        assert "cleanup confirmed" in result.error
    assert calls[1] == ("docker", "rm", "-f", "a" * 64)
    assert not Path(calls[0][calls[0].index("--cidfile") + 1]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cid", ["", "unrelated-name", "../other"])
async def test_unknown_identity_never_removed(monkeypatch, cid):
    calls = []

    async def spawn(*argv, **kw):
        calls.append(argv)
        Path(argv[argv.index("--cidfile") + 1]).write_text(cid)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("wuwei.sandbox.docker.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("wuwei.sandbox.docker.communicate", AsyncMock(side_effect=TimeoutError))
    result = await DockerDriver()._run(
        ["docker", "run", "--rm", "image"], stdin_data=b"", timeout=1
    )
    assert len(calls) == 1
    assert "unconfirmed" in result.error


@pytest.mark.asyncio
async def test_daemon_failure_not_claimed_clean(monkeypatch):
    calls = []

    async def spawn(*argv, **kw):
        calls.append(argv)
        if argv[1] == "run":
            Path(argv[argv.index("--cidfile") + 1]).write_text("b" * 64)
            return SimpleNamespace(returncode=0)
        raise OSError("daemon unavailable")

    monkeypatch.setattr("wuwei.sandbox.docker.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("wuwei.sandbox.docker.communicate", AsyncMock(side_effect=TimeoutError))
    result = await DockerDriver()._run(
        ["docker", "run", "--rm", "image"], stdin_data=b"", timeout=1
    )
    assert "unconfirmed: daemon unavailable" in result.error
    assert calls[1] == ("docker", "rm", "-f", "b" * 64)
