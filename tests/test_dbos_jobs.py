"""Optional real DBOS+SQLite smoke, isolated from pytest's global tracing state."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("dbos") is None, reason="install wuwei[dbos]")
def test_dbos_sqlite_queue_executes_real_agent(tmp_path):
    source = r"""
import asyncio
import sys
from pathlib import Path
from dbos import DBOS, DBOSClient, Queue
from wuwei.persistence import SQLiteJobStore, JobService, DBOSJobBackend, build_dbos_workflow
from tests.test_runtime_hardening import make_agent, response
root = Path(sys.argv[1])
url = 'sqlite:///' + str(root / 'dbos.sqlite')
DBOS(config={'name': 'wuwei-offline-test', 'system_database_url': url, 'enable_otlp': False})
store = SQLiteJobStore(root / 'truth.sqlite')
service = JobService(store, lambda: make_agent([response('from real DBOS')]))
workflow = build_dbos_workflow(service)
DBOS.launch()
queue = DBOS.register_queue('wuwei')
client = DBOSClient(system_database_url=url)
async def main():
    backend = DBOSJobBackend(service, client)
    job = await backend.submit('hello', idempotency_key='request')
    for _ in range(200):
        record = await backend.get(job.id)
        if record.status == 'completed':
            assert record.result['content'] == 'from real DBOS'
            again = await backend.submit('hello', idempotency_key='request')
            assert again.id == job.id
            print('DBOS queued Agent completed without duplicate execution')
            return
        await asyncio.sleep(.05)
    raise AssertionError(record)
try:
    asyncio.run(main())
finally:
    client.destroy()
    DBOS.destroy()
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
