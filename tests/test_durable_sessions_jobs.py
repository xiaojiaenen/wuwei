import asyncio

import pytest

from wuwei.agent import AgentSession


@pytest.mark.asyncio
async def test_session_store_restore_and_owner_lock(tmp_path):
    from wuwei.persistence import SQLiteSessionStore, SessionBusy

    store = SQLiteSessionStore(tmp_path / "sessions.db", namespace="tenant-a")
    session = AgentSession("s")
    async with store.checkout(session):
        session.context.add_user_message("remember")
        with pytest.raises(SessionBusy):
            async with SQLiteSessionStore(tmp_path / "sessions.db", namespace="tenant-a").checkout(
                AgentSession("s")
            ):
                pass
    restored = await store.load("s")
    assert restored.context.get_messages()[-1].content == "remember"
    assert (
        await SQLiteSessionStore(tmp_path / "sessions.db", namespace="tenant-b").load("s") is None
    )
    await store.delete("s")
    assert await store.load("s") is None


@pytest.mark.asyncio
async def test_agent_restarts_with_session_store(tmp_path):
    from tests.test_runtime_hardening import FakeLLM, response
    from wuwei import Agent
    from wuwei.persistence import SQLiteSessionStore

    path = tmp_path / "sessions.db"
    first = Agent(
        llm=FakeLLM([response("one")]), load_builtins=False, session_store=SQLiteSessionStore(path)
    )
    session = first.create_session(session_id="saved")
    await first.run("remember", session=session)
    await first.release_session("saved")
    second = Agent(
        llm=FakeLLM([response("two")]), load_builtins=False, session_store=SQLiteSessionStore(path)
    )
    restored = await second.load_session("saved")
    await second.run("next", session=restored)
    assert [m.content for m in second.llm.calls[0] if m.role == "user"] == ["remember", "next"]
    assert second.create_session(session_id="saved") is restored


@pytest.mark.asyncio
async def test_session_stream_terminal_after_persist(tmp_path):
    from tests.test_runtime_hardening import make_agent, response
    from wuwei.persistence import SQLiteSessionStore

    agent = make_agent([response("done")])
    agent.session_store = SQLiteSessionStore(tmp_path / "s.db")
    session = agent.create_session(session_id="stream")
    async for event in agent.stream_events("hello", session=session):
        if event.type == "run_end":
            saved = await agent.session_store.load("stream")
            assert saved.context.get_messages()[-1].content == "done"


@pytest.mark.asyncio
async def test_session_scope_persists_block_after_unknown(tmp_path):
    from tests.test_runtime_hardening import make_agent, response
    from wuwei.persistence import SQLiteSessionStore

    agent = make_agent([response("done")])
    agent.session_store = SQLiteSessionStore(tmp_path / "s.db")
    session = agent.create_session("s")
    session.metadata["_blocked_run"] = {"reason": "unknown"}
    with pytest.raises(ValueError):
        await agent.run("hello", session=session)
    assert (await agent.session_store.load("s")).metadata["_blocked_run"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_atomic_checkpoint_replacement(tmp_path, backend):
    from wuwei.graph.checkpoint import MemoryCheckpointer, SQLiteCheckpointer
    from wuwei.graph.state import State

    cp = (
        MemoryCheckpointer() if backend == "memory" else SQLiteCheckpointer(str(tmp_path / "cp.db"))
    )
    state = State(metadata={"run_id": "run"})
    cid = await cp.save(state, metadata={"run_id": "run"})
    new = await cp.replace_current(cid, state, metadata={"run_id": "run"})
    assert await cp.claim(cid) is False
    with pytest.raises(ValueError):
        await cp.replace_current(cid, state, metadata={"run_id": "run"})
    assert await cp.claim(new) is True


@pytest.mark.asyncio
async def test_job_restart_claim_idempotency_and_approval(tmp_path):
    from wuwei.persistence import SQLiteJobStore, JobService
    from tests.test_runtime_hardening import make_agent, response, call, tool
    from wuwei.graph.checkpoint import SQLiteCheckpointer
    from wuwei.middleware import HitlMiddleware

    path = tmp_path / "jobs.db"
    store = SQLiteJobStore(path)
    job = await store.submit("write", idempotency_key="key")
    assert (await store.submit("write", idempotency_key="key")).id == job.id
    with pytest.raises(ValueError):
        await store.submit("different", idempotency_key="key")
    writes = []

    async def write():
        writes.append(1)
        return "ok"

    def before():
        return make_agent(
            [response(calls=[call("write")])],
            [tool("write", write, True)],
            [HitlMiddleware(mode="interrupt", approval_policy="declared")],
            SQLiteCheckpointer(str(tmp_path / "cp.db")),
        )

    first = await JobService(store, before).execute(job.id)
    assert first.status == "waiting_approval" and writes == []
    store = SQLiteJobStore(path)  # New process-equivalent instance.
    await store.signal(job.id, approved=True, operator="reviewer", signal_id="decision-1")
    await store.signal(job.id, approved=True, operator="reviewer", signal_id="decision-1")
    with pytest.raises(ValueError):
        await store.signal(job.id, approved=False, operator="reviewer", signal_id="decision-1")

    def after():
        return make_agent(
            [response("done")],
            [tool("write", write, True)],
            [HitlMiddleware(mode="interrupt", approval_policy="declared")],
            SQLiteCheckpointer(str(tmp_path / "cp.db")),
        )

    result = await JobService(store, after).execute(job.id)
    assert result.status == "completed" and writes == [1]
    assert (await JobService(store, after).execute(job.id)).status == "completed"
    assert writes == [1]


@pytest.mark.asyncio
async def test_expired_job_not_replayed_and_stale_owner_fenced(tmp_path):
    from wuwei.persistence import SQLiteJobStore

    store = SQLiteJobStore(tmp_path / "j.db", lease_seconds=0.02)
    job = await store.submit("external write")
    claimed = await store.claim(job.id)
    assert await store.claim(job.id) is None
    await asyncio.sleep(0.04)
    assert (await store.get(job.id)).status == "unknown"
    assert await store.claim(job.id) is None
    with pytest.raises(RuntimeError):
        await store.finish(job.id, claimed[1], {"status": "completed"})


@pytest.mark.asyncio
async def test_job_cancel_before_claim_and_namespace(tmp_path):
    from wuwei.persistence import SQLiteJobStore

    store = SQLiteJobStore(tmp_path / "j.db", namespace="a")
    job = await store.submit("do something")
    assert await SQLiteJobStore(tmp_path / "j.db", namespace="b").get(job.id) is None
    assert (await store.cancel(job.id)).status == "cancelled"
    assert await store.claim(job.id) is None


@pytest.mark.asyncio
async def test_dbos_client_delivery_contract_and_retry(tmp_path):
    from wuwei.persistence import DBOSJobBackend, SQLiteJobStore, JobService
    from tests.test_runtime_hardening import make_agent, response

    seen = []

    class Client:
        async def enqueue_async(self, options, job_id):
            seen.append((options, job_id))

    service = JobService(SQLiteJobStore(tmp_path / "j.db"), lambda: make_agent([response("ok")]))
    backend = DBOSJobBackend(service, Client())
    first = await backend.submit("hello", idempotency_key="stable")
    second = await backend.submit("hello", idempotency_key="stable")
    assert first.id == second.id
    assert seen[0] == seen[1]  # Retry the same workflow id, not a second delivery identity.
    assert (await service.execute(first.id)).status == "completed"
    assert (await backend.get(first.id)).result["content"] == "ok"


@pytest.mark.asyncio
async def test_abandoned_session_lease_does_not_enable_tool_replay(tmp_path):
    import time
    from wuwei.persistence import SQLiteSessionStore

    store = SQLiteSessionStore(tmp_path / "s.db", lease_seconds=0.02, ttl_seconds=0.02)
    # Equivalent persisted state to a process killed after claiming a session.
    original = AgentSession("s")
    _, revision = store._acquire(original, "dead-worker")
    await asyncio.sleep(0.04)
    recovered = AgentSession("s")
    async with store.checkout(recovered):
        assert recovered.metadata["_blocked_run"]["reason"] == "session_lease_expired"
    assert (await store.load("s")).metadata["_blocked_run"]
    with pytest.raises(Exception, match="stale"):
        store._commit(original, "dead-worker", revision)


@pytest.mark.asyncio
async def test_active_session_heartbeat_and_expiration(tmp_path):
    from wuwei.persistence import SQLiteSessionStore

    store = SQLiteSessionStore(tmp_path / "s.db", lease_seconds=0.06, ttl_seconds=0.03)
    session = AgentSession("s")
    async with store.checkout(session):
        await asyncio.sleep(0.13)
        session.context.add_user_message("still owned")
    assert (await store.load("s")).context.get_messages()[-1].content == "still owned"
    await asyncio.sleep(0.05)
    assert await store.load("s") is None


@pytest.mark.asyncio
async def test_durable_blocked_history_cannot_be_deleted_after_restart(tmp_path):
    from wuwei.persistence import SQLiteSessionStore

    path = tmp_path / "s.db"
    store = SQLiteSessionStore(path)
    session = AgentSession("blocked")
    async with store.checkout(session):
        session.metadata["_blocked_run"] = {"reason": "external action"}
    with pytest.raises(ValueError, match="Reconcile"):
        await SQLiteSessionStore(path).delete("blocked")


@pytest.mark.asyncio
async def test_job_cleanup_failure_cannot_erase_known_result(tmp_path):
    from types import SimpleNamespace
    from wuwei.persistence import SQLiteJobStore, JobService
    from wuwei.llm import AgentRunResult

    class Agent:
        def create_or_get_session(self, **kw):
            return SimpleNamespace()

        async def run(self, *args, **kw):
            return AgentRunResult(content="done", status="completed")

        async def aclose(self):
            raise OSError("resource close failed")

    store = SQLiteJobStore(tmp_path / "j.db")
    record = await store.submit("go")
    result = await JobService(store, Agent).execute(record.id)
    assert result.status == "completed"
    assert (await store.get(record.id)).result["content"] == "done"


@pytest.mark.asyncio
async def test_agent_close_rejects_external_active_session():
    from tests.test_runtime_hardening import make_agent
    from wuwei.llm import LLMResponse

    entered, finish = asyncio.Event(), asyncio.Event()

    class LLM:
        async def generate(self, *args, **kwargs):
            entered.set()
            await finish.wait()
            return LLMResponse(content="done", finish_reason="stop")

    from wuwei import Agent

    agent = Agent(llm=LLM(), load_builtins=False)
    task = asyncio.create_task(agent.run("go", session=AgentSession("external")))
    await entered.wait()
    with pytest.raises(RuntimeError, match="active"):
        await agent.aclose()
    finish.set()
    await task
    await agent.aclose()


@pytest.mark.asyncio
async def test_agent_media_input_roundtrip_and_ownership(tmp_path):
    from wuwei import HumanMessage, FilePart, ArtifactStore, SQLiteSessionStore
    from tests.test_runtime_hardening import make_agent, response

    blobs = ArtifactStore(tmp_path / "artifacts")
    ref = blobs.put(b"%PDF-test", owner="tenant", media_type="application/pdf")
    message = HumanMessage("read", content_parts=[FilePart(artifact=ref, filename="r.pdf")])
    agent = make_agent([response("done")])
    agent.session_store = SQLiteSessionStore(tmp_path / "sessions.db", namespace="tenant")
    session = agent.create_session("media")
    await agent.run(message, session=session)
    saved = await agent.session_store.load("media")
    part = next(m for m in saved.context.get_messages() if m.role == "user").content_parts[0]
    assert part.artifact == ref and blobs.read(part.artifact, owner="tenant") == b"%PDF-test"
    assert agent.llm.calls[0][-1].content_parts[0].artifact == ref
    assert "%PDF-test" not in str(saved.to_dict())


@pytest.mark.asyncio
async def test_job_unknown_reconciliation_does_not_reexecute(tmp_path):
    from wuwei import ToolRegistry, ToolOutcome, SQLiteJobStore, JobService, Agent
    from wuwei.graph.checkpoint import SQLiteCheckpointer
    from tests.test_runtime_hardening import FakeLLM, response, call

    store = SQLiteJobStore(tmp_path / "j.db")
    cp = SQLiteCheckpointer(str(tmp_path / "cp.db"))
    writes = []
    registry = ToolRegistry()

    def write():
        writes.append(1)
        return ToolOutcome(status="unknown", content="receipt not available")

    registry.register_callable(write)
    job = await store.submit("write")
    factory = lambda: Agent(
        llm=FakeLLM([response(calls=[call("write", "write1")])]),
        tools=registry,
        load_builtins=False,
        checkpointer=cp,
    )
    assert (await JobService(store, factory).execute(job.id)).status == "unknown"
    records = await cp.list_checkpoints(limit=1)
    confirmed = JobService(
        store,
        lambda: Agent(
            llm=FakeLLM([response("done")]), tools=registry, load_builtins=False, checkpointer=cp
        ),
    )
    result = await confirmed.reconcile(
        job.id,
        checkpoint_id=records[0]["id"],
        tool_call_id="write1",
        outcome=ToolOutcome(content="confirmed"),
        operator="ops",
        evidence="receipt-123",
    )
    assert result.status == "completed" and writes == [1]


@pytest.mark.asyncio
async def test_unresolved_memory_only_session_cannot_be_silently_evicted():
    from tests.test_runtime_hardening import make_agent

    agent = make_agent([])
    session = agent.create_session("unknown")
    session.metadata["_blocked_run"] = "unfinished"
    with pytest.raises(ValueError, match="only history"):
        await agent.release_session("unknown")
    assert agent.create_or_get_session("unknown") is session
