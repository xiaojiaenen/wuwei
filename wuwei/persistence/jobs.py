"""Durable job truth and a single-job worker, not a distributed scheduler.

Interrupted jobs accept explicit approval signals. Expired running jobs become
unknown: a replacement worker never blindly reruns an external operation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "completed",
    "incomplete",
    "unknown",
    "error",
    "cancelled",
]


class JobRecord(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    schema_version: Literal[1] = 1
    id: str
    input: str
    session_id: str
    status: JobStatus = "queued"
    result: dict[str, Any] | None = None
    continuation: dict[str, Any] | None = None
    cancel_requested: bool = False
    revision: int = 0
    error: str | None = None
    audit: list[dict[str, Any]] = Field(default_factory=list)


class JobBackend(Protocol):
    async def submit(
        self, user_input: str, *, idempotency_key: str | None = None, session_id: str | None = None
    ) -> JobRecord: ...
    async def get(self, job_id: str) -> JobRecord | None: ...
    async def cancel(self, job_id: str) -> JobRecord: ...
    async def signal(
        self, job_id: str, *, approved: bool, operator: str, signal_id: str, reason: str = ""
    ) -> JobRecord: ...


class SQLiteJobStore:
    def __init__(self, path=".wuwei/jobs.db", *, namespace="default", lease_seconds=60):
        if not namespace or lease_seconds <= 0:
            raise ValueError("namespace/lease must be nonempty/positive")
        self.path, self.namespace, self.lease_seconds = str(path), namespace, lease_seconds
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                namespace TEXT NOT NULL, id TEXT NOT NULL, key TEXT NOT NULL,
                payload TEXT NOT NULL, token TEXT, lease_until REAL,
                PRIMARY KEY(namespace,id), UNIQUE(namespace,key))""")
            db.execute("""CREATE TABLE IF NOT EXISTS job_signals (
                namespace TEXT NOT NULL, job_id TEXT NOT NULL, signal_id TEXT NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(namespace,job_id,signal_id))""")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def _read(self, db, job_id):
        row = db.execute(
            "SELECT payload,token,lease_until FROM jobs WHERE namespace=? AND id=?",
            (self.namespace, job_id),
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        record = JobRecord.model_validate_json(row[0])
        if record.status == "running" and (row[2] is None or row[2] <= time.time()):
            record.status, record.error = "unknown", "Worker lease expired; reconcile before retry"
            record.revision += 1
            record.audit.append({"action": "lease_expired", "time": time.time()})
            self._write(db, record)
            return record, None
        return record, row[1]

    def _write(self, db, record, token=None, lease_until=None):
        db.execute(
            "UPDATE jobs SET payload=?,token=?,lease_until=? WHERE namespace=? AND id=?",
            (record.model_dump_json(), token, lease_until, self.namespace, record.id),
        )

    async def submit(self, user_input, *, idempotency_key=None, session_id=None):
        if not isinstance(user_input, str) or not user_input:
            raise ValueError("Job input must be nonempty text")
        key = idempotency_key or uuid4().hex
        if not isinstance(key, str):
            raise ValueError("Idempotency key must be a string")

        def create():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT payload FROM jobs WHERE namespace=? AND key=?", (self.namespace, key)
                ).fetchone()
                if row:
                    record = JobRecord.model_validate_json(row[0])
                    if record.input != user_input or (
                        session_id and record.session_id != session_id
                    ):
                        raise ValueError("Idempotency key was used with different input/session")
                    return record
                record = JobRecord(
                    id=uuid4().hex, input=user_input, session_id=session_id or uuid4().hex
                )
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,NULL,NULL)",
                    (self.namespace, record.id, key, record.model_dump_json()),
                )
                return record

        return await asyncio.to_thread(create)

    async def get(self, job_id):
        def read():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                try:
                    return self._read(db, job_id)[0]
                except KeyError:
                    return None

        return await asyncio.to_thread(read)

    async def claim(self, job_id, *, reconciliation=False):
        def claim():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                record, _ = self._read(db, job_id)
                expected = "unknown" if reconciliation else "queued"
                if record.status != expected:
                    return None
                record.status, token = "running", uuid4().hex
                record.revision += 1
                self._write(db, record, token, time.time() + self.lease_seconds)
                return record, token

        return await asyncio.to_thread(claim)

    async def heartbeat(self, job_id, token):
        def renew():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                record, current_token = self._read(db, job_id)
                if current_token != token or record.status != "running":
                    raise RuntimeError("Job ownership lost")
                self._write(db, record, token, time.time() + self.lease_seconds)
                return record.cancel_requested

        return await asyncio.to_thread(renew)

    async def finish(self, job_id, token, result):
        def finish():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                record, current_token = self._read(db, job_id)
                if current_token != token or record.status != "running":
                    raise RuntimeError("Stale job owner cannot commit")
                status = result.get("status")
                record.status = "waiting_approval" if status == "interrupted" else status
                record.result = result
                record.continuation = None
                record.revision += 1
                self._write(db, record)
                return record

        return await asyncio.to_thread(finish)

    async def annotate(self, job_id, entry):
        def append():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                record, token = self._read(db, job_id)
                if token is not None:
                    raise ValueError("Cannot annotate another executing owner")
                record.audit.append(entry)
                self._write(db, record)
                return record

        return await asyncio.to_thread(append)

    async def cancel(self, job_id):
        def cancel():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                record, token = self._read(db, job_id)
                if record.status in ("completed", "incomplete", "error", "cancelled", "unknown"):
                    return record
                record.cancel_requested = True
                if record.status in ("queued", "waiting_approval"):
                    record.status = "cancelled"
                record.revision += 1
                record.audit.append({"action": "cancel_requested", "time": time.time()})
                self._write(db, record, token, time.time() + self.lease_seconds if token else None)
                return record

        return await asyncio.to_thread(cancel)

    async def signal(self, job_id, *, approved, operator, signal_id, reason=""):
        if not operator.strip() or not signal_id.strip() or not isinstance(approved, bool):
            raise ValueError("Approval signal requires operator, signal_id and a boolean decision")
        data = json.dumps(
            {"approved": approved, "operator": operator, "reason": reason}, sort_keys=True
        )

        def signal():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                record, _ = self._read(db, job_id)
                old = db.execute(
                    "SELECT payload FROM job_signals WHERE namespace=? AND job_id=? AND signal_id=?",
                    (self.namespace, job_id, signal_id),
                ).fetchone()
                if old:
                    if old[0] != data:
                        raise ValueError("Signal id reused with a different decision")
                    return record
                if record.status != "waiting_approval" or not (record.result or {}).get(
                    "interrupt"
                ):
                    raise ValueError("Job is not awaiting an approval signal")
                record.continuation = {
                    "interrupt": record.result["interrupt"],
                    "approved": approved,
                    "reason": reason,
                }
                record.status = "queued"
                record.revision += 1
                record.audit.append(
                    {
                        "action": "approval",
                        "operator": operator,
                        "signal_id": signal_id,
                        "approved": approved,
                        "reason": reason,
                        "time": time.time(),
                    }
                )
                db.execute(
                    "INSERT INTO job_signals VALUES (?,?,?,?)",
                    (self.namespace, job_id, signal_id, data),
                )
                self._write(db, record)
                return record

        return await asyncio.to_thread(signal)


class JobService:
    """Execute one claimed job with the real Agent lifecycle and durable truth.

    agent_factory must return a fresh Agent for each job; application resources
    and runtime dependencies are never serialized in a job payload.
    """

    def __init__(self, store: SQLiteJobStore, agent_factory):
        self.store, self.agent_factory = store, agent_factory

    async def execute(self, job_id):
        return await self._execute(job_id)

    async def reconcile(self, job_id, *, checkpoint_id, tool_call_id, outcome, operator, evidence):
        from wuwei.tools import ToolOutcome

        if (
            not isinstance(outcome, ToolOutcome)
            or outcome.status == "unknown"
            or not operator
            or not evidence
        ):
            raise ValueError("A known outcome, operator and evidence are required")
        return await self._execute(
            job_id,
            reconciliation={
                "checkpoint_id": checkpoint_id,
                "tool_call_id": tool_call_id,
                "outcome": outcome,
                "operator": operator,
                "evidence": evidence,
            },
        )

    async def _execute(self, job_id, reconciliation=None):
        claim = await self.store.claim(job_id, reconciliation=reconciliation is not None)
        if claim is None:
            record = await self.store.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record
        record, token = claim
        parent, owner_error = asyncio.current_task(), []

        async def heartbeat():
            try:
                while True:
                    if await self.store.heartbeat(job_id, token):
                        parent.cancel()
                        return
                    await asyncio.sleep(min(self.store.lease_seconds / 3, 1))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                owner_error.append(exc)
                parent.cancel()

        task = asyncio.create_task(heartbeat())
        agent = None
        committed = None
        try:
            try:
                agent = self.agent_factory()
                if reconciliation is not None:
                    result = await agent.reconcile(
                        **reconciliation,
                        session=agent.create_or_get_session(session_id=record.session_id),
                    )
                elif record.continuation:
                    decision = record.continuation
                    result = await agent.resume(
                        decision["interrupt"],
                        approved=decision["approved"],
                        reason=decision["reason"],
                    )
                else:
                    session = agent.create_or_get_session(session_id=record.session_id)
                    result = await agent.run(record.input, session=session)
                payload = result if isinstance(result, dict) else result.model_dump(mode="json")
            except asyncio.CancelledError:
                # The runner has closed first, recording whether a side effect is unknown.
                blocked = agent is not None and any(
                    s.metadata.get("_blocked_run") for s in agent._sessions.values()
                )
                payload = {
                    "status": "unknown" if blocked or owner_error else "cancelled",
                    "content": "",
                    "termination_reason": "worker_cancelled",
                }
                if owner_error:
                    raise RuntimeError(
                        "Job ownership lost; no stale result committed"
                    ) from owner_error[0]
                await asyncio.shield(self.store.finish(job_id, token, payload))
                raise
            except Exception as exc:
                payload = {
                    "status": "unknown" if reconciliation is not None else "error",
                    "content": "",
                    "termination_reason": type(exc).__name__,
                    "error": str(exc),
                }
            committed = await self.store.finish(job_id, token, payload)
            return committed
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if agent is not None:
                try:
                    await agent.aclose()
                except Exception as exc:
                    # Cleanup must not erase a durably committed execution result.
                    logging.getLogger("wuwei.jobs").warning(
                        "Job %s resource cleanup failed (%s)", job_id, type(exc).__name__
                    )
                    if committed is not None:
                        entry = {"action": "cleanup_failed", "error_type": type(exc).__name__}
                        try:
                            updated = await self.store.annotate(job_id, entry)
                            committed.audit = updated.audit
                        except Exception:
                            logging.getLogger("wuwei.jobs").warning(
                                "Could not persist job cleanup diagnostic"
                            )
