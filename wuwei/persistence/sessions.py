"""Versioned conversation snapshots with cross-instance fenced ownership.

The namespace is selected by a trusted host, never by model-generated input.
A lease prevents concurrent workers; expired owners cannot commit stale state.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from wuwei.agent.session import AgentSession


class SessionBusy(RuntimeError):
    """Another execution owns the session, or ownership expired."""


class SessionStore(Protocol):
    async def load(self, session_id: str) -> AgentSession | None: ...
    async def delete(self, session_id: str) -> None: ...
    def checkout(self, session: AgentSession): ...


class SQLiteSessionStore:
    """SQLite durable session store with optional TTL and renewed execution lease."""

    def __init__(
        self,
        path=".wuwei/sessions.db",
        *,
        namespace="default",
        ttl_seconds: float | None = None,
        lease_seconds=60,
    ):
        if not namespace or lease_seconds <= 0 or (ttl_seconds is not None and ttl_seconds <= 0):
            raise ValueError("namespace and TTL/lease must be nonempty/positive")
        self.path, self.namespace = str(path), namespace
        self.ttl_seconds, self.lease_seconds = ttl_seconds, lease_seconds
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (
                namespace TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0, expires REAL, token TEXT, lease_until REAL,
                PRIMARY KEY(namespace,id))""")

    def _expires(self, session, now):
        if session.metadata.get("_blocked_run") or session.metadata.get("_pending_interrupt"):
            return None  # Do not TTL away unresolved actions/approval truth.
        return now + self.ttl_seconds if self.ttl_seconds else None

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    async def load(self, session_id):
        def read():
            with self._connect() as db:
                row = db.execute(
                    "SELECT payload,expires FROM sessions WHERE namespace=? AND id=?",
                    (self.namespace, session_id),
                ).fetchone()
            if row is None or (row[1] is not None and row[1] <= time.time()):
                return None
            return AgentSession.from_dict(json.loads(row[0]))

        return await asyncio.to_thread(read)

    async def delete(self, session_id):
        def delete():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT lease_until,payload FROM sessions WHERE namespace=? AND id=?",
                    (self.namespace, session_id),
                ).fetchone()
                if row and row[0] is not None and row[0] > time.time():
                    raise SessionBusy("Cannot delete an executing session")
                if row:
                    data = json.loads(row[1])
                    metadata = data.get("metadata") or {}
                    if (
                        metadata.get("_blocked_run")
                        or metadata.get("_pending_interrupt")
                        or row[0] is not None
                    ):
                        raise ValueError(
                            "Reconcile/resume unresolved session before deleting history"
                        )
                db.execute(
                    "DELETE FROM sessions WHERE namespace=? AND id=?", (self.namespace, session_id)
                )

        await asyncio.to_thread(delete)

    def _acquire(self, session, token):
        payload = json.dumps(session.to_dict(), ensure_ascii=False)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload,revision,expires,lease_until FROM sessions "
                "WHERE namespace=? AND id=?",
                (self.namespace, session.session_id),
            ).fetchone()
            if row and row[3] is not None and row[3] > now:
                raise SessionBusy("Session is executing in another worker")
            if row and row[3] is not None:
                # An owner disappeared without committing. Its tools may already
                # have acted; never expose its old clean snapshot as a fresh run.
                interrupted = AgentSession.from_dict(json.loads(row[0]))
                interrupted.metadata["_blocked_run"] = {"reason": "session_lease_expired"}
                payload, revision = json.dumps(interrupted.to_dict()), row[1]
            elif row and (row[2] is None or row[2] > now):
                payload, revision = row[0], row[1]
            else:
                revision = 0
            AgentSession.from_dict(
                json.loads(payload)
            )  # Validate before acquiring a durable lease.
            db.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?)",
                (
                    self.namespace,
                    session.session_id,
                    payload,
                    revision,
                    None,
                    token,
                    now + self.lease_seconds,
                ),
            )
        return payload, revision

    def _renew(self, session_id, token):
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE sessions SET lease_until=? WHERE namespace=? AND id=? "
                "AND token=? AND lease_until>?",
                (now + self.lease_seconds, self.namespace, session_id, token, now),
            )
            if cursor.rowcount != 1:
                raise SessionBusy("Session lease lost; stale execution cannot commit")

    def _commit(self, session, token, revision):
        payload, now = json.dumps(session.to_dict(), ensure_ascii=False), time.time()
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE sessions SET payload=?,revision=revision+1,expires=?, "
                "token=NULL,lease_until=NULL WHERE namespace=? AND id=? "
                "AND token=? AND revision=? AND lease_until>?",
                (
                    payload,
                    self._expires(session, now),
                    self.namespace,
                    session.session_id,
                    token,
                    revision,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise SessionBusy("Session lease lost; stale state was not saved")

    @asynccontextmanager
    async def checkout(self, session):
        token = uuid4().hex
        payload, revision = await asyncio.to_thread(self._acquire, session, token)
        restored = AgentSession.from_dict(json.loads(payload))
        for name in (
            "context",
            "metadata",
            "system_prompt",
            "max_steps",
            "parallel_tool_calls",
            "summary",
            "last_usage",
            "last_latency_ms",
            "last_llm_calls",
        ):
            setattr(session, name, getattr(restored, name))
        parent, lease_error = asyncio.current_task(), []

        async def heartbeat():
            try:
                while True:
                    await asyncio.sleep(self.lease_seconds / 3)
                    await asyncio.to_thread(self._renew, session.session_id, token)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                lease_error.append(exc)
                parent.cancel()

        task = asyncio.create_task(heartbeat())
        try:
            yield session
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if lease_error:
                raise SessionBusy("Session ownership lost during execution") from lease_error[0]
            # Cancellation still records committed history and unknown-side-effect blockers.
            await asyncio.shield(asyncio.to_thread(self._commit, session, token, revision))
