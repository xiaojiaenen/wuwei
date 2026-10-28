"""JSON checkpoints with atomic, one-shot continuation claims.

A consumed or non-current checkpoint cannot be replayed. A tool_running snapshot
is deliberately not automatically recoverable: remote effects need reconciliation.
"""

import asyncio
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from uuid import uuid4

from wuwei.graph.state import State


def _generate_checkpoint_id() -> str:
    return f"ckpt_{datetime.now(UTC).isoformat()}_{uuid4().hex[:8]}"


class BaseCheckpointer(ABC):
    @abstractmethod
    async def save(
        self, state: State, checkpoint_id: str | None = None, metadata: dict | None = None
    ) -> str: ...

    @abstractmethod
    async def load(self, checkpoint_id: str) -> State: ...

    @abstractmethod
    async def list_checkpoints(self, limit: int = 10, before: str | None = None) -> list[dict]: ...

    async def claim(self, checkpoint_id: str) -> bool:
        """Atomically consume the current continuation, or fail closed on old backends."""
        raise NotImplementedError("This checkpointer must implement atomic claim() for resume")

    async def put_writes(self, checkpoint_id: str, writes: list[dict], task_id: str) -> None:
        pass

    @staticmethod
    def _serialize_state(state: State) -> dict:
        return state.to_dict()

    @staticmethod
    def _deserialize_state(data: dict) -> State:
        return State.from_dict(data)


class MemoryCheckpointer(BaseCheckpointer):
    def __init__(self):
        self.checkpoints = {}
        self._writes = {}
        self._claims = set()
        self._heads = {}

    async def save(self, state, checkpoint_id=None, metadata=None):
        cid = checkpoint_id or _generate_checkpoint_id()
        metadata = metadata or {}
        # JSON roundtrip makes this backend's copy and serializability semantics match SQLite.
        snapshot = json.loads(json.dumps(state.to_dict(), ensure_ascii=False))
        self.checkpoints[cid] = {
            "state": snapshot,
            "metadata": dict(metadata),
            "timestamp": datetime.now(UTC).isoformat(),
            "parent_checkpoint_id": metadata.get("parent_checkpoint_id"),
        }
        if metadata.get("run_id"):
            self._heads[metadata["run_id"]] = cid
        return cid

    async def load(self, checkpoint_id):
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点不存在: {checkpoint_id}")
        return State.from_dict(json.loads(json.dumps(self.checkpoints[checkpoint_id]["state"])))

    async def list_checkpoints(self, limit=10, before=None):
        if limit <= 0:
            return []
        items = list(self.checkpoints.items())
        if before:
            index = next((i for i, (key, _) in enumerate(items) if key == before), 0)
            items = items[:index]
        return [
            {"id": key, **{k: v for k, v in value.items() if k != "state"}}
            for key, value in reversed(items[-limit:])
        ]

    async def claim(self, checkpoint_id):
        record = self.checkpoints.get(checkpoint_id)
        if record is None or checkpoint_id in self._claims:
            return False
        run_id = record["metadata"].get("run_id")
        if run_id and self._heads.get(run_id) != checkpoint_id:
            return False
        self._claims.add(checkpoint_id)
        return True

    async def put_writes(self, checkpoint_id, writes, task_id):
        self._writes.setdefault(checkpoint_id, []).extend(writes)


class SQLiteCheckpointer(BaseCheckpointer):
    def __init__(self, db_path=".wuwei/checkpoints.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with sqlite3.connect(db_path) as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY, state TEXT NOT NULL, metadata TEXT DEFAULT '{}',
                parent_checkpoint_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL, writes TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS checkpoint_claims (checkpoint_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS run_heads (run_id TEXT PRIMARY KEY, checkpoint_id TEXT NOT NULL);
            """)

    async def save(self, state, checkpoint_id=None, metadata=None):
        cid = checkpoint_id or _generate_checkpoint_id()
        metadata = metadata or {}
        snapshot = json.dumps(state.to_dict(), ensure_ascii=False)
        meta_json = json.dumps(metadata, ensure_ascii=False)

        def write():
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    "INSERT OR REPLACE INTO checkpoints (id,state,metadata,parent_checkpoint_id,created_at) VALUES (?,?,?,?,?)",
                    (
                        cid,
                        snapshot,
                        meta_json,
                        metadata.get("parent_checkpoint_id"),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                if metadata.get("run_id"):
                    db.execute(
                        "INSERT OR REPLACE INTO run_heads VALUES (?,?)", (metadata["run_id"], cid)
                    )
            return cid

        return await asyncio.to_thread(write)

    async def load(self, checkpoint_id):
        def read():
            with sqlite3.connect(self.db_path) as db:
                row = db.execute(
                    "SELECT state FROM checkpoints WHERE id=?", (checkpoint_id,)
                ).fetchone()
            if not row:
                raise ValueError(f"检查点不存在: {checkpoint_id}")
            return State.from_dict(json.loads(row[0]))

        return await asyncio.to_thread(read)

    async def list_checkpoints(self, limit=10, before=None):
        if limit <= 0:
            return []

        def read():
            with sqlite3.connect(self.db_path) as db:
                if before:
                    cursor = db.execute(
                        "SELECT rowid FROM checkpoints WHERE id=?", (before,)
                    ).fetchone()
                    if not cursor:
                        return []
                    rows = db.execute(
                        "SELECT id,metadata,created_at,parent_checkpoint_id FROM checkpoints WHERE rowid<? ORDER BY rowid DESC LIMIT ?",
                        (cursor[0], limit),
                    ).fetchall()
                else:
                    rows = db.execute(
                        "SELECT id,metadata,created_at,parent_checkpoint_id FROM checkpoints ORDER BY rowid DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            return [
                {
                    "id": r[0],
                    "metadata": json.loads(r[1]),
                    "timestamp": r[2],
                    "parent_checkpoint_id": r[3],
                }
                for r in rows
            ]

        return await asyncio.to_thread(read)

    async def claim(self, checkpoint_id):
        def consume():
            with sqlite3.connect(self.db_path) as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT metadata FROM checkpoints WHERE id=?", (checkpoint_id,)
                ).fetchone()
                if not row:
                    return False
                run_id = json.loads(row[0]).get("run_id")
                if run_id:
                    head = db.execute(
                        "SELECT checkpoint_id FROM run_heads WHERE run_id=?", (run_id,)
                    ).fetchone()
                    if head and head[0] != checkpoint_id:
                        return False
                cursor = db.execute(
                    "INSERT OR IGNORE INTO checkpoint_claims VALUES (?)", (checkpoint_id,)
                )
                return cursor.rowcount == 1

        return await asyncio.to_thread(consume)

    async def put_writes(self, checkpoint_id, writes, task_id):
        payload = json.dumps(writes, ensure_ascii=False)

        def write():
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    "INSERT INTO checkpoint_writes (checkpoint_id,task_id,writes) VALUES (?,?,?)",
                    (checkpoint_id, task_id, payload),
                )

        await asyncio.to_thread(write)
