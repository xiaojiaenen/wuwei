"""JSON checkpoints with atomic, one-shot continuation claims.

A consumed or non-current checkpoint cannot be replayed. A tool_running snapshot
is deliberately not automatically recoverable: remote effects need reconciliation.
"""

import asyncio
import json
from contextlib import closing
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from uuid import uuid4

from wuwei.graph.state import State


def _generate_checkpoint_id() -> str:
    return f"ckpt_{datetime.now(UTC).isoformat()}_{uuid4().hex[:8]}"


def _encode_state_dict(state: State, metadata: dict | None) -> dict:
    """编码为快照 dict。

    v2（调用方显式传入 metadata 时）：runner 的 metadata 只存于独立列，快照列
    不再内嵌同一份——此前同一份 metadata 在 state 列与 metadata 列各写一次；
    runner 的消息另以 ``State.messages`` 与 ``metadata.session.context`` 双份
    落盘。v2 快照列只保留 messages/step（调用方自带的 state.metadata 非空时
    照旧保留——那是 StateGraph 直存路径的既有语义，不是重复）。v1（无
    metadata 参数）：形状不变，向后兼容。
    """
    data = state.to_dict()
    if metadata:
        if not data.get("metadata"):
            data.pop("metadata", None)
        data["schema_version"] = 2
    return data


def _decode_state(snapshot: dict, metadata: dict | None) -> State:
    """解码快照；v2 时 metadata 取自独立列，并从会话副本重灌 messages。

    重灌保持所有 ``state.messages`` 消费方（resume 恢复上下文、debug.list_steps、
    fork 重放）零改动；消息对象与保存前的 ``session.context`` 同型。
    v1 快照内嵌 metadata，形状照旧。
    """
    if snapshot.get("schema_version") == 2:
        state = State(step=snapshot.get("step", 0))
        snapshot_meta = snapshot.get("metadata") or {}
        # StateGraph 直存路径的 state.metadata 优先（与 v1 消费行为一致）；
        # runner 路径快照 metadata 为空，取独立列。
        state.metadata = snapshot_meta if snapshot_meta else (metadata or {})
        if snapshot.get("messages"):
            state.messages = State.from_dict(
                {"messages": snapshot["messages"]}
            ).messages
        else:
            session = (metadata or {}).get("session") or {}
            context = session.get("context") or {}
            if context.get("messages"):
                from wuwei.memory.context import Context

                state.messages = Context.from_dict(context).get_messages()
        return state
    return State.from_dict(snapshot)


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

    async def replace_current(self, checkpoint_id: str, state: State, *, metadata: dict) -> str:
        """Consume and replace one current checkpoint in a single transaction."""
        raise NotImplementedError("Atomic replacement is required for reconciliation")

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
        snapshot = json.loads(json.dumps(_encode_state_dict(state, metadata), ensure_ascii=False))
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
        record = self.checkpoints[checkpoint_id]
        snapshot = json.loads(json.dumps(record["state"]))
        metadata = json.loads(json.dumps(record["metadata"]))
        return _decode_state(snapshot, metadata)

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

    async def replace_current(self, checkpoint_id, state, *, metadata):
        # Serialize before consuming so invalid payloads leave the old state usable.
        snapshot = json.loads(
            json.dumps(_encode_state_dict(state, metadata), ensure_ascii=False)
        )
        meta = json.loads(json.dumps(metadata, ensure_ascii=False))
        record = self.checkpoints.get(checkpoint_id)
        if record is None or checkpoint_id in self._claims:
            raise ValueError("Checkpoint was already consumed or does not exist")
        run_id = record["metadata"].get("run_id")
        if run_id and self._heads.get(run_id) != checkpoint_id:
            raise ValueError("Checkpoint was superseded")
        if meta.get("run_id") != run_id:
            raise ValueError("Replacement must belong to the same run")
        cid = _generate_checkpoint_id()
        self.checkpoints[cid] = {
            "state": snapshot,
            "metadata": meta,
            "timestamp": datetime.now(UTC).isoformat(),
            "parent_checkpoint_id": checkpoint_id,
        }
        self._claims.add(checkpoint_id)
        if run_id:
            self._heads[run_id] = cid
        return cid

    async def put_writes(self, checkpoint_id, writes, task_id):
        self._writes.setdefault(checkpoint_id, []).extend(writes)


class SQLiteCheckpointer(BaseCheckpointer):
    def __init__(self, db_path=".wuwei/checkpoints.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with closing(sqlite3.connect(db_path)) as db, db:
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
        snapshot = json.dumps(_encode_state_dict(state, metadata), ensure_ascii=False)
        meta_json = json.dumps(metadata, ensure_ascii=False)

        def write():
            with closing(sqlite3.connect(self.db_path)) as db, db:
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
            with closing(sqlite3.connect(self.db_path)) as db, db:
                row = db.execute(
                    "SELECT state,metadata FROM checkpoints WHERE id=?", (checkpoint_id,)
                ).fetchone()
            if not row:
                raise ValueError(f"检查点不存在: {checkpoint_id}")
            return _decode_state(json.loads(row[0]), json.loads(row[1] or "{}"))

        return await asyncio.to_thread(read)

    async def list_checkpoints(self, limit=10, before=None):
        if limit <= 0:
            return []

        def read():
            with closing(sqlite3.connect(self.db_path)) as db, db:
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
            with closing(sqlite3.connect(self.db_path)) as db, db:
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

    async def replace_current(self, checkpoint_id, state, *, metadata):
        snapshot = json.dumps(_encode_state_dict(state, metadata), ensure_ascii=False)
        meta_json = json.dumps(metadata, ensure_ascii=False)
        cid = _generate_checkpoint_id()

        def replace():
            with closing(sqlite3.connect(self.db_path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                record = db.execute(
                    "SELECT metadata FROM checkpoints WHERE id=?", (checkpoint_id,)
                ).fetchone()
                if record is None:
                    raise ValueError("Checkpoint does not exist")
                run_id = json.loads(record[0]).get("run_id")
                if metadata.get("run_id") != run_id:
                    raise ValueError("Replacement must belong to the same run")
                if run_id:
                    head = db.execute(
                        "SELECT checkpoint_id FROM run_heads WHERE run_id=?", (run_id,)
                    ).fetchone()
                    if head and head[0] != checkpoint_id:
                        raise ValueError("Checkpoint was superseded")
                used = db.execute(
                    "INSERT OR IGNORE INTO checkpoint_claims VALUES (?)", (checkpoint_id,)
                )
                if used.rowcount != 1:
                    raise ValueError("Checkpoint was already consumed")
                db.execute(
                    "INSERT INTO checkpoints (id,state,metadata,parent_checkpoint_id,created_at) VALUES (?,?,?,?,?)",
                    (cid, snapshot, meta_json, checkpoint_id, datetime.now(UTC).isoformat()),
                )
                if run_id:
                    db.execute("INSERT OR REPLACE INTO run_heads VALUES (?,?)", (run_id, cid))
            return cid

        return await asyncio.to_thread(replace)

    async def put_writes(self, checkpoint_id, writes, task_id):
        payload = json.dumps(writes, ensure_ascii=False)

        def write():
            with closing(sqlite3.connect(self.db_path)) as db, db:
                db.execute(
                    "INSERT INTO checkpoint_writes (checkpoint_id,task_id,writes) VALUES (?,?,?)",
                    (checkpoint_id, task_id, payload),
                )

        await asyncio.to_thread(write)
