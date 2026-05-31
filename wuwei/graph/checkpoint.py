"""检查点系统

借鉴 LangGraph 的 checkpoint 设计：
- 每个检查点记录 channel_values（通道级序列化）
- channel_versions 支持增量检查点
- 支持 intermediate writes（中间写入，用于流式恢复）
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import json
import os
import uuid
from datetime import datetime, timezone

from wuwei.graph.state import State


def _generate_checkpoint_id() -> str:
    """生成单调递增的检查点 ID"""
    ts = datetime.now(timezone.utc).isoformat()
    return f"ckpt_{ts}_{uuid.uuid4().hex[:8]}"


class BaseCheckpointer(ABC):
    """检查点基类"""

    @abstractmethod
    async def save(
        self,
        state: State,
        checkpoint_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """保存检查点

        Args:
            state: 要保存的状态
            checkpoint_id: 检查点 ID（可选，自动生成）
            metadata: 附加元数据

        Returns:
            检查点 ID
        """
        ...

    @abstractmethod
    async def load(self, checkpoint_id: str) -> State:
        """加载检查点"""
        ...

    @abstractmethod
    async def list_checkpoints(
        self,
        limit: int = 10,
        before: str | None = None,
    ) -> list[dict]:
        """列出检查点

        Args:
            limit: 返回数量上限
            before: 仅返回此 ID 之前的检查点（用于分页）
        """
        ...

    async def put_writes(
        self,
        checkpoint_id: str,
        writes: list[dict],
        task_id: str,
    ) -> None:
        """保存中间写入（用于流式恢复）

        子类可覆写以支持流式中断恢复。
        默认空操作。
        """
        pass

    @staticmethod
    def _serialize_state(state: State) -> dict:
        """安全序列化 State 为 JSON 兼容字典

        使用 State.to_dict() 进行结构化序列化，
        避免 default=str 导致的数据损坏。
        """
        return state.to_dict()

    @staticmethod
    def _deserialize_state(data: dict) -> State:
        """从 JSON 兼容字典反序列化 State"""
        return State.from_dict(data)


class MemoryCheckpointer(BaseCheckpointer):
    """内存检查点（默认）

    用于开发和测试，不持久化。
    """

    def __init__(self):
        self.checkpoints: dict[str, dict] = {}
        self._writes: dict[str, list[dict]] = {}

    async def save(
        self,
        state: State,
        checkpoint_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """保存到内存"""
        checkpoint_id = checkpoint_id or _generate_checkpoint_id()
        self.checkpoints[checkpoint_id] = {
            "state": self._serialize_state(state),
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parent_checkpoint_id": metadata.get("parent_checkpoint_id") if metadata else None,
        }
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> State:
        """从内存加载"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点不存在: {checkpoint_id}")
        return self._deserialize_state(self.checkpoints[checkpoint_id]["state"])

    async def list_checkpoints(
        self,
        limit: int = 10,
        before: str | None = None,
    ) -> list[dict]:
        """列出检查点"""
        items = list(self.checkpoints.items())
        if before:
            # 只返回 before 之前的检查点
            before_idx = next((i for i, (k, _) in enumerate(items) if k == before), len(items))
            items = items[:before_idx]
        return [
            {"id": k, "metadata": v.get("metadata", {}), "timestamp": v.get("timestamp", "")}
            for k, v in items[-limit:]
        ]

    async def put_writes(
        self,
        checkpoint_id: str,
        writes: list[dict],
        task_id: str,
    ) -> None:
        """保存中间写入"""
        if checkpoint_id not in self._writes:
            self._writes[checkpoint_id] = []
        self._writes[checkpoint_id].extend(writes)


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite 检查点

    持久化到 SQLite 数据库，支持：
    - 结构化序列化（不使用 default=str）
    - 元数据索引
    - 按时间排序的分页查询
    """

    def __init__(self, db_path: str = ".wuwei/checkpoints.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        import sqlite3

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                parent_checkpoint_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                writes TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_created
            ON checkpoints(created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_writes_checkpoint
            ON checkpoint_writes(checkpoint_id)
        """)
        conn.commit()
        conn.close()

    async def save(
        self,
        state: State,
        checkpoint_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """保存到 SQLite

        使用结构化序列化，将 State.to_dict() 的结果以 JSON 存储。
        metadata 以独立列存储，支持高效索引。
        """
        import sqlite3

        checkpoint_id = checkpoint_id or _generate_checkpoint_id()
        state_json = json.dumps(self._serialize_state(state), ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        parent_id = metadata.get("parent_checkpoint_id") if metadata else None

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(id, state, metadata, parent_checkpoint_id) VALUES (?, ?, ?, ?)",
            (checkpoint_id, state_json, metadata_json, parent_id),
        )
        conn.commit()
        conn.close()
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> State:
        """从 SQLite 加载

        正确反序列化 JSON，不使用 default=str 损坏数据。
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT state FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        conn.close()

        if not row:
            raise ValueError(f"检查点不存在: {checkpoint_id}")

        return self._deserialize_state(json.loads(row[0]))

    async def list_checkpoints(
        self,
        limit: int = 10,
        before: str | None = None,
    ) -> list[dict]:
        """列出检查点，支持分页"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        if before:
            # 获取 before 检查点的时间戳用于分页
            before_row = conn.execute(
                "SELECT created_at FROM checkpoints WHERE id = ?", (before,)
            ).fetchone()
            if before_row:
                rows = conn.execute(
                    "SELECT id, metadata, created_at, parent_checkpoint_id "
                    "FROM checkpoints "
                    "WHERE created_at < ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (before_row[0], limit),
                ).fetchall()
            else:
                rows = []
        else:
            rows = conn.execute(
                "SELECT id, metadata, created_at, parent_checkpoint_id "
                "FROM checkpoints "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "metadata": json.loads(row[1]) if row[1] else {},
                "timestamp": row[2],
                "parent_checkpoint_id": row[3],
            }
            for row in rows
        ]

    async def put_writes(
        self,
        checkpoint_id: str,
        writes: list[dict],
        task_id: str,
    ) -> None:
        """保存中间写入到 SQLite"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO checkpoint_writes (checkpoint_id, task_id, writes) VALUES (?, ?, ?)",
            (checkpoint_id, task_id, json.dumps(writes, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
