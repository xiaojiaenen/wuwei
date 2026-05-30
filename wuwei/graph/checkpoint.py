"""检查点系统"""

from abc import ABC, abstractmethod
from typing import Optional
import json
import os
from datetime import datetime
from wuwei.graph.state import State


class BaseCheckpointer(ABC):
    """检查点基类"""

    @abstractmethod
    async def save(self, state: State, checkpoint_id: str = None) -> str:
        """保存检查点

        Args:
            state: 要保存的状态
            checkpoint_id: 检查点 ID（可选）

        Returns:
            检查点 ID
        """
        ...

    @abstractmethod
    async def load(self, checkpoint_id: str) -> State:
        """加载检查点"""
        ...

    @abstractmethod
    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        """列出检查点"""
        ...


class MemoryCheckpointer(BaseCheckpointer):
    """内存检查点（默认）"""

    def __init__(self):
        self.checkpoints: dict[str, dict] = {}

    async def save(self, state: State, checkpoint_id: str = None) -> str:
        """保存到内存"""
        checkpoint_id = checkpoint_id or f"cp_{datetime.now().isoformat()}"
        self.checkpoints[checkpoint_id] = {
            "state": state.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> State:
        """从内存加载"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点不存在: {checkpoint_id}")
        return State.from_dict(self.checkpoints[checkpoint_id]["state"])

    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        """列出检查点"""
        items = list(self.checkpoints.items())[-limit:]
        return [{"id": k, **v} for k, v in items]


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite 检查点"""

    def __init__(self, db_path: str = ".wuwei/checkpoints.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        import sqlite3

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    async def save(self, state: State, checkpoint_id: str = None) -> str:
        """保存到 SQLite"""
        import sqlite3

        checkpoint_id = checkpoint_id or f"cp_{datetime.now().isoformat()}"
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (id, state) VALUES (?, ?)",
            (checkpoint_id, json.dumps(state.to_dict(), default=str)),
        )
        conn.commit()
        conn.close()
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> State:
        """从 SQLite 加载"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT state FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        conn.close()

        if not row:
            raise ValueError(f"检查点不存在: {checkpoint_id}")

        return State.from_dict(json.loads(row[0]))

    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        """列出检查点"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, created_at FROM checkpoints ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()

        return [{"id": row[0], "created_at": row[1]} for row in rows]
