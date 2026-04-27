from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from wuwei.agent.session import AgentSession
    from wuwei.llm import Message


class Storage(Protocol):
    """持久化存储协议。消息增量追加，元数据覆盖保存。"""

    async def save_meta(self, session: AgentSession) -> None:
        """保存会话元数据（不含消息）。"""
        ...

    async def append_message(self, session_id: str, message: Message) -> None:
        """追加一条消息。"""
        ...

    async def load(self, session_id: str) -> AgentSession | None:
        """加载完整会话（元数据 + 全部消息）。"""
        ...

    async def delete(self, session_id: str) -> None:
        """删除会话。"""
        ...


class FileStorage:
    """文件存储：meta.json 存元数据，jsonl 逐条追加消息。"""

    def __init__(self, root: str | Path = ".wuwei_sessions"):
        self.root = Path(root)

    async def save_meta(self, session: AgentSession) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": session.session_id,
            "system_prompt": session.system_prompt,
            "max_steps": session.max_steps,
            "parallel_tool_calls": session.parallel_tool_calls,
            "summary": session.summary,
            "metadata": session.metadata,
            "last_usage": session.last_usage,
            "last_latency_ms": session.last_latency_ms,
            "last_llm_calls": session.last_llm_calls,
        }
        path = self.root / f"{session.session_id}.meta.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    async def append_message(self, session_id: str, message: Message) -> None:
        path = self.root / f"{session_id}.jsonl"
        line = json.dumps(message.model_dump(exclude_none=True), ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def load(self, session_id: str) -> AgentSession | None:
        from wuwei.agent.session import AgentSession

        meta_path = self.root / f"{session_id}.meta.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        session = AgentSession.from_dict(meta)
        session.context.add_system_message(session.system_prompt)

        msg_path = self.root / f"{session_id}.jsonl"
        if msg_path.exists():
            from wuwei.llm import Message

            for line in msg_path.read_text(encoding="utf-8").strip().splitlines():
                if line:
                    session.context._messages.append(
                        Message.model_validate(json.loads(line))
                    )

        return session

    async def delete(self, session_id: str) -> None:
        for ext in (".meta.json", ".jsonl"):
            path = self.root / f"{session_id}{ext}"
            if path.exists():
                path.unlink()
