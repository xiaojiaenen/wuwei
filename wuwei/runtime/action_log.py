"""共享动作日志（Action Log）

一次 run 内所有 agent（主 agent + 子代理）往 append-only 日志写入
每一步的动作摘要，用于：
- 多 agent 决策一致性（子代理可读取相关片段，回应"决策碎片化"痛点）
- 运行审计
- time-travel 回放的索引数据

日志通过 session.metadata 传递（一次会话一份），子代理继承父代理的日志。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

ACTION_LOG_KEY = "action_log"


@dataclass
class ActionEntry:
    """一条动作记录。"""

    run_id: str
    agent: str
    step: int
    kind: str  # "llm" | "tool" | "interrupt" | "resume" | "error"
    summary: str
    digest: str = ""  # 内容寻址摘要（content/args 的 hash 前缀）
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        if not self.digest and self.data:
            raw = json.dumps(self.data, ensure_ascii=False, sort_keys=True, default=str)
            self.digest = hashlib.sha1(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "step": self.step,
            "kind": self.kind,
            "summary": self.summary,
            "digest": self.digest,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class ActionLog:
    """Append-only 动作日志（内存态，可整体序列化进 checkpoint）。"""

    def __init__(self, entries: list[ActionEntry] | None = None):
        self.entries: list[ActionEntry] = entries or []

    def append(
        self,
        *,
        run_id: str,
        agent: str,
        step: int,
        kind: str,
        summary: str,
        data: dict[str, Any] | None = None,
    ) -> ActionEntry:
        entry = ActionEntry(
            run_id=run_id,
            agent=agent,
            step=step,
            kind=kind,
            summary=summary,
            data=data or {},
        )
        self.entries.append(entry)
        return entry

    def read(
        self,
        query: str | None = None,
        *,
        kind: str | None = None,
        agent: str | None = None,
        limit: int = 20,
    ) -> list[ActionEntry]:
        """按关键词/类型/agent 过滤，返回最近的记录。"""
        matched = self.entries
        if kind:
            matched = [e for e in matched if e.kind == kind]
        if agent:
            matched = [e for e in matched if e.agent == agent]
        if query:
            q = query.lower()
            matched = [
                e for e in matched
                if q in e.summary.lower()
                or q in json.dumps(e.data, ensure_ascii=False, default=str).lower()
            ]
        return matched[-limit:]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionLog:
        entries = [
            ActionEntry(
                run_id=e.get("run_id", ""),
                agent=e.get("agent", ""),
                step=e.get("step", 0),
                kind=e.get("kind", ""),
                summary=e.get("summary", ""),
                digest=e.get("digest", ""),
                timestamp=e.get("timestamp", ""),
                data=e.get("data") or {},
            )
            for e in data.get("entries", [])
        ]
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)


def get_or_create_log(session) -> ActionLog:
    """从 session.metadata 取（或创建）会话级 Action Log。"""
    log = session.metadata.get(ACTION_LOG_KEY)
    if log is None:
        log = ActionLog()
        session.metadata[ACTION_LOG_KEY] = log
    return log
