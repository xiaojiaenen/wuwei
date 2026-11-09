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
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

ACTION_LOG_KEY = "action_log"
ACTION_LOG_TYPE = "ActionLog"

__all__ = [
    "ACTION_LOG_KEY",
    "ACTION_LOG_TYPE",
    "ActionEntry",
    "ActionLog",
    "coerce_action_log",
    "get_or_create_log",
]


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
                e
                for e in matched
                if q in e.summary.lower()
                or q in json.dumps(e.data, ensure_ascii=False, default=str).lower()
            ]
        return matched[-limit:]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionLog:
        """还原日志快照。

        历史上存在两种落盘形态，都必须认：

        - ``{"__type__": "ActionLog", "value": {"entries": [...]}}``
          —— ``AgentSession.to_dict`` 从 3.2 起写的带标记形态；
        - ``{"entries": [...]}`` —— 直接调 ``ActionLog.to_dict()`` 得到的裸形态，
          旧 checkpoint 与外部写入方都可能是这个。

        只认裸形态的话，把带标记的 dict 喂进来会得到 ``entries=[]``，
        历史被静默清空 —— 比崩在 append 上更难发现，所以宁可报错也不猜。
        """
        if not isinstance(data, dict):
            raise TypeError(f"ActionLog 快照需要 dict，实际是 {type(data).__name__}")

        payload = data
        if data.get("__type__") == ACTION_LOG_TYPE:
            payload = data.get("value")
            if not isinstance(payload, dict):
                raise ValueError("ActionLog 快照的 value 字段不是 dict")
        if payload and "entries" not in payload:
            raise ValueError("ActionLog 快照缺少 entries 字段，拒绝按空日志静默处理")

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
            for e in payload.get("entries", [])
        ]
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)


def coerce_action_log(value: Any) -> ActionLog:
    """把 ``metadata[action_log]`` 的任意历史形态收敛成 ActionLog 实例。

    ``session.metadata`` 是跨持久化的无类型边界：正常路径由
    ``AgentSession.to_dict / from_dict`` 用 ``__type__`` 标记还原，但
    3.2 之前的 checkpoint 没有标记，第三方 SessionStore、外部代码、
    手工构造的 session 也可能直接塞一个裸 dict 进来。写侧再规范，
    读侧也不能假设拿到的类型 —— 这里就是唯一的收敛点。
    """
    if isinstance(value, ActionLog):
        return value
    if value is None:
        return ActionLog()
    if isinstance(value, Mapping):
        return ActionLog.from_dict(dict(value))
    raise TypeError(
        f"session.metadata['{ACTION_LOG_KEY}'] 期望 ActionLog / dict / None，"
        f"实际是 {type(value).__name__}；请检查 checkpoint 版本或写入方"
    )


def get_or_create_log(session) -> ActionLog:
    """从 session.metadata 取（或创建）会话级 Action Log。

    无论新建、还原还是原样返回，都把实例回填 metadata：不回填的话，
    每次调用都要重跑一次全量反序列化，而且同一会话里可能拿到不同实例，
    子代理写入的记录会丢。
    """
    current = session.metadata.get(ACTION_LOG_KEY)
    log = coerce_action_log(current)
    if log is not current:
        session.metadata[ACTION_LOG_KEY] = log
    return log
