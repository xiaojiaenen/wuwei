"""Context Ledger — 上下文保真账本

正面回答"压缩=失忆"这一最高频抱怨（Chroma Context Rot / HN compaction 声讨）：

- 超过阈值的大块工具结果不再原样留在窗口，而是以内容寻址（hash）方式
  存入账本，窗口内只留一条带 id 的占位引用；
- 模型随时可通过 `read_context(id)` 工具按需取回全文（半自动召回）；
- `pin_context(id)` 可钉住关键内容永不归档；
- 账本是"全量真相"，压缩只发生在窗口视图——任何时刻都能完整还原。

与 Compaction 中间件互补：Ledger 在写入时就外置大块内容，
让压缩需要处理的高信号密度上下文变小。

用法：
    ledger = ContextLedger()
    agent = Agent(llm=llm, middleware=MiddlewareStack(LedgerMiddleware(ledger)))
    agent.tool_registry.register_many(ledger_tools(ledger))  # read/pin/list
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.tools import Tool, ToolParameters


class LedgerEntry(BaseModel):
    id: str
    kind: str = "tool_result"
    content: str
    source: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    pinned: bool = False


class ContextLedger:
    """内容寻址账本（内存 + 可选文件持久化）。"""

    def __init__(self, persist_dir: str | None = None, max_entries: int = 5000):
        self._entries: dict[str, LedgerEntry] = {}
        self._archive: dict[str, LedgerEntry] = {}
        self._order: list[str] = []
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
        self.max_entries = max_entries

    # ── 存储 ─────────────────────────────────────────────────────────

    def put(self, content: str, *, kind: str = "tool_result", source: str = "") -> str:
        """存入内容，返回稳定 id（内容寻址：同内容同 id，天然去重）。"""
        entry_id = hashlib.sha1(content.encode()).hexdigest()[:12]
        if self.get(entry_id) is not None:
            return entry_id
        entry = LedgerEntry(id=entry_id, kind=kind, content=content, source=source)
        self._entries[entry_id] = entry
        self._order.append(entry_id)
        if self.persist_dir:
            self._dump_entry(entry)
        self._evict()
        return entry_id

    def get(self, entry_id: str) -> LedgerEntry | None:
        entry = self._entries.get(entry_id) or self._archive.get(entry_id)
        if entry is None and self.persist_dir and re.fullmatch(r"[0-9a-f]{12}", entry_id):
            path = self.persist_dir / f"{entry_id}.json"
            if path.exists():
                entry = LedgerEntry.model_validate_json(path.read_text(encoding="utf-8"))
        return entry

    def pin(self, entry_id: str, pinned: bool = True) -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.pinned = pinned
        if self.persist_dir:
            self._dump_entry(entry)
        return True

    def list_entries(self, limit: int = 20) -> list[LedgerEntry]:
        return [self._entries[i] for i in self._order[-limit:]]

    def __len__(self) -> int:
        return len(self._entries)

    def _evict(self) -> None:
        """超限时淘汰最旧的未钉住条目（pinned 永不淘汰）。"""
        while len(self._order) > self.max_entries:
            for eid in list(self._order):
                if not self._entries[eid].pinned:
                    self._order.remove(eid)
                    entry = self._entries.pop(eid)
                    if not self.persist_dir:
                        self._archive[eid] = entry
                    break
            else:
                break

    def _dump_entry(self, entry: LedgerEntry) -> None:
        path = self.persist_dir / f"{entry.id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(entry.model_dump_json(), encoding="utf-8")
        os.replace(tmp, path)

    def _load_from_disk(self) -> None:
        for path in sorted(self.persist_dir.glob("*.json")):
            try:
                entry = LedgerEntry.model_validate_json(path.read_text(encoding="utf-8"))
                self._entries[entry.id] = entry
                self._order.append(entry.id)
            except Exception:
                continue


_PLACEHOLDER = (
    "[{kind} 已归档 id={eid}｜原文 {chars} 字符，来源 {source}]\n"
    '如需全文请调用 read_context("{eid}")；如需后续不再归档可 pin_context("{eid}")。'
)


class LedgerMiddleware(Middleware):
    """工具结果归档中间件。

    after_tool 阶段：超过 archive_threshold 的工具结果存入账本，
    窗口内替换为带 id 的占位引用。
    """

    def __init__(
        self,
        ledger: ContextLedger,
        archive_threshold: int = 2000,
        archive_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ):
        """
        Args:
            archive_threshold: 结果超过该字符数才归档
            archive_tools: 只归档这些工具（None = 全部）
            exclude_tools: 永不归档的工具（如 read_context 本身）
        """
        self.ledger = ledger
        self.archive_threshold = archive_threshold
        self.archive_tools = set(archive_tools) if archive_tools else None
        self.exclude_tools = set(exclude_tools or ()) | {"read_context", "pin_context", "list_context"}

    async def after_tool(self, ctx: MiddlewareContext, tool_message):
        name = getattr(tool_message, "name", "") or ""
        if name in self.exclude_tools:
            return tool_message
        if self.archive_tools is not None and name not in self.archive_tools:
            return tool_message

        content = tool_message.content or ""
        if len(content) <= self.archive_threshold:
            return tool_message
        # 错误结果不归档（模型需要立即看到错误详情）
        try:
            if isinstance(json.loads(content), dict) and json.loads(content).get("ok") is False:
                return tool_message
        except (json.JSONDecodeError, TypeError):
            pass

        entry_id = self.ledger.put(content, kind="tool_result", source=name)
        placeholder = _PLACEHOLDER.format(
            kind="工具结果", eid=entry_id, chars=len(content), source=name
        )
        tool_message.content = placeholder
        return tool_message


# ── 账本工具（供模型按需召回） ───────────────────────────────────────


def ledger_tools(ledger: ContextLedger) -> list[Tool]:
    """构建 read_context / pin_context / list_context 三个工具。"""

    def read_context(entry_id: str) -> str:
        """按 id 读取账本中归档的原文。

        :param entry_id: 占位引用中的 id，如 a3f2...
        """
        entry = ledger.get(entry_id)
        if entry is None:
            return f"未找到账本条目: {entry_id}"
        return entry.content

    def pin_context(entry_id: str) -> str:
        """钉住一个账本条目（永不淘汰）。通常无需主动调用。

        :param entry_id: 账本条目 id
        """
        ok = ledger.pin(entry_id, True)
        return f"已钉住 {entry_id}" if ok else f"未找到 {entry_id}"

    def list_context() -> str:
        """列出最近归档的账本条目（id / 来源 / 摘要）。"""
        entries = ledger.list_entries()
        if not entries:
            return "账本为空"
        lines = []
        for e in entries:
            preview = re.sub(r"\s+", " ", e.content[:80])
            lines.append(f"{e.id} [{e.kind}] {e.source}: {preview}...")
        return "\n".join(lines)

    return [
        Tool(name="read_context", description="读取上下文账本中归档的原文", parameters=ToolParameters(properties={"entry_id": {"type": "string"}}, required=["entry_id"]), handler=read_context),
        Tool(name="pin_context", description="钉住账本条目使其永不淘汰", parameters=ToolParameters(properties={"entry_id": {"type": "string"}}, required=["entry_id"]), handler=pin_context),
        Tool(name="list_context", description="列出最近归档的账本条目", parameters=ToolParameters(), handler=list_context),
    ]
