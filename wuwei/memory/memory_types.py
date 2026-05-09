from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryRecord:
    """一条长期记忆。"""

    id: str
    content: str
    memory_type: str = "fact"
    namespace: str = "default"
    importance: float = 0.5
    confidence: float = 0.8
    embedding: list[float] | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    access_count: int = 0


@dataclass
class KnowledgeChunk:
    """知识库中的一个片段。"""

    id: str
    text: str
    source: str
    namespace: str = "default"
    title: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
