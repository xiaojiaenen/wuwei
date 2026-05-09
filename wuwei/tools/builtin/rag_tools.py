from __future__ import annotations

from pathlib import Path

from wuwei.memory.knowledge_store import KnowledgeStore
from wuwei.tools.registry import ToolRegistry


def register_rag_tools(registry: ToolRegistry, *, knowledge_store: KnowledgeStore) -> None:
    @registry.tool(
        name="ingest_document",
        description="将文档导入知识库。支持 txt/md 文件。导入后可用于 RAG 检索。",
    )
    async def ingest_document(file_path: str) -> dict:
        path = Path(file_path).resolve()
        if not path.exists():
            return {"ok": False, "error": f"文件不存在: {file_path}"}
        if path.suffix not in (".txt", ".md", ".markdown"):
            return {"ok": False, "error": "仅支持 .txt / .md 文件"}

        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = await knowledge_store.ingest(text, source=str(path))
        return {"ok": True, "chunks": len(chunks), "source": str(path)}

    @registry.tool(
        name="search_knowledge",
        description="从知识库中检索相关文档片段。",
    )
    async def search_knowledge(query: str, limit: int = 4) -> dict:
        chunks = await knowledge_store.search(query, limit=limit)
        results = []
        for c in chunks:
            results.append({"source": c.source, "text": c.text[:500]})
        return {"ok": True, "results": results}
