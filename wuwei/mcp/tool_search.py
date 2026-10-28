"""tool_search — MCP 工具治理的按需检索工具

当接入大量 MCP 服务器时，把全部工具注册进上下文会吃掉大量 token
（例如 15 个 server 占 40% 上下文）。本模块提供轻量方案：

1. ``ToolSearchIndex``：对所有发现的 MCP 工具建 (name, description) 的
   轻量语义索引（子串匹配 + 词重叠打分，不引入 embedding 依赖）。
2. ``build_tool_search_tool``：生成一个 ``tool_search`` 工具，模型输入
   query + limit，按相关性得分返回工具名/描述/来源 server，并把选中的
   工具通过 register_fn 注册进 ToolRegistry。

打分规则（纯词法，无外部依赖）：
- query 整体是工具名的子串：+5
- query 整体是描述的子串：+2
- query 每个词命中工具名 token：+2/词；命中描述 token：+1/词
- 得分为 0 的条目不返回
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from wuwei.tools.tool import Tool, ToolExecutionPolicy, ToolParameters

__all__ = ["ToolSearchIndex", "build_tool_search_tool"]

_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


def _tokenize(text: str) -> list[str]:
    """小写分词：snake_case / camelCase / 标点均拆开，过滤单字符噪声"""
    text = _CAMEL_RE.sub(r"\1 \2", text)
    parts = _TOKEN_SPLIT_RE.split(text.lower())
    return [p for p in parts if len(p) > 1]


@dataclass
class ToolSearchIndex:
    """MCP 工具的轻量语义索引（子串 + 词重叠打分）"""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, tool: Tool, server: str) -> None:
        """加入一个工具条目"""
        self.entries.append(
            {
                "name": tool.name,
                "description": tool.description,
                "server": server,
                "tool": tool,
            }
        )

    def add_tools(self, tools: list[Tool], server: str) -> None:
        """加入某个服务器的全部工具"""
        for tool in tools:
            self.add(tool, server)

    def __len__(self) -> int:
        return len(self.entries)

    def get_tool(self, name: str) -> Tool | None:
        """按工具名取回 Tool 对象"""
        for entry in self.entries:
            if entry["name"] == name:
                return entry["tool"]
        return None

    def score(self, query: str, entry: dict[str, Any]) -> float:
        """对单条目打分（词法匹配，见模块 docstring）"""
        q = query.lower().strip()
        if not q:
            return 0.0
        name = entry["name"].lower()
        desc = (entry["description"] or "").lower()

        score = 0.0
        if q in name:
            score += 5.0
        if q in desc:
            score += 2.0

        q_tokens = _tokenize(q)
        if q_tokens:
            name_tokens = set(_tokenize(name))
            desc_tokens = set(_tokenize(desc))
            for tok in q_tokens:
                if tok in name_tokens:
                    score += 2.0
                if tok in desc_tokens:
                    score += 1.0
        return score

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """按相关性返回匹配条目（name/description/server/score），不含 Tool 对象"""
        scored = []
        for entry in self.entries:
            s = self.score(query, entry)
            if s > 0:
                scored.append((s, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "name": entry["name"],
                "description": entry["description"],
                "server": entry["server"],
                "score": round(s, 2),
            }
            for s, entry in scored[: max(1, limit)]
        ]


def build_tool_search_tool(
    index: ToolSearchIndex,
    register_fn: Callable[[Tool], bool],
) -> Tool:
    """生成 ``tool_search`` 工具

    Args:
        index: 覆盖所有已发现 MCP 工具的索引
        register_fn: 注册回调，把选中的 Tool 注册进 ToolRegistry；
            返回 True 表示新注册成功，False 表示已存在/注册失败。

    Returns:
        可注册进 ToolRegistry 的 Tool 对象（name="tool_search"）。
    """
    description = (
        "按需检索并注册 MCP 工具。当前只注入了 MCP 服务器概览，具体工具未加载；"
        "先用 query 检索（如 'read file'、'search web'），返回按相关性排序的 "
        "工具名/描述/来源服务器，匹配到的工具会被自动注册，随后即可直接调用。"
    )

    async def tool_search(query: str, limit: int = 5) -> dict[str, Any]:
        matches = index.search(query, limit=limit)
        registered: list[str] = []
        for match in matches:
            tool = index.get_tool(match["name"])
            if tool is not None and register_fn(tool):
                registered.append(match["name"])
        return {
            "query": query,
            "matches": matches,
            "registered": registered,
            "hint": (
                "matched tools are now registered and callable"
                if registered
                else "no tool registered; try a different query"
            ),
        }

    return Tool(
        name="tool_search",
        description=description,
        parameters=ToolParameters(
            type="object",
            properties={
                "query": {
                    "type": "string",
                    "description": "要检索的工具能力关键词",
                },
                "limit": {
                    "type": "number",
                    "description": "最多返回的匹配数量，默认 5",
                },
            },
            required=["query"],
        ),
        handler=tool_search,
        execution=ToolExecutionPolicy(side_effect=False),
    )
