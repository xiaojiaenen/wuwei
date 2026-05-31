"""状态图构建器

借鉴 LangGraph 的 StateGraph，支持：
- 节点：处理函数
- 边：节点间连接
- 条件边：根据状态动态路由
- 内置 Agent 循环：自动构建 LLM → Tool 的标准 agent 图
- 流式执行：基于 async generator 的节点级事件流
"""

import asyncio
import inspect
import time
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional

from wuwei.graph.state import State


class StateGraph:
    """状态图构建器

    轻量版 LangGraph StateGraph，支持：
    - 节点：处理函数
    - 边：节点间连接
    - 条件边：根据状态动态路由
    - 内置 Agent 循环

    示例：
        graph = StateGraph(State)
        graph.add_node("llm", call_llm)
        graph.add_node("tool", execute_tool)
        graph.add_conditional_edges("llm", should_continue, {"tool": "tool", "end": END})
        graph.add_edge("tool", "llm")
        graph.set_entry_point("llm")
        app = graph.compile()
        result = await app.invoke(initial_state)
    """

    def __init__(self, state_type: type = State):
        self.state_type = state_type
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | tuple[Callable, dict]] = {}
        self.entry_point: Optional[str] = None
        self._max_steps: int = 100

    def add_node(self, name: str, func: Callable) -> "StateGraph":
        """添加节点

        Args:
            name: 节点名称
            func: 节点处理函数 async (state, config) -> State
        """
        self.nodes[name] = func
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        """添加静态边

        Args:
            source: 源节点
            target: 目标节点
        """
        self.edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: Callable,
        targets: dict[str, str],
    ) -> "StateGraph":
        """添加条件边

        Args:
            source: 源节点
            condition: 条件函数 async (state, config) -> str，返回目标节点名
            targets: 条件返回值到目标节点的映射
        """
        self.edges[source] = (condition, targets)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        """设置入口节点"""
        self.entry_point = name
        return self

    def set_max_steps(self, max_steps: int) -> "StateGraph":
        """设置最大执行步数"""
        self._max_steps = max_steps
        return self

    def compile(self) -> "CompiledGraph":
        """编译图为可执行图"""
        if not self.entry_point:
            raise ValueError("未设置入口点，请调用 set_entry_point()")
        return CompiledGraph(self, max_steps=self._max_steps)


# 特殊常量
END = "__end__"


class CompiledGraph:
    """编译后的可执行图

    支持 invoke（同步风格）和 stream（流式事件）两种执行模式。
    """

    def __init__(self, graph: StateGraph, max_steps: int = 100):
        self.graph = graph
        self.checkpointer = None
        self.max_steps = max_steps

    def set_checkpointer(self, checkpointer) -> "CompiledGraph":
        """设置检查点"""
        self.checkpointer = checkpointer
        return self

    async def _resolve_next_node(
        self,
        current_node: str,
        state: State,
        config: dict | None,
    ) -> str | None:
        """解析下一个节点"""
        edge = self.graph.edges.get(current_node)
        if edge is None:
            return None
        if isinstance(edge, str):
            return edge
        # 条件边
        condition, targets = edge
        sig = inspect.signature(condition)
        if len(sig.parameters) >= 2:
            next_key = await condition(state, config)
        else:
            next_key = await condition(state)
        return targets.get(next_key)

    async def invoke(
        self,
        input_state: State | None = None,
        config: dict | None = None,
    ) -> State:
        """执行图（非流式）

        Args:
            input_state: 初始状态
            config: 运行时配置

        Returns:
            最终状态
        """
        state = input_state or self.graph.state_type()
        current_node = self.graph.entry_point
        step = 0

        while current_node and step < self.max_steps:
            if current_node == END:
                break

            node_func = self.graph.nodes[current_node]
            state = await node_func(state, config)
            state.step = step

            if self.checkpointer:
                await self.checkpointer.save(
                    state,
                    metadata={"node": current_node, "step": step},
                )

            next_node = await self._resolve_next_node(current_node, state, config)
            current_node = next_node
            step += 1

        return state

    async def ainvoke(
        self,
        input_state: State | None = None,
        config: dict | None = None,
    ) -> State:
        """异步执行图（invoke 的别名）"""
        return await self.invoke(input_state, config)

    async def stream(
        self,
        input_state: State | None = None,
        config: dict | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式执行图

        每执行一个节点，yield 一个包含节点名和状态的事件。

        Yields:
            {"node": str, "state": State, "step": int, "event": str}
        """
        state = input_state or self.graph.state_type()
        current_node = self.graph.entry_point
        step = 0

        while current_node and step < self.max_steps:
            if current_node == END:
                yield {
                    "node": END,
                    "state": state,
                    "step": step,
                    "event": "graph_end",
                }
                break

            node_func = self.graph.nodes[current_node]
            start_time = time.monotonic()

            # 如果节点函数是 async generator（用于细粒度事件）
            result = node_func(state, config)
            if inspect.isasyncgen(result):
                async for node_event in result:
                    node_event["node"] = current_node
                    node_event["step"] = step
                    yield node_event
                # async generator 的最后一项应该是 state
                # 从最后一个 yield 获取 state
                state = node_event.get("state", state)
            else:
                state = await result
                yield {
                    "node": current_node,
                    "state": state,
                    "step": step,
                    "event": "node_complete",
                    "latency_ms": int((time.monotonic() - start_time) * 1000),
                }

            state.step = step

            if self.checkpointer:
                await self.checkpointer.save(
                    state,
                    metadata={"node": current_node, "step": step},
                )

            next_node = await self._resolve_next_node(current_node, state, config)
            current_node = next_node
            step += 1
