"""状态图构建器"""

from typing import Any, Callable, Optional
from wuwei.graph.state import State


class StateGraph:
    """状态图构建器

    轻量版 LangGraph StateGraph，支持：
    - 节点：处理函数
    - 边：节点间连接
    - 条件边：根据状态动态路由

    示例：
        graph = StateGraph(State)
        graph.add_node("llm", call_llm)
        graph.add_node("tool", execute_tool)
        graph.add_edge("llm", "tool")
        graph.add_conditional_edges("tool", should_continue, {"llm": "llm", "end": END})
        graph.set_entry_point("llm")
        app = graph.compile()
    """

    def __init__(self, state_type: type = State):
        self.state_type = state_type
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | tuple[Callable, dict]] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, func: Callable) -> "StateGraph":
        """添加节点"""
        self.nodes[name] = func
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        """添加边"""
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
            condition: 条件函数，接收 State 返回 key
            targets: key 到目标节点的映射
        """
        self.edges[source] = (condition, targets)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        """设置入口点"""
        self.entry_point = name
        return self

    def compile(self) -> "CompiledGraph":
        """编译图为可执行图"""
        if not self.entry_point:
            raise ValueError("未设置入口点，请调用 set_entry_point()")
        return CompiledGraph(self)


# 特殊常量
END = "__end__"


class CompiledGraph:
    """编译后的可执行图"""

    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.checkpointer = None

    def set_checkpointer(self, checkpointer) -> "CompiledGraph":
        """设置检查点"""
        self.checkpointer = checkpointer
        return self

    async def invoke(
        self,
        input_state: State = None,
        config: dict = None,
    ) -> State:
        """执行图

        Args:
            input_state: 初始状态
            config: 配置

        Returns:
            最终状态
        """
        state = input_state or self.graph.state_type()
        current_node = self.graph.entry_point

        while current_node:
            if current_node == END:
                break

            # 执行节点
            node_func = self.graph.nodes[current_node]
            state = await node_func(state, config)

            # 保存检查点
            if self.checkpointer:
                await self.checkpointer.save(state)

            # 确定下一个节点
            edge = self.graph.edges.get(current_node)
            if edge is None:
                break
            elif isinstance(edge, str):
                current_node = edge
            else:
                condition, targets = edge
                # 支持 condition(state) 或 condition(state, config) 两种签名
                import inspect
                sig = inspect.signature(condition)
                if len(sig.parameters) >= 2:
                    next_key = await condition(state, config)
                else:
                    next_key = await condition(state)
                current_node = targets.get(next_key)

        return state

    async def ainvoke(
        self,
        input_state: State = None,
        config: dict = None,
    ) -> State:
        """异步执行图（别名）"""
        return await self.invoke(input_state, config)

    def stream(self, input_state: State = None, config: dict = None):
        """流式执行图（TODO: 实现）"""
        raise NotImplementedError("流式执行暂未实现")
