"""状态图模块测试"""

import pytest
from wuwei.graph import State, StateGraph, CompiledGraph, MemoryCheckpointer
from wuwei.graph.graph import END
from wuwei.core.message import HumanMessage, AIMessage


class TestState:
    """State 测试"""

    def test_create_state(self):
        """测试创建状态"""
        state = State()
        assert state.messages == []
        assert state.metadata == {}
        assert state.step == 0

    def test_add_message(self):
        """测试添加消息"""
        state = State()
        state.add_message(HumanMessage(content="hello"))
        assert len(state.messages) == 1
        assert state.messages[0].content == "hello"

    def test_get_last_user_message(self):
        """测试获取最后用户消息"""
        state = State()
        state.add_message(HumanMessage(content="hello"))
        state.add_message(AIMessage(content="world"))
        state.add_message(HumanMessage(content="bye"))

        last_user = state.get_last_user_message()
        assert last_user is not None
        assert last_user.content == "bye"

    def test_to_dict(self):
        """测试转换为字典"""
        state = State()
        state.add_message(HumanMessage(content="hello"))
        state.metadata["key"] = "value"

        d = state.to_dict()
        assert "messages" in d
        assert "metadata" in d
        assert d["metadata"]["key"] == "value"

    def test_from_dict(self):
        """测试从字典创建"""
        d = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "metadata": {"key": "value"},
            "step": 5,
        }
        state = State.from_dict(d)
        assert len(state.messages) == 2
        assert state.metadata["key"] == "value"
        assert state.step == 5


class TestStateGraph:
    """StateGraph 测试"""

    def test_add_node(self):
        """测试添加节点"""
        graph = StateGraph(State)
        graph.add_node("llm", lambda s, c: s)
        assert "llm" in graph.nodes

    def test_add_edge(self):
        """测试添加边"""
        graph = StateGraph(State)
        graph.add_node("a", lambda s, c: s)
        graph.add_node("b", lambda s, c: s)
        graph.add_edge("a", "b")
        assert graph.edges["a"] == "b"

    def test_set_entry_point(self):
        """测试设置入口点"""
        graph = StateGraph(State)
        graph.add_node("start", lambda s, c: s)
        graph.set_entry_point("start")
        assert graph.entry_point == "start"

    def test_compile_without_entry_point(self):
        """测试没有入口点时编译失败"""
        graph = StateGraph(State)
        graph.add_node("start", lambda s, c: s)
        with pytest.raises(ValueError):
            graph.compile()


class TestCompiledGraph:
    """CompiledGraph 测试"""

    @pytest.mark.asyncio
    async def test_simple_graph(self):
        """测试简单图执行"""

        async def node_a(state: State, config: dict) -> State:
            state.add_message(HumanMessage(content="a"))
            return state

        async def node_b(state: State, config: dict) -> State:
            state.add_message(AIMessage(content="b"))
            return state

        graph = StateGraph(State)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")

        app = graph.compile()
        state = await app.invoke()

        assert len(state.messages) == 2
        assert state.messages[0].content == "a"
        assert state.messages[1].content == "b"

    @pytest.mark.asyncio
    async def test_conditional_edges(self):
        """测试条件边"""

        async def router(state: State, config: dict) -> str:
            # 检查是否有 AI 消息
            has_ai = any(m.role == "assistant" for m in state.messages)
            return "end" if has_ai else "b"

        async def node_a(state: State, config: dict) -> State:
            state.add_message(HumanMessage(content="a"))
            return state

        async def node_b(state: State, config: dict) -> State:
            state.add_message(AIMessage(content="b"))
            return state

        graph = StateGraph(State)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_conditional_edges("a", router, {"b": "b", "end": END})
        graph.set_entry_point("a")

        app = graph.compile()
        state = await app.invoke()

        # 执行 a -> b，共 2 条消息
        assert len(state.messages) == 2
        assert state.messages[0].content == "a"
        assert state.messages[1].content == "b"


class TestMemoryCheckpointer:
    """MemoryCheckpointer 测试"""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """测试保存和加载"""
        checkpointer = MemoryCheckpointer()
        state = State()
        state.add_message(HumanMessage(content="hello"))

        checkpoint_id = await checkpointer.save(state)
        loaded = await checkpointer.load(checkpoint_id)

        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "hello"

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """测试列出检查点"""
        checkpointer = MemoryCheckpointer()

        for i in range(5):
            state = State()
            state.add_message(HumanMessage(content=f"msg {i}"))
            await checkpointer.save(state)

        checkpoints = await checkpointer.list_checkpoints(limit=3)
        assert len(checkpoints) == 3
