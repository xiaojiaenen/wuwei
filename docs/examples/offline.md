# 不用密钥的可执行示例

以下各段独立运行；使用本地 fake 模型验证接线，不请求供应商服务。

## 类型化工具与参数验证

```python
import asyncio
from wuwei import ToolRegistry

registry = ToolRegistry()

@registry.tool()
def join_names(names: list[str]) -> str:
    return ", ".join(names)

async def main():
    tool = registry.get("join_names")
    assert tool.parameters.properties["names"]["type"] == "array"
    assert await tool.invoke({"names": ["甲", "乙"]}) == "甲, 乙"
    try:
        await tool.invoke({"names": "不是列表"})
    except ValueError:
        pass
    else:
        raise AssertionError("无效参数不能执行")

asyncio.run(main())
```

## 普通与流式使用同一会话

```python
import asyncio
from wuwei import Agent, LLMResponse, LLMResponseChunk

class LocalModel:
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        content = "收到：" + messages[-1].content
        if not stream:
            return LLMResponse(content=content, finish_reason="stop")
        async def chunks():
            yield LLMResponseChunk(content=content, finish_reason="stop")
        return chunks()

async def main():
    agent = Agent(llm=LocalModel(), load_builtins=False)
    session = agent.create_session()
    result = await agent.run("记住项目", session=session)
    assert result.status == "completed"
    events = [event async for event in agent.stream_events("继续", session=session)]
    assert events[-1].type == "run_end"
    assert len([m for m in session.context.get_messages() if m.role == "user"]) == 2

asyncio.run(main())
```

## 图节点更新与 reducer

```python
import asyncio
from wuwei.graph import StateGraph, END
from wuwei.graph.reducers import add

async def first(state, config):
    return {"count": 1}

async def second(state, config):
    return {"count": 2}

async def main():
    graph = StateGraph()
    graph.add_node("first", first, writes={"count": add})
    graph.add_node("second", second, writes={"count": add})
    graph.set_entry_point("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    app = graph.compile()
    state = await app.invoke()
    streamed = [event async for event in app.stream()]
    assert state.metadata["count"] == 3
    assert streamed[-1]["state"].metadata["count"] == 3

asyncio.run(main())
```

## 结构化输出的返回契约

```python
import asyncio
from pydantic import BaseModel
from wuwei import Agent, LLMResponse

class Summary(BaseModel):
    title: str

class LocalModel:
    async def generate(self, messages, **kwargs):
        return LLMResponse(content='{"title":"本地示例"}', finish_reason="stop")

async def main():
    agent = Agent(llm=LocalModel(), load_builtins=False)
    output, result = await agent.run_structured("输出标题", Summary)
    assert result.status == "completed"
    assert output.title == "本地示例"

asyncio.run(main())
```

真实审批挂起时 output 为 None；不要在判断 completed 之前访问它的字段。
