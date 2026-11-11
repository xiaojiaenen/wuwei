import asyncio, collections
from wuwei import Agent, ToolRegistry
from wuwei.memory import InMemoryMemoryStore
from wuwei.graph.checkpoint import MemoryCheckpointer
from wuwei.llm.types import FunctionCall, LLMResponse, Message, ToolCall

class FakeLLM:
    model="fake"; provider="fake"
    def __init__(self, steps): self.steps=steps; self.calls=0
    async def generate(self, messages, tools=None, stream=False, **kw):
        self.calls+=1
        if self.calls<=self.steps:
            return LLMResponse(message=Message(role="assistant",content="",tool_calls=[ToolCall(id=f"c{self.calls}",type="function",function=FunctionCall(name="echo",arguments={"text":"hi"}))]),finish_reason="tool_calls",usage={},model="fake")
        return LLMResponse(message=Message(role="assistant",content="done"),finish_reason="stop",usage={},model="fake")

class Tracing(MemoryCheckpointer):
    def __init__(self): super().__init__(); self.seq=[]
    async def save(self, state, checkpoint_id=None, metadata=None):
        md = metadata or {}
        self.seq.append((md.get("step"), md.get("kind"), [c.get("id") for c in md.get("pending_tool_calls",[])]))
        return await super().save(state, checkpoint_id=checkpoint_id, metadata=metadata)

async def main():
    reg=ToolRegistry()
    @reg.tool()
    def echo(text: str) -> str:
        """echo"""
        return text
    cp=Tracing()
    agent=Agent(llm=FakeLLM(3), tools=reg, checkpointer=cp, load_builtins=False)
    s=agent.create_session(); s.max_steps=8
    await agent.run("go", session=s)
    for row in cp.seq: print(row)

asyncio.run(main())
