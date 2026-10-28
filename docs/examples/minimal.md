# 最小在线示例


设置 `OPENAI_API_KEY`，然后运行：

```python
import asyncio
from wuwei import Agent, LLMGateway, ToolRegistry

registry = ToolRegistry()

@registry.tool()
def greet(name: str) -> str:
    """按姓名生成问候。"""
    return f"你好，{name}"

async def main():
    async with LLMGateway.from_env(provider="openai") as llm:
        agent = Agent(llm=llm, tools=registry, load_builtins="minimal")
        session = agent.create_session()
        result = await agent.run("向小明问好", session=session)
        print(result.status, result.content)
        async for event in agent.stream_events("改成更简短的问候", session=session):
            if event.type == "text_delta":
                print(event.data.get("content", ""), end="", flush=True)

asyncio.run(main())
```

`minimal` 只加载 calc/time/json/text。文件、网络、Git、npm 和脚本执行需按名选择；`load_builtins="all"` 会扩大可用能力，不应作为生产默认值。权限中间件、人工审批与 OS 沙箱是不同边界。
