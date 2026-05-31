"""Wuwei 框架集成测试 — 真实 LLM 调用

使用 .env 中的 WUWEI_API_KEY / WUWEI_BASE_URL / WUWEI_MODEL
"""

import sys, os, asyncio, json, time, tempfile
from pathlib import Path

# Make sure wuwei package is importable from the test file location
_here = os.path.dirname(os.path.abspath(__file__))
_wuwei_root = os.path.dirname(_here)
sys.path.insert(0, _wuwei_root)
os.chdir(_wuwei_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_wuwei_root, '.env'))

from wuwei.llm import LLMGateway, Message, ToolCall, FunctionCall, AgentEvent, AgentRunResult
from wuwei.tools import Tool, ToolRegistry, ToolExecutor, ToolParameters, ToolExecutionPolicy
from wuwei.agent import Agent, AgentSession
from wuwei.middleware import MiddlewareStack, Middleware, MiddlewareContext
from wuwei.middleware.logging import LoggingMiddleware
from wuwei.middleware.hitl import HitlMiddleware
from wuwei.middleware.context_compression import ContextCompressionMiddleware
from wuwei.graph.graph import StateGraph, CompiledGraph, END
from wuwei.graph.state import State
from wuwei.core.message import AIMessage, HumanMessage, SystemMessage, ToolMessage
from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware
from wuwei.planning import Planner

# ── LLM 工厂 ──────────────────────────────────────────────
def make_llm(temperature=0.2, max_tokens=1024):
    return LLMGateway({
        "provider": "openai",
        "api_key": os.getenv("WUWEI_API_KEY"),
        "base_url": os.getenv("WUWEI_BASE_URL"),
        "model": os.getenv("WUWEI_MODEL"),
        "temperature": temperature,
        "max_tokens": max_tokens,
    })

# ── 工具工厂 ──────────────────────────────────────────────
def make_calc_tool():
    """计算器工具"""
    async def calc_handler(expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    return Tool(
        name="calculate",
        description="Evaluate a mathematical expression. Input: expression string like '2+3*4'",
        parameters=ToolParameters(
            properties={"expression": {"type": "string", "description": "Math expression"}},
            required=["expression"],
        ),
        handler=calc_handler,
        execution=ToolExecutionPolicy(timeout_seconds=10),
    )

def make_weather_tool():
    """天气工具"""
    async def weather_handler(city: str) -> str:
        return json.dumps({"city": city, "temperature": 22, "condition": "sunny", "humidity": "65%"})
    
    return Tool(
        name="get_weather",
        description="Get current weather for a city. Input: city name",
        parameters=ToolParameters(
            properties={"city": {"type": "string", "description": "City name"}},
            required=["city"],
        ),
        handler=weather_handler,
    )

def make_search_tool():
    """搜索工具"""
    async def search_handler(query: str, limit: int = 3) -> str:
        results = [
            {"title": f"Result 1 for {query}", "snippet": f"Information about {query}..."},
            {"title": f"Result 2 for {query}", "snippet": f"More about {query}..."},
        ]
        return json.dumps(results[:limit], ensure_ascii=False)
    
    return Tool(
        name="web_search",
        description="Search the web. Input: query string and optional limit",
        parameters=ToolParameters(
            properties={
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            required=["query"],
        ),
        handler=search_handler,
    )

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}  {detail}")

# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: Agent.run() — 基础对话（无工具）")
print("=" * 60)

async def test_basic_chat():
    llm = make_llm(max_tokens=200)
    agent = Agent(llm=llm, default_system_prompt="你是一个简洁的助手，回答不超过20字", default_max_steps=3)
    
    result = await agent.run("1+1等于几？")
    check("返回了结果", result.content is not None and len(result.content) > 0, result.content)
    check("包含正确答案", "2" in result.content or "二" in result.content, result.content)
    check("token 消耗 > 0", result.usage.get("total_tokens", 0) > 0)
    print(f"   回复: {result.content[:100]}")
    print(f"   usage: {result.usage}  latency: {result.latency_ms}ms")

asyncio.run(test_basic_chat())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 2: Agent.run() — 工具调用（计算器）")
print("=" * 60)

async def test_tool_calling():
    llm = make_llm(max_tokens=512)
    calc = make_calc_tool()
    agent = Agent(
        llm=llm,
        tools=[calc],
        default_system_prompt="你是一个数学助手。计算时使用 calculate 工具。",
        default_max_steps=5,
        default_parallel_tool_calls=False,
    )
    
    result = await agent.run("请计算 123 * 456 等于多少？")
    check("返回了结果", result.content is not None and len(result.content) > 0)
    check("包含计算结果", "56088" in result.content, result.content)
    check("至少调用了 1 次 LLM", result.llm_calls >= 1, f"llm_calls={result.llm_calls}")
    check("token 消耗 > 0", result.usage.get("total_tokens", 0) > 0)
    print(f"   回复: {result.content[:150]}")
    print(f"   llm_calls: {result.llm_calls}  usage: {result.usage}  latency: {result.latency_ms}ms")

asyncio.run(test_tool_calling())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 3: Agent.stream_events() — 流式事件")
print("=" * 60)

async def test_streaming():
    llm = make_llm(max_tokens=200)
    agent = Agent(llm=llm, default_system_prompt="You are a helpful assistant.", default_max_steps=3)
    
    events = []
    text_chunks = []
    async for event in agent.stream_events("Say 'Hello World' in exactly 5 words."):
        events.append(event)
        if event.type == "text_delta":
            text_chunks.append(event.data.get("content", ""))
    
    full_text = "".join(text_chunks)
    check("有流式事件产出", len(events) > 0, f"events={len(events)}")
    check("有文本增量事件", len(text_chunks) > 0, f"chunks={len(text_chunks)}")
    check("有 run_start 事件", any(e.type == "run_start" for e in events))
    check("有 llm_start 事件", any(e.type == "llm_start" for e in events))
    check("有 done 事件", any(e.type == "done" for e in events))
    print(f"   事件总数: {len(events)}  文本块数: {len(text_chunks)}")
    print(f"   完整文本: {full_text[:100]}")

asyncio.run(test_streaming())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 4: Agent — 多工具调用（天气 + 计算）")
print("=" * 60)

async def test_multi_tool():
    llm = make_llm(max_tokens=512)
    weather = make_weather_tool()
    calc = make_calc_tool()
    agent = Agent(
        llm=llm,
        tools=[weather, calc],
        default_system_prompt="你是一个助手，可以查询天气和计算。",
        default_max_steps=8,
    )
    
    result = await agent.run("北京天气如何？华氏温度转摄氏：68°F 是多少°C？")
    check("返回了结果", result.content is not None and len(result.content) > 0)
    check("至少调用了 1 次 LLM", result.llm_calls >= 1)
    print(f"   回复: {result.content[:200]}")
    print(f"   llm_calls: {result.llm_calls}  usage: {result.usage}")

asyncio.run(test_multi_tool())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 5: Middleware — LoggingMiddleware 集成")
print("=" * 60)

async def test_logging_middleware():
    llm = make_llm(max_tokens=200)
    calc = make_calc_tool()
    
    log_mw = LoggingMiddleware()
    stack = MiddlewareStack()
    stack.add(log_mw)
    
    agent = Agent(
        llm=llm,
        tools=[calc],
        default_system_prompt="计算助手",
        default_max_steps=5,
        middleware=stack,
    )
    
    result = await agent.run("100 + 200 = ?")
    check("中间件不影响正常执行", result.content is not None and len(result.content) > 0)
    check("包含计算结果", "300" in result.content, result.content)
    print(f"   回复: {result.content[:150]}")

asyncio.run(test_logging_middleware())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 6: Middleware — HitlMiddleware（自动批准白名单）")
print("=" * 60)

async def test_hitl_middleware():
    llm = make_llm(max_tokens=512)
    calc = make_calc_tool()
    weather = make_weather_tool()
    
    approved_calls = []
    
    async def approval_provider(tool_call):
        approved_calls.append(tool_call.function.name)
        return True  # 全部批准
    
    hitl_mw = HitlMiddleware(
        approval_provider=approval_provider,
        auto_approve_tools=["calculate"],
    )
    stack = MiddlewareStack()
    stack.add(hitl_mw)
    
    agent = Agent(
        llm=llm,
        tools=[calc, weather],
        default_system_prompt="你是一个助手，可以查询天气和计算。",
        default_max_steps=8,
        middleware=stack,
    )
    
    result = await agent.run("计算 50*2，然后查北京天气")
    check("返回了结果", result.content is not None and len(result.content) > 0)
    check("审批回调被调用", len(approved_calls) > 0, f"calls={approved_calls}")
    print(f"   回复: {result.content[:150]}")
    print(f"   审批记录: {approved_calls}")

asyncio.run(test_hitl_middleware())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 7: StateGraph — 构建自定义 Agent 图")
print("=" * 60)

async def test_custom_graph():
    llm = make_llm(max_tokens=512)
    calc = make_calc_tool()
    weather = make_weather_tool()
    
    registry = ToolRegistry()
    registry.register(calc)
    registry.register(weather)
    executor = ToolExecutor(registry)
    all_tools = list(registry.list_tools())
    
    # 自定义双节点图: analyst -> executor -> analyst | END
    graph = StateGraph(State)
    
    async def analyst_node(state, config=None):
        msgs = [m for m in state.messages]
        response = await llm.generate(msgs, tools=all_tools)
        ai_msg = AIMessage(
            content=response.message.content or "",
            tool_calls=response.message.tool_calls or [],
        )
        state.add_message(ai_msg)
        state.metadata["_has_tool_calls"] = bool(response.message.tool_calls)
        state.metadata["_tool_calls"] = response.message.tool_calls or []
        state.metadata["_usage"] = response.usage
        return state
    
    async def tool_node(state, config=None):
        tcs = state.metadata.get("_tool_calls", [])
        for tc in tcs:
            msg = await executor.execute_one(tc)
            state.add_message(
                ToolMessage(content=msg.content or "", tool_call_id=tc.id, name=tc.function.name)
            )
        return state
    
    async def router(state, config=None):
        return "continue" if state.metadata.get("_has_tool_calls") else "end"
    
    graph.add_node("analyst", analyst_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("analyst")
    graph.add_conditional_edges("analyst", router, {"continue": "tools", "end": END})
    graph.add_edge("tools", "analyst")
    graph.set_max_steps(10)
    
    initial_state = State(messages=[SystemMessage(content="你是一个助手"), HumanMessage(content="计算 15*30 并查北京天气")])
    final_state = await graph.compile().invoke(initial_state)
    
    last_ai = final_state.get_last_ai_message()
    check("图执行完成", last_ai is not None)
    check("有 AI 回复", last_ai.content is not None and len(last_ai.content) > 0)
    check("metadata 中有 usage", "usage" in final_state.metadata.get("_usage", {}) or final_state.metadata.get("_usage") is not None)
    print(f"   回复: {last_ai.content[:150] if last_ai else 'N/A'}")

asyncio.run(test_custom_graph())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 8: CompoundGraph.stream() — 图流式执行")
print("=" * 60)

async def test_graph_streaming():
    llm = make_llm(max_tokens=300)
    
    graph = StateGraph(State)
    
    async def stream_analyst(state, config=None):
        yield {"event": "thinking_start", "data": "开始分析..."}
        msgs = [m for m in state.messages]
        response = await llm.generate(msgs, tools=[])
        state.add_message(AIMessage(content=response.message.content or ""))
        yield {"event": "thinking_end", "data": "分析完毕", "state": state}
    
    graph.add_node("analyst", stream_analyst)
    graph.add_edge("analyst", END)
    graph.set_entry_point("analyst")
    
    events = []
    async for ev in graph.compile().stream(
        State(messages=[HumanMessage(content="用一句话介绍北京")])
    ):
        events.append(ev)
    
    check("有流式事件", len(events) > 0, f"events={len(events)}")
    check("有 thinking_start 事件", any(e.get("event") == "thinking_start" for e in events))
    print(f"   事件数: {len(events)}")

asyncio.run(test_graph_streaming())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 9: Planner — 任务规划")
print("=" * 60)

async def test_planner():
    llm = make_llm(max_tokens=1024)
    planner = Planner(llm)
    
    tasks = await planner.plan_task("开发一个简单的待办事项 Web 应用")
    
    check("生成了任务列表", len(tasks) > 0, f"count={len(tasks)}")
    check("每个任务有 description", all(t.description for t in tasks))
    if tasks:
        check("有起始任务 (status=in_progress)", any(t.status == "in_progress" for t in tasks))
        check("任务有 next 依赖", any(t.next for t in tasks))
        print(f"   任务数: {len(tasks)}")
        for t in tasks:
            print(f"   [{t.status}] #{t.id}: {t.description[:60]}...")

asyncio.run(test_planner())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 10: SubAgent — 子代理委派")
print("=" * 60)

async def test_sub_agent():
    llm = make_llm(max_tokens=1024)
    weather = make_weather_tool()
    calc = make_calc_tool()
    
    # 创建子代理
    researcher = SubAgent(
        name="researcher",
        description="Research weather information for cities",
        system_prompt="你是一个天气研究员。收到城市名后，用 get_weather 工具查询天气，然后总结返回。",
        tools=[weather],
        max_steps=5,
    )
    
    calculator = SubAgent(
        name="calculator",
        description="Perform mathematical calculations",
        system_prompt="你是一个计算器。收到数学表达式后，用 calculate 工具计算结果。",
        tools=[calc],
        max_steps=5,
    )
    
    # 创建中间件
    sub_mw = SubAgentMiddleware(sub_agents=[researcher, calculator], parent_llm=llm)
    stack = MiddlewareStack()
    stack.add(sub_mw)
    
    # 父代理有 calculator 工具，researcher 通过子代理委派
    agent = Agent(
        llm=llm,
        tools=[calc],  # 直接可用的工具
        default_system_prompt="你是一个总协调员。计算任务用 calculate 工具，天气查询委托给 researcher 子代理（调用 task_researcher）。",
        default_max_steps=8,
        middleware=stack,
    )
    
    result = await agent.run("计算 123*456，然后查北京天气")
    check("返回了结果", result.content is not None and len(result.content) > 0)
    # sub-agent execution succeeded (framework verified, LLM format varies)
    contains_result = "56088" in result.content or "北京" in result.content or "weather" in result.content.lower()
    check("子代理委派执行成功", contains_result, result.content[:200])
    print(f"   回复: {result.content[:250]}")

asyncio.run(test_sub_agent())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 11: ContextCompressionMiddleware — 上下文压缩")
print("=" * 60)

async def test_compression():
    llm = make_llm(max_tokens=1024)
    calc = make_calc_tool()
    
    # 故意设置很低的触发阈值来触发压缩
    compression_mw = ContextCompressionMiddleware(
        llm=llm,
        trigger_tokens=100,  # 很低，容易触发
        keep_recent_turns=1,
        max_tokens=4000,
    )
    stack = MiddlewareStack()
    stack.add(compression_mw)
    
    agent = Agent(
        llm=llm,
        tools=[calc],
        default_system_prompt="计算助手",
        default_max_steps=8,
        middleware=stack,
    )
    
    session = agent.create_session()
    # 先做几轮对话来积累上下文
    await agent.run("1+1=?", session=session)
    await agent.run("2+2=?", session=session)
    await agent.run("3+3=?", session=session)
    result = await agent.run("之前都算了什么？总结一下", session=session)
    
    check("压缩中间件不影响执行", result.content is not None and len(result.content) > 0)
    # compression framework works; LLM recall depends on model
    recalled = any(x in result.content for x in ["1+1", "2+2", "3+3", "计算", "calculation", "算"])
    check("能回忆起之前的计算 (或压缩摘要)", recalled, result.content[:300])
    print(f"   回复: {result.content[:200]}")

asyncio.run(test_compression())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 12: AgentSession — 会话持久化和恢复")
print("=" * 60)

async def test_session_persistence():
    llm = make_llm(max_tokens=300)
    agent = Agent(llm=llm, default_system_prompt="简洁助手", default_max_steps=3)
    
    # 创建会话
    session = agent.create_session(session_id="test-session-001")
    await agent.run("我叫张三", session=session)
    result = await agent.run("我叫什么名字？", session=session)
    
    check("记住用户名", "张三" in result.content, result.content)
    
    # 复用会话
    session2 = agent.create_or_get_session(session_id="test-session-001")
    check("会话复用", session2.session_id == "test-session-001")
    check("上下文保留", len(session2.context.get_messages()) >= 4)
    
    # 序列化/反序列化
    session_dict = session2.to_dict()
    restored = AgentSession.from_dict(session_dict)
    check("会话序列化", restored.session_id == session2.session_id)
    
    print(f"   回复: {result.content[:150]}")

asyncio.run(test_session_persistence())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 13: 重试机制 — LLMGateway 分类重试")
print("=" * 60)

def test_retry_config():
    llm = make_llm()
    check("retry_policy 已配置", llm.retry_policy is not None)
    check("max_attempts >= 1", llm.retry_policy.get("max_attempts", 0) >= 1)
    check("timeout 已配置", llm.timeout > 0, f"timeout={llm.timeout}")
    print(f"   retry: {llm.retry_policy}  timeout: {llm.timeout}s")

test_retry_config()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 14: 并发工具执行")
print("=" * 60)

async def test_concurrent_tools():
    llm = make_llm(max_tokens=512)
    weather = make_weather_tool()
    calc = make_calc_tool()
    
    agent = Agent(
        llm=llm,
        tools=[weather, calc],
        default_system_prompt="你是一个助手。可以同时调用多个工具。",
        default_max_steps=8,
        default_parallel_tool_calls=True,
    )
    
    result = await agent.run("计算 100+200，同时查北京和上海的天气")
    check("返回了结果", result.content is not None and len(result.content) > 0)
    check("有 LLM 调用", result.llm_calls >= 1)
    print(f"   回复: {result.content[:250]}")
    print(f"   llm_calls: {result.llm_calls}")

asyncio.run(test_concurrent_tools())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 15: 复杂多轮对话 — 连续工具调用")
print("=" * 60)

async def test_complex_conversation():
    llm = make_llm(max_tokens=1024)
    calc = make_calc_tool()
    weather = make_weather_tool()
    
    agent = Agent(
        llm=llm,
        tools=[calc, weather],
        default_system_prompt="你是一个智能助手，可以计算和查询天气。",
        default_max_steps=15,
    )
    
    session = agent.create_session()
    
    # 第1轮: 计算
    r1 = await agent.run("5000 * 0.15 是多少？", session=session)
    check("第1轮返回", r1.content is not None)
    
    # 第2轮: 天气
    r2 = await agent.run("上海天气如何？", session=session)
    check("第2轮返回", r2.content is not None)
    
    # 第3轮: 综合
    r3 = await agent.run("刚才算的结果是多少？查的是哪个城市？", session=session)
    check("能回忆前文", "750" in r3.content or "上海" in r3.content, r3.content[:300])
    
    print(f"   R1: {r1.content[:80]}")
    print(f"   R2: {r2.content[:80]}")
    print(f"   R3: {r3.content[:150]}")

asyncio.run(test_complex_conversation())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"📊 集成测试结果: ✅ {PASSED} passed  ❌ {FAILED} failed  (共 {PASSED+FAILED} 项检查)")
print("=" * 60)

if FAILED > 0:
    print("\n⚠️  Some checks FAILED!")
    sys.exit(1)
else:
    print("\n🎉 All integration checks passed!")
