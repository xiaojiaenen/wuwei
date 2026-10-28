"""2026-10-27 当前工作树缺陷复现；只输出观察，不修复业务代码。
从仓库根目录运行：PYTHONPATH=. .venv/bin/python docs/audits/2026-10-27/reproduce.py
所有模型与工具结果均为本地 fake；文件写入仅发生在临时目录。
Responses 探针在 SDK 参数校验阶段失败，不发出 HTTP 请求。
"""
import asyncio, json, runpy, tempfile, inspect
from pathlib import Path
from types import SimpleNamespace as N
from wuwei.agent import Agent
from wuwei.core.message import ToolCall, FunctionCall, ToolMessage
from wuwei.middleware.stack import MiddlewareStack
from wuwei.middleware.hitl import HitlMiddleware
from wuwei.middleware.base import MiddlewareContext
from wuwei.middleware.storage import StorageMiddleware
from wuwei.graph.state import State
from wuwei.graph.checkpoint import MemoryCheckpointer, SQLiteCheckpointer
from wuwei.security.permissions import PermissionMiddleware, PermissionConfig
from wuwei.tools import Tool, ToolParameters
from wuwei.llm import LLMGateway
from wuwei.llm.adapters.anthropic import AnthropicAdapter
from wuwei.planning import PlanExecutor, Task, TaskList
from wuwei.middleware.ledger import ContextLedger, ledger_tools
from wuwei.serve.ag_ui import ag_ui_events

from wuwei.llm.types import LLMResponse, LLMResponseChunk
from wuwei.core.message import BaseMessage

class FakeGateway:
    def __init__(self, responses):
        self.adapter = N(responses=list(responses), calls=[])
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        self.adapter.calls.append({"messages": messages, "tools": tools or [], "stream": stream})
        response = self.adapter.responses.pop(0)
        if not stream:
            return response
        async def chunks():
            yield LLMResponseChunk(content=response.message.content,
                finish_reason=response.finish_reason,
                tool_calls_complete=response.message.tool_calls or None,
                usage=response.usage)
        return chunks()

def resp(content="", tool_calls=None):
    return LLMResponse(message=BaseMessage(role="assistant", content=content,
        tool_calls=tool_calls or []), finish_reason="tool_calls" if tool_calls else "stop",
        usage={"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},
        model="fake", latency_ms=0)

def tc(name,id='c1',**args):return ToolCall(id=id,function=FunctionCall(name=name,arguments=args))
def tool(name, handler):return Tool(name=name,description=name,parameters=ToolParameters(),handler=handler)
def agent(responses,tools=(),mws=(),cp=None,max_steps=10):
    stack=MiddlewareStack()
    for mw in mws:stack.add(mw)
    return Agent(llm=FakeGateway(responses),tools=list(tools),middleware=stack,load_builtins=False,checkpointer=cp,default_max_steps=max_steps)
def out(name,**result): print(json.dumps(dict(probe=name,**result),ensure_ascii=False,default=str))

async def main():
    # Multiple before_tool middleware, with rejection as terminal result
    s=MiddlewareStack(); s.add(HitlMiddleware(auto_reject_tools=['deploy'],approval_provider=lambda x:None)); s.add(PermissionMiddleware(PermissionConfig()))
    try:await s.execute_before_tool(MiddlewareContext(state=State()),tc('deploy'))
    except Exception as e:out('middleware_composition',error=repr(e))
    # All calls from one assistant message must get results across interrupt
    calls=[]
    async def a():calls.append('a');return 'a'
    async def b():calls.append('b');return 'b'
    ag=agent([resp(tool_calls=[tc('a'),tc('b','c2')]),resp('done'),resp('done again')],[tool('a',a),tool('b',b)],[HitlMiddleware(mode='interrupt')],MemoryCheckpointer())
    r=await ag.run('go'); state=await ag.checkpointer.load(r.interrupt['checkpoint_id']); r2=await ag.resume(r.interrupt)
    out('interrupt_batch',pending=[x['function']['name'] for x in state.metadata['pending_tool_calls']],executed=calls,tool_results=[m.tool_call_id for m in ag._default_session.context.get_messages() if m.role=='tool'])
    await ag.resume(r.interrupt)
    out('repeat_resume',executed=calls)
    # New step drops taint and per-run quotas
    calls=[]
    async def http_get(url):calls.append('http');return 'untrusted content'
    async def write_text_file(path):calls.append('write');return 'written'
    ag=agent([resp(tool_calls=[tc('http_get',url='https://example.com')]),resp(tool_calls=[tc('write_text_file','c2',path='out/a.txt')]),resp('done')],[tool('http_get',http_get),tool('write_text_file',write_text_file)],[PermissionMiddleware(PermissionConfig(network_allow=['example.com'],filesystem_write=['out/**']))])
    await ag.run('go');out('taint_across_steps',executed=calls)
    out('permission_paths',traversal_allowed=PermissionMiddleware._glob_match(['out/**'],'out/../../secret.txt'),sibling_allowed=PermissionMiddleware._glob_match(['out/**'],'outside/secret.txt'))
    # Streaming and run defaults do not share session
    ag=agent([resp('first'),resp('second')]);await ag.run('remember X')
    async for _ in ag.stream_events('what was X?'):pass
    out('session_stream',sessions=len(ag._sessions),second_call_humans=[m.content for m in ag.llm.adapter.calls[1]['messages'] if m.role=='user'])
    # Storage never sees actual session id
    with tempfile.TemporaryDirectory() as tmp:
        ag=agent([resp('one'),resp('two')],mws=[StorageMiddleware(tmp)])
        await ag.run('one',session=ag.create_session('A'));await ag.run('two',session=ag.create_session('B'))
        out('storage_session',directories=[p.name for p in Path(tmp).iterdir()])
    # Responses request uses an argument absent from SDK signature
    gw=LLMGateway({'provider':'responses','api_key':'fake'})
    request=gw.adapter.build_request([])
    try: await gw.adapter.call(request)
    except Exception as e:out('responses_api',error=repr(e),request_keys=list(request))
    # Anthropic usage events use nested message.usage on start
    ad=object.__new__(AnthropicAdapter)
    x=ad.parse_stream_chunk(N(type='message_start',message=N(usage=N(input_tokens=100,output_tokens=1))))
    y=ad.parse_stream_chunk(N(type='message_delta',delta=N(stop_reason='end_turn'),usage=N(output_tokens=20)))
    out('anthropic_usage',start=x,end=y)
    # Thread id only appears in mapped events, not Agent session
    ag=agent([resp('first'),resp('second')])
    async for _ in ag_ui_events(ag,'remember X','thread-A'):pass
    async for _ in ag_ui_events(ag,'what was X?','thread-A'):pass
    out('agui_session',sessions=len(ag._sessions),second_call_humans=[m.content for m in ag.llm.adapter.calls[1]['messages'] if m.role=='user'])
    # Structured output bypasses run lifecycle
    from pydantic import BaseModel
    class Answer(BaseModel):value:int
    ag=agent([resp('{"value":1}')]);session=ag.create_session('structured')
    _,r=await ag.run_structured('answer',Answer,session=session)
    out('structured_output',result_type=type(r).__name__,history_roles=[m.role for m in session.context.get_messages()])
    # Step exhaustion masquerades as successful plan task
    async def noop():return 'ok'
    ex=PlanExecutor(FakeGateway([resp(tool_calls=[tc('noop')])]),max_steps_per_task=1)
    from wuwei.tools import ToolRegistry
    ex.tools=ToolRegistry();ex.tools.register(tool('noop',noop))
    result=await ex.execute([Task(id=1,description='finish',next=[],status='pending')])
    out('plan_step_limit',status=result.tasks[0].status,result=result.tasks[0].result)
    result=await PlanExecutor(FakeGateway([])).execute([Task(id=1,description='a',next=[2],status='pending'),Task(id=2,description='b',next=[1],status='pending')])
    out('plan_cycle',statuses=[t.status for t in result.tasks])
    out('ledger_tool_schema',tools={t.name:t.parameters.to_schema() for t in ledger_tools(ContextLedger())})
    # Runtime ActionLog stored in metadata breaks session serialization
    ag=agent([resp('hi')]);await ag.run('hi')
    try:json.dumps(ag._default_session.to_dict())
    except Exception as e:out('session_serialization',error=repr(e))
    # Same-second SQLite pagination drops entries
    with tempfile.TemporaryDirectory() as tmp:
        cp=SQLiteCheckpointer(str(Path(tmp)/'cp.db'))
        ids=[await cp.save(State()) for _ in range(3)]
        page=await cp.list_checkpoints(limit=1);nextpage=await cp.list_checkpoints(limit=10,before=page[0]['id'])
        out('checkpoint_pagination',saved=3,page=len(page),nextpage=len(nextpage))


from wuwei.graph.graph import StateGraph, END
from wuwei.config.yaml_loader import AgentConfig, create_agent_from_config
from wuwei.agent.sub_agent import SubAgent, SubAgentMiddleware
from wuwei.plugin.builtin.mcp import MCPToolDiscoveryMiddleware
from wuwei.plugin import PluginContext

async def extra():
    async def node(s,c):return {'value':42}
    g=StateGraph();g.add_node('node',node);g.set_entry_point('node');g.add_edge('node',END);app=g.compile()
    r=await app.invoke();out('graph_invoke',metadata=r.metadata)
    try:
        async for _ in app.stream():pass
    except Exception as e:out('graph_stream_reducer',error=repr(e))
    calls=[]
    async def deploy():calls.append('deploy');return 'ok'
    ag=agent([resp(tool_calls=[tc('deploy')]),resp('done'),resp('fork done')],[tool('deploy',deploy)],cp=MemoryCheckpointer())
    await ag.run('go');before=ag._default_session.context.to_dict()
    checkpoints=await ag.checkpointer.list_checkpoints();pre=next(x for x in checkpoints if x['metadata']['kind']=='pre_tool')
    await ag.fork(pre['id']);out('fork',sessions=len(ag._sessions),same_original_history=before==ag._default_session.context.to_dict(),executed=calls)
    # callback malformed config must fail closed rather than omit security hook
    config=AgentConfig(llm={'api_key':'fake'},middleware=[{'type':'hitl','policy':'typo'}])
    ag=create_agent_from_config(config);out('config_fail_open',middleware_count=len(ag.middleware))
    # child does not inherit parent policies / usage
    calls=[]
    async def danger():calls.append('danger');return 'executed'
    child_llm=FakeGateway([resp(tool_calls=[tc('danger')]),resp('child done')])
    sub=SubAgentMiddleware([SubAgent(name='worker',description='worker',tools=[tool('danger',danger)],model=child_llm)],parent_llm=child_llm)
    ag=agent([resp(tool_calls=[tc('task_worker',task='do work')]),resp('parent done')],mws=[PermissionMiddleware(PermissionConfig(denied_tools=['danger'])),sub])
    result=await ag.run('go');out('subagent_policy_and_usage',executed=calls,parent_llm_calls=result.llm_calls,actual_llm_calls=len(ag.llm.adapter.calls)+len(child_llm.adapter.calls))
    # MCP lazy discovery occurs after runner captures tools
    class Manager:
        failures={}
        async def connect_all(self):pass
        def list_servers(self):return ['server']
        def get_tools_by_server(self,name):return [tool('mcp__server__read',lambda: 'ok')]
    ag=agent([resp('first'),resp('second')]);ctx=PluginContext(tool_registry=ag.tool_registry,mcp_manager=Manager(),middleware_stack=ag.middleware)
    ag.middleware.add(MCPToolDiscoveryMiddleware(ctx));await ag.run('use mcp');await ag.run('retry')
    out('mcp_lazy_discovery',first_tools=[t.name for t in ag.llm.adapter.calls[0]['tools']],second_tools=[t.name for t in ag.llm.adapter.calls[1]['tools']],overview_first=any('MCP' in m.content for m in ag.llm.adapter.calls[0]['messages']),overview_second=any('MCP' in m.content for m in ag.llm.adapter.calls[1]['messages']))
    # a dynamically registered tool remains absent within same run
    ag=agent([resp(tool_calls=[tc('discover')]),resp('done')])
    async def discover():ag.tool_registry.register(tool('new_tool',lambda:'ok'));return 'new_tool registered'
    ag.tool_registry.register(tool('discover',discover));await ag.run('find and use')
    out('dynamic_tools',second_step_tools=[t.name for t in ag.llm.adapter.calls[1]['tools']],registry=[t.name for t in ag.tool_registry.list_tools()])


async def run_all():
    await main()
    await extra()

if __name__ == "__main__":
    asyncio.run(run_all())
