"""下游3.0.2反馈针对当前工作树的离线验证；不访问用户服务/真实模型。"""
import asyncio
import inspect
import json
from types import SimpleNamespace as N
from unittest.mock import AsyncMock, patch
import importlib.metadata as metadata
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from anthropic.types import Message as AnthropicMessage
from wuwei.llm.adapters.openai import OpenAIAdapter
from wuwei.llm.adapters.anthropic import AnthropicAdapter
from wuwei.llm.gateway import LLMGateway
from wuwei.llm.types import LLMResponse, LLMResponseChunk
from wuwei.core.types import LLMResponse as LegacyResponse
from wuwei.core.message import BaseMessage, AIMessage

def emit(probe, **kw):print(json.dumps({'probe':probe,**kw},ensure_ascii=False,default=str))
def raw_response(finish='stop',arguments=None):
    return ChatCompletion.model_validate({'id':'fake','created':0,'object':'chat.completion','model':'fake','choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':'ok','tool_calls':([{'id':'call_1','type':'function','function':{'name':'search','arguments':arguments}}] if arguments is not None else None)}}],'usage':None})
class Adapter:
    def __init__(self,events=(),error=None):self.events=list(events);self.error=error;self.calls=0
    def build_request(self,*args,**kwargs):return {}
    async def call(self,request):
        self.calls+=1
        if self.error:raise self.error
        async def gen():
            for event in self.events:yield event
        return gen()
    def parse_stream_chunk(self,chunk):return chunk
    def parse_response(self,raw):return raw

def gateway(adapter):
    gw=LLMGateway({'api_key':'fake','retry':{'max_attempts':3},'timeout':.01});gw.adapter=adapter;return gw

async def main():
    emit('versions',**{n:metadata.version(n) for n in ['openai','anthropic','mcp','pydantic']})
    emit('duplicate_response_types',same=LegacyResponse is LLMResponse,legacy_fields=list(LegacyResponse.model_fields),public_fields=list(LLMResponse.model_fields))
    ad=OpenAIAdapter(api_key='fake',temperature=.73,max_tokens=17)
    request=ad.build_request([])
    emit('openai_config_ignored',temperature=request.get('temperature'),max_tokens=request.get('max_tokens'))
    polluted=OpenAIAdapter(api_key='fake',max_retries=0)
    request=polluted.build_request([])
    try:inspect.signature(polluted.client.chat.completions.create).bind(**request)
    except TypeError as e:emit('client_request_parameter_mix',error=str(e),sdk_retries=polluted.client.max_retries)
    response=ad.parse_response(raw_response())
    emit('usage_none',content=response.message.content,usage=response.usage)
    chunk=ChatCompletionChunk.model_validate({'id':'fake','created':0,'object':'chat.completion.chunk','model':'fake','choices':[],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
    emit('usage_only_frame',parsed=ad.parse_stream_chunk(chunk))
    try:ad.parse_response(raw_response('eos'))
    except Exception as e:emit('unknown_finish_reason',error=type(e).__name__)
    parsed=ad.parse_response(raw_response(arguments='{bad'))
    emit('invalid_tool_json',actual_arguments=parsed.message.tool_calls[0].function.arguments)
    events=[{'tool_calls_delta':[{'index':0,'id':'call_1','name':'search','arguments':'{"q":"x"}'}]},{'finish_reason':'stop'}]
    gw=gateway(Adapter(events));chunks=[c async for c in await gw.generate([],stream=True)]
    emit('tool_delta_ends_stop',complete=[c.tool_calls_complete for c in chunks])
    primary=Adapter(error=TypeError('bad local argument'));fallback=Adapter(error=RuntimeError('fallback used'))
    gw=gateway(primary);gw._fallback_adapter=fallback
    with patch('wuwei.llm.gateway.asyncio.sleep',new_callable=AsyncMock):
        try:await gw.generate([])
        except Exception as e:emit('deterministic_retry',primary_calls=primary.calls,fallback_calls=fallback.calls,error=type(e).__name__)
    primary=Adapter(error=ConnectionError('offline fake'));fallback=Adapter()
    gw=gateway(primary);gw._fallback_adapter=fallback
    try:
        async for _ in await gw.generate([],stream=True):pass
    except Exception as e:emit('stream_retry_fallback',primary_calls=primary.calls,fallback_calls=fallback.calls,error=type(e).__name__)
    class Hanging(Adapter):
        async def call(self,request):
            self.calls+=1
            async def gen():
                await asyncio.sleep(10)
                yield {'content':'late'}
            return gen()
    gw=gateway(Hanging())
    async def consume():
        async for _ in await gw.generate([],stream=True):pass
    try:await asyncio.wait_for(consume(),timeout=.05)
    except TimeoutError:emit('stream_timeout',configured=.01,external_cancel_after=.05,observed='outer timeout needed')
    an=AnthropicAdapter(api_key='fake')
    native=AnthropicMessage.model_validate({'id':'fake','type':'message','role':'assistant','content':[{'type':'text','text':'cut off'}],'model':'fake','stop_reason':'max_tokens','stop_sequence':None,'usage':{'input_tokens':10,'output_tokens':5}})
    emit('anthropic_truncation',raw=native.stop_reason,parsed=an.parse_response(native).finish_reason,stream=an.parse_stream_chunk(N(type='message_delta',delta=N(stop_reason='max_tokens'),usage=N(output_tokens=5))))
    try:LLMGateway({'provider':'anthropic','api_key':'fake','base_url':'https://example.invalid'})
    except Exception as e:emit('anthropic_base_url',error=str(e))
    emit('lifecycle',gateway_has_close=hasattr(gw,'aclose'),gateway_has_context_manager=hasattr(gw,'__aenter__'))
    msg=BaseMessage(role='assistant',content='',tool_calls=parsed.message.tool_calls)
    emit('base_message_serialization',output=msg.to_openai())
    reason=ad.build_request([AIMessage(content='ok',reasoning_content='provider-specific')])
    emit('reasoning_sent_unconditionally',messages=reason['messages'])
    class StatusError(Exception):status_code=401
    emit('status_error_classification',classification=gw._classify_error(StatusError('gateway at :5000')))
    from wuwei.tools import ToolRegistry
    registry=ToolRegistry()
    async def list_tool(items:list[str]):return items
    registry.register_callable(list_tool)
    emit('generic_tool_schema',schema=registry.list_tools()[0].parameters.to_schema())
    import httpx
    from wuwei.serve.a2a import build_a2a_app, build_agent_card
    class FakeAgent:
        async def run(self, text):return N(content='ok',status='completed',usage={})
    app=build_a2a_app(FakeAgent(),name='audit',description='audit')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://audit') as client:
        for method in ['message/send','SendMessage','tasks/send']:
            payload={'jsonrpc':'2.0','id':'rpc-1','method':method,'params':{'message':{'role':'user','messageId':'msg-1','parts':[{'kind':'text','text':'hi'}]}}}
            response=await client.post('/',json=payload)
            emit('a2a_method',method=method,status=response.status_code,payload=response.json())
    card=build_agent_card(FakeAgent(),name='audit',description='audit')
    emit('a2a_interface',interfaces=card['supportedInterfaces'])
    # Close only real SDK clients constructed here; fake adapters own no resources.
    await ad.client.close();await polluted.client.close();await an.client.close()

if __name__=='__main__':asyncio.run(main())
