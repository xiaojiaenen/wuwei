from types import SimpleNamespace as Ns

import pytest

from wuwei.llm.adapters.openai import OpenAIAdapter
from wuwei.llm.adapters.responses import ResponsesAdapter
from wuwei.llm.gateway import LLMGateway
from wuwei.runtime.agent_runner import _merge_usage


def test_modern_responses_omits_sampling_unless_explicit():
    adapter = LLMGateway({'provider': 'responses', 'api_key': 'unused', 'client': Ns()}).adapter
    assert 'temperature' not in adapter.build_request([])
    assert adapter.build_request([], temperature=0.7)['temperature'] == 0.7
    assert 'temperature' not in adapter.build_request([], temperature=None)


def test_legacy_chat_defaults_and_explicit_modern_parameters():
    old = OpenAIAdapter('unused', client=Ns())
    assert old.build_request([])['temperature'] == 0.2
    modern = LLMGateway({'api_key': 'unused', 'client': Ns(), 'temperature': None,
                         'max_completion_tokens': 100, 'reasoning_effort': 'high'})
    request = modern.adapter.build_request([])
    assert 'temperature' not in request and 'max_tokens' not in request
    assert request['max_completion_tokens'] == 100
    assert request['reasoning_effort'] == 'high'


@pytest.mark.parametrize('stream', [False, True])
def test_cache_parameters_reach_sdk(stream):
    adapter = ResponsesAdapter('unused', client=Ns(),
                               prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
                               prompt_cache_key='stable-prefix')
    request = adapter.build_request([], stream=stream)
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['prompt_cache_key'] == 'stable-prefix'
    assert request['stream'] == stream


def response_usage():
    from openai.types.responses.response_usage import ResponseUsage
    return ResponseUsage(input_tokens=100, output_tokens=30, total_tokens=130,
                         input_tokens_details={'cached_tokens': 60, 'cache_write_tokens': 20},
                         output_tokens_details={'reasoning_tokens': 10})


def test_responses_usage_details_nonstream_and_stream():
    adapter = ResponsesAdapter('unused', client=Ns())
    raw = Ns(output=[], usage=response_usage(), status='completed', model='test')
    usage = adapter.parse_response(raw).usage
    assert usage == {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130,
                     'cached_tokens': 60, 'cache_write_tokens': 20, 'reasoning_tokens': 10}
    event = adapter.parse_stream_chunk(Ns(type='response.completed', response=raw))
    assert event['usage'] == usage
    total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    _merge_usage(total, usage)
    _merge_usage(total, usage)
    assert total['cached_tokens'] == 120 and total['total_tokens'] == 260


def test_chat_usage_details_and_missing_fields():
    usage = Ns(prompt_tokens=100, completion_tokens=30, total_tokens=130,
               prompt_tokens_details=Ns(cached_tokens=60),
               completion_tokens_details=Ns(reasoning_tokens=10))
    parsed = OpenAIAdapter._parse_usage(usage)
    assert parsed['cached_tokens'] == 60 and parsed['reasoning_tokens'] == 10
    minimal = OpenAIAdapter._parse_usage(Ns(prompt_tokens=1, completion_tokens=2))
    assert minimal == {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_details_survive_agent_and_session(stream):
    from tests.test_runtime_hardening import make_agent, response
    answer = response('done')
    answer.usage = ResponsesAdapter._parse_usage(response_usage())
    agent = make_agent([answer])
    session = agent.create_session()
    if stream:
        events = [event async for event in agent.stream_events('hello', session=session)]
        usage = events[-1].data['usage']
    else:
        usage = (await agent.run('hello', session=session)).usage
    assert usage['cached_tokens'] == 60 and usage['reasoning_tokens'] == 10
    assert usage['total_tokens'] == 130
    assert session.last_usage == usage


def test_plan_aggregation_preserves_details():
    from wuwei.planning import PlanExecutor
    executor = PlanExecutor(Ns())
    detail = ResponsesAdapter._parse_usage(response_usage())
    executor._tasks_usage = [(detail, 1), (detail, 1)]
    usage, calls = executor._aggregate_usage()
    assert calls == 2 and usage['cache_write_tokens'] == 40
    assert usage['total_tokens'] == 260


@pytest.mark.asyncio
async def test_cache_request_serialized_by_actual_sdk():
    import json
    import httpx
    from openai import AsyncOpenAI

    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'resp_1', 'object': 'response',
                              'created_at': 0, 'status': 'completed', 'model': 'test',
                              'output': [], 'usage': response_usage().model_dump()})

    async with AsyncOpenAI(api_key='unused', max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
        adapter = ResponsesAdapter('unused', client=client,
                                   prompt_cache_options={'mode': 'implicit', 'ttl': '30m'})
        raw = await adapter.call(adapter.build_request([]))
        assert adapter.parse_response(raw).usage['cached_tokens'] == 60
    assert seen[0]['prompt_cache_options']['ttl'] == '30m'
    assert 'temperature' not in seen[0]


def test_responses_original_positional_constructor():
    client = Ns()
    adapter = ResponsesAdapter('unused', 'test', None, 0.4, 42, client)
    request = adapter.build_request([])
    assert request['max_output_tokens'] == 42 and request['temperature'] == 0.4
    assert adapter.client is client
