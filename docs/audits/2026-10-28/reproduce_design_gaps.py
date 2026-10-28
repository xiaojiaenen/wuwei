"""Offline evidence for the 3.2.2 design review; run from repository root."""
import asyncio
from types import SimpleNamespace as Ns

from pydantic import ValidationError
from wuwei.config.yaml_loader import LLMConfig, AgentConfig, create_agent_from_config
from wuwei.evals import EvalRunner, EvalCase
from tests.test_runtime_hardening import make_agent, response, call, tool


async def main():
    for data in ({'provider': 'responses', 'prompt_cache_options': {'mode': 'implicit', 'ttl': '30m'}},
                 {'provider': 'responses', 'temperature': None}):
        try:
            LLMConfig(**data)
            print('yaml accepts', data)
        except ValidationError as error:
            print('yaml rejects:', error.errors()[0]['loc'], error.errors()[0]['type'])
    agent = create_agent_from_config(AgentConfig(llm=LLMConfig(provider='responses', api_key='unused'),
                                                load_builtins=False))
    print('yaml Responses temperature:', agent.llm.adapter.build_request([]).get('temperature'))
    await agent.llm.aclose()

    class StubAgent:
        def create_session(self):
            return Ns(context=Ns(get_messages=lambda: []))
        async def run(self, *args, **kwargs):
            return Ns(status='incomplete', content='expected', usage={})
    result = await EvalRunner(StubAgent).run_case(EvalCase(name='terminal-state', input='x',
                                                         response_contains=['expected']))
    print('incomplete eval passed:', result.passed)

    async def business_status():
        return {'status': 'unknown', 'meaning': 'ordinary domain data'}
    agent = make_agent([response(calls=[call('lookup')]), response('done')],
                       [tool('lookup', business_status)])
    result = await agent.run('lookup')
    print('successful lookup run status:', result.status, result.termination_reason)


if __name__ == '__main__':
    asyncio.run(main())
