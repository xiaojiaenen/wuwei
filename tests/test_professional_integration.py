"""Cross-feature entry points, not only individual model construction."""

import json
from types import SimpleNamespace as Ns

import pytest
from pydantic import BaseModel

from wuwei import (
    Agent,
    ArtifactStore,
    FilePart,
    HumanMessage,
    LLMGateway,
    SQLiteSessionStore,
    TextPart,
)


@pytest.mark.asyncio
async def test_agent_native_schema_media_and_durable_history(tmp_path):
    seen = []

    class Answer(BaseModel):
        title: str
        note: str | None = None

    async def create(**request):
        seen.append(request)
        from openai.types.responses import Response

        return Response.model_validate(
            {
                "id": "response1",
                "created_at": 0,
                "object": "response",
                "model": "test",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "output": [
                    {
                        "id": "message1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"title":"PDF","note":null}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    blobs = ArtifactStore(tmp_path / "blobs")
    ref = blobs.put(b"%PDF-test", owner="a", media_type="application/pdf")
    gateway = LLMGateway(
        {
            "provider": "responses",
            "client": Ns(responses=Ns(create=create)),
            "artifact_store": blobs,
            "artifact_owner": "a",
        }
    )
    store = SQLiteSessionStore(tmp_path / "history.db", namespace="a")
    agent = Agent(gateway, load_builtins=False, session_store=store)
    session = agent.create_session("document")
    output, result = await agent.run_structured(
        HumanMessage("summarize", content_parts=[FilePart(artifact=ref, filename="r.pdf")]),
        Answer,
        session=session,
    )
    assert output.title == "PDF" and result.status == "completed"
    assert seen[0]["text"]["format"]["strict"] is True
    assert seen[0]["text"]["format"]["schema"]["additionalProperties"] is False
    assert "data:application/pdf;base64," in json.dumps(seen[0])
    history = (await store.load("document")).to_dict()
    assert "data:application/pdf;base64," not in json.dumps(history)
    assert ref.id in json.dumps(history)


@pytest.mark.asyncio
async def test_refusal_is_not_successful_structured_output():
    class Answer(BaseModel):
        value: int

    async def create(**request):
        return Ns(
            output=[Ns(type="message", content=[Ns(type="refusal", refusal="not allowed")])],
            status="completed",
            model="test",
            usage=None,
        )

    gateway = LLMGateway({"provider": "responses", "client": Ns(responses=Ns(create=create))})
    agent = Agent(gateway, load_builtins=False)
    output, result = await agent.run_structured("disallowed", Answer)
    assert output is None and result.termination_reason == "refusal"
    assert result.status == "incomplete" and result.metadata["refusal"]


@pytest.mark.asyncio
async def test_diagnostic_summary_excludes_prompt_and_tool_arguments(tmp_path, capsys):
    from wuwei.graph.checkpoint import SQLiteCheckpointer
    from wuwei.graph.state import State
    from wuwei.core.message import HumanMessage
    from wuwei.debug import main

    cp = SQLiteCheckpointer(str(tmp_path / "cp.db"))
    cid = await cp.save(
        State(
            messages=[HumanMessage("secret prompt")],
            metadata={
                "run_state": {
                    "run_id": "r",
                    "status": "completed",
                    "usage_total": {"total_tokens": 12},
                    "tool_states": {"call": "success"},
                    "api_key": "must-not-appear",
                },
                "session_id": "s",
                "kind": "final",
            },
        )
    )
    # main owns asyncio.run; execute in a thread rather than nesting event loops.
    import asyncio

    assert await asyncio.to_thread(main, [str(tmp_path / "cp.db"), "--summary", cid]) == 0
    output = capsys.readouterr().out
    assert "secret prompt" not in output and "must-not-appear" not in output
    assert "completed" in output and "12" in output
