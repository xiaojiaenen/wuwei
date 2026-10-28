import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from wuwei.serve._http import json_response
from wuwei.serve.ag_ui import _sse_bytes


class Payload(BaseModel):
    when: datetime
    amount: Decimal


@dataclass
class Record:
    name: str


def data():
    return {
        "payload": Payload(
            when=datetime(2026, 10, 28, tzinfo=timezone.utc), amount=Decimal("1.234567890123456789")
        ),
        "id": UUID("12345678-1234-1234-1234-123456789abc"),
        "path": Path("report.pdf"),
        "record": Record("中文\n行"),
    }


def test_http_and_sse_share_typed_json_contract():
    body = json.loads(json_response(data()).body)
    wire = _sse_bytes({"type": "CUSTOM", "value": data()})
    event = json.loads(wire.removeprefix("data: ").strip())
    assert body == event["value"]
    assert body["payload"]["amount"] == "1.234567890123456789"
    assert body["record"]["name"] == "中文\n行"
    assert wire.count("\n\n") == 1


def test_unknown_objects_are_not_stringified_or_exposed():
    class Secret:
        def __str__(self):
            return "secret"

    with pytest.raises((TypeError, ValueError)):
        json_response({"bad": Secret()})
