"""JSON wire encoding for typed values; unsupported objects fail explicitly.

Unlike default=str this preserves models/dataclasses as structured data and never
calls arbitrary objects' __str__ to expose them to remote clients.
"""

import json

from pydantic_core import to_jsonable_python


def json_default(value):
    return to_jsonable_python(value, by_alias=True)


def wire_json_dumps(value):
    return json.dumps(value, ensure_ascii=False, default=json_default, allow_nan=False)
