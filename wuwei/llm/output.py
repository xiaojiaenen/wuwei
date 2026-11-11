"""Structured output strategy shared by the runner and gateway."""

import json
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .profile import ModelProfile

STRUCTURED_TOOL_NAME = "__wuwei_structured_output"
STRUCTURED_SCHEMA_MARKER = "只输出符合以下JSON Schema的JSON对象："


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: dict = Field(alias="schema")
    name: str = Field(default="structured_output", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    mode: Literal["auto", "native", "tool", "prompted"] = "auto"

    def resolve(self, profile: ModelProfile, *, has_tools: bool = False) -> str:
        """解析实际输出模式。``auto`` 按 native → tool → prompted 降级。

        ``tool`` 模式要求该次请求没有可执行工具（结构化输出工具必须独占
        ``tool_choice``），因此带工具的请求在 auto 下继续落到 prompted，
        与既有行为一致；不带工具时优先选择比提示词更可靠的 tool 强制输出。
        """
        if self.mode == "auto":
            if profile.capabilities.native_schema:
                mode = "native"
            elif profile.capabilities.tool_calling and not has_tools:
                mode = "tool"
            else:
                mode = "prompted"
        else:
            mode = self.mode
        if mode == "native" and not profile.capabilities.native_schema:
            raise ValueError(f"{profile.name} does not support native structured output")
        if mode == "tool" and not profile.capabilities.tool_calling:
            raise ValueError(f"{profile.name} does not support tool output")
        return mode


def inject_prompted_schema(messages, schema) -> None:
    """把结构化输出约束并入首条 system 消息（runner 与 gateway 共用的唯一实现）。

    不能 ``insert(0, SystemMessage(...))``：会话第 0 条已经是 system prompt，
    再插一条就形成相邻两条 system，OpenAI 兼容端点（DeepSeek / vLLM / 各类
    网关）会直接 400 ``System message must be at the beginning``。把 schema
    追加到消息列表尾部同样会产生严格端点拒绝的 ``['system', ..., 'system']``
    形状，因此两条路径都必须合并进首条 system。

    合并而非新增还有两个好处：system 条数与顺序不变，prompt caching 的前缀
    保持稳定；注入的说明不会随轮次累加。重复调用是幂等的。
    """
    from ..core.message import SystemMessage

    text = STRUCTURED_SCHEMA_MARKER + json.dumps(schema, ensure_ascii=False)
    if not messages:
        messages.append(SystemMessage(content=text))
        return
    head = messages[0]
    if head.role != "system":
        # 没有可合并的 system（调用方自建上下文）时才新增，且必须放在首位。
        messages.insert(0, SystemMessage(content=text))
        return
    if STRUCTURED_SCHEMA_MARKER in (head.content or ""):
        return
    head.content = f"{head.content}\n\n{text}" if head.content else text
    # 首条 system 若带多模态块，字符串追加到 content 后 provider 会把它当文本块发，
    # 这里无需重排 content_parts。


def strict_json_schema(schema: dict) -> dict:
    """Build the strict object subset without mutating or weakening open maps.

    Native strict output requires explicit fields (nullable when optional).
    Pydantic defaults are applied locally, not sent as unsupported JSON defaults.
    Dynamic dictionaries/extra=allow cannot be represented faithfully; callers
    can select prompted mode rather than silently losing those fields.
    """
    result = deepcopy(schema)
    if result.get("type") != "object" and "$ref" not in result:
        raise ValueError("Strict structured output requires an object root")

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                additional = node.get("additionalProperties", False)
                if additional is not False:
                    raise ValueError(
                        "Strict output cannot represent open additionalProperties; use prompted mode"
                    )
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for child in node.values():
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(result)
    return result


def apply_output(request: dict, spec: OutputSpec, provider: str):
    schema = (
        strict_json_schema(spec.schema_) if provider in {"openai", "responses"} else spec.schema_
    )
    if provider == "openai":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": spec.name, "strict": True, "schema": schema},
        }
    elif provider == "responses":
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": spec.name,
                "strict": True,
                "schema": schema,
            }
        }
    elif provider == "anthropic":
        request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    else:
        raise ValueError(f"Native structured output is unsupported by {provider}")
