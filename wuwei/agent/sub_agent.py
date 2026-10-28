"""子代理系统

借鉴 DeepAgents 的 SubAgent 设计：
- SubAgent：声明式子代理配置（name, description, system_prompt, tools, model）
- SubAgentMiddleware：将子代理暴露为 task 工具
- 子代理结果通过 ToolMessage 返回父代理

使用方式：
    sub = SubAgent(
        name="researcher",
        description="Research topics and gather information",
        system_prompt="You are a research assistant...",
        tools=[search_tool, fetch_tool],
    )
    middleware = SubAgentMiddleware(sub_agents=[sub])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from wuwei.core.message import ToolMessage
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.tools import Tool


@dataclass
class SubAgent:
    """声明式子代理配置

    定义子代理的名称、描述、系统提示和可用工具。
    子代理继承父代理的 middleware（可覆写）。

    Attributes:
        name: 子代理名称（用作 task 工具名的一部分）
        description: 描述（LLM 用来决定何时委派）
        system_prompt: 子代理的系统提示
        tools: 可用工具列表
        model: 可选，独立指定 LLM 模型
        max_steps: 最大执行步数
        middleware: 可选，额外的 middleware
    """

    name: str
    description: str
    system_prompt: str = "You are a helpful assistant."
    tools: list[Tool] | None = None
    model: Any | None = None
    max_steps: int = 10
    middleware: list[Middleware] | None = None
    input_model: Any | None = None
    output_model: Any | None = None
    context_keys: tuple[str, ...] = ()
    artifact_keys: tuple[str, ...] = ()
    inherit_deps: bool = False

    def to_tool_schema(self) -> dict:
        """生成 task 工具的 OpenAI function schema"""
        return {
            "type": "function",
            "function": {
                "name": f"task_{self.name}",
                "description": (
                    f"Delegate a task to the '{self.name}' sub-agent. {self.description}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": f"Task description for {self.name}",
                        },
                    },
                    "required": ["task"],
                },
            },
        }


class SubAgentMiddleware(Middleware):
    """子代理中间件

    将配置的 SubAgent 暴露为 task 工具，
    LLM 可以通过调用 task_{name} 工具来委派任务给子代理。

    示例：
        sub = SubAgent(name="coder", description="Write and edit code", ...)
        middleware = SubAgentMiddleware(sub_agents=[sub], parent_llm=llm)
    """

    is_tool_handler = True

    def __init__(
        self,
        sub_agents: list[SubAgent],
        parent_llm: Any,
    ):
        """
        Args:
            sub_agents: 子代理配置列表
            parent_llm: 父代理的 LLM 网关（子代理默认继承）
        """
        self.sub_agents: dict[str, SubAgent] = {}
        self.parent_llm = parent_llm
        for sa in sub_agents:
            self.sub_agents[sa.name] = sa

    async def before_llm(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """注入子代理工具到可用工具列表

        将子代理工具 schema 写入 ctx.metadata，
        上层调用者 (AgentRunner.agent_node) 应读取并合并到 tools 列表。
        """
        ctx.metadata["_sub_agent_tools"] = self.get_task_tools()
        return ctx

    async def wrap_model_call(
        self,
        ctx: MiddlewareContext,
        messages: list[Any],
        tools: list[Any],
        next_handler: Any,
    ) -> Any:
        """包装 LLM 调用，注入子代理工具 schema

        将 task_{name} 工具 schema 追加到 tools 列表，
        使 LLM 可以调用子代理。
        """
        sub_schemas = self.get_task_tools()
        return await next_handler(messages, list(tools) + sub_schemas)

    async def before_tool(
        self,
        ctx: MiddlewareContext,
        tool_call: Any,
    ) -> Any:
        """拦截 task_* 工具调用并执行子代理

        返回 ToolMessage 表示"中间件已处理"——AgentRunner 会把它直接
        作为工具结果写回会话历史（对流式/非流式路径行为一致）。
        """
        tool_name = (
            tool_call.function.name
            if hasattr(tool_call, "function")
            else getattr(tool_call, "name", "")
        )
        if not tool_name.startswith("task_"):
            return tool_call  # 不是子代理工具，放行

        # 获取原始 tool_call_id
        tc_id = tool_call.id if hasattr(tool_call, "id") else getattr(tool_call, "tool_call_id", "")

        from wuwei.runtime.interrupt import InterruptSignal

        delegations = ctx.metadata.setdefault("delegations", {})
        previous = delegations.get(tc_id)
        args = tool_call.function.arguments if hasattr(tool_call, "function") else {}
        sub_agent = self.sub_agents.get(tool_name[5:])
        if sub_agent is None:
            return tool_call
        resume = previous.get("child_interrupt") if previous else None
        result = await self._run_sub_agent(
            sub_agent,
            args.get("task", ""),
            sub_agent.model or self.parent_llm,
            ctx=ctx,
            resume=resume,
        )
        payload = json.loads(result)
        if payload["status"] == "interrupted":
            continuation = {
                "kind": "sub_agent",
                "call_id": tc_id,
                "child_interrupt": payload["interrupt"],
                "child_session_id": payload["session_id"],
            }
            delegations[tc_id] = continuation
            signal = InterruptSignal(tool_call, reason="Sub-agent awaiting approval")
            signal.continuation = continuation
            raise signal
        delegations.pop(tc_id, None)
        from wuwei.tools.outcome import ToolOutcome

        result_msg = ToolOutcome.from_delegation(payload).to_message(tc_id, tool_name)
        # 确保 tool_call_id 匹配 LLM 的 tool call
        result_msg.tool_call_id = tc_id
        return result_msg

    def get_task_tools(self) -> list[dict]:
        """获取所有子代理的工具 schema"""
        return [sa.to_tool_schema() for sa in self.sub_agents.values()]

    async def handle_task_tool(
        self,
        tool_name: str,
        arguments: dict,
        ctx: MiddlewareContext | None = None,
    ) -> ToolMessage:
        """处理 task 工具调用

        从 tool_name 中提取子代理名，构建并执行子代理。

        Args:
            tool_name: 工具名（格式：task_{sub_agent_name}）
            arguments: 工具参数（包含 task 描述）
            ctx: 中间件上下文

        Returns:
            包含子代理执行结果的 ToolMessage
        """
        # 解析子代理名
        if not tool_name.startswith("task_"):
            raise ValueError(f"Unknown task tool: {tool_name}")

        sub_name = tool_name[5:]  # 去掉 "task_" 前缀
        sub_agent = self.sub_agents.get(sub_name)
        if not sub_agent:
            raise ValueError(f"Unknown sub-agent: {sub_name}")

        task_description = arguments.get("task", "")
        if not task_description:
            raise ValueError("Missing 'task' argument")

        # 构建子代理的 LLM（继承或独立）
        llm = sub_agent.model or self.parent_llm

        # 构建子代理图
        result = await self._run_sub_agent(sub_agent, task_description, llm, ctx=ctx)

        from wuwei.tools.outcome import ToolOutcome

        return ToolOutcome.from_delegation(json.loads(result)).to_message(tool_name, tool_name)

    async def _run_sub_agent(
        self,
        sub_agent: SubAgent,
        task: str,
        llm: Any,
        ctx: MiddlewareContext | None = None,
        resume: dict | None = None,
    ) -> str:
        """Use the same governed engine as ordinary agent runs."""
        from wuwei.agent.agent import Agent
        from wuwei.runtime.context import current_context

        from wuwei.runtime.handoff import DelegationInput

        parent = ctx or current_context.get()
        if sub_agent.input_model is not None:
            validated = sub_agent.input_model.model_validate_json(task)
            task = validated.model_dump_json()
        selected = {
            key: parent.metadata[key]
            for key in sub_agent.context_keys
            if parent is not None and key in parent.metadata
        }
        import mimetypes
        from wuwei.artifacts import ArtifactRef
        from wuwei.core.message import FilePart, HumanMessage, ImagePart

        refs = []
        for key in sub_agent.artifact_keys if not resume else ():
            value = parent.metadata.get(key) if parent is not None else None
            for ref in value if isinstance(value, list) else ([] if value is None else [value]):
                refs.append(ArtifactRef.model_validate(ref))
        if selected or refs:
            handoff = DelegationInput(task=task, context=selected, artifact_refs=refs)
            task = handoff.model_dump_json()
        if refs:
            store, owner = (
                getattr(llm, "artifact_store", None),
                getattr(llm, "artifact_owner", None),
            )
            if store is None or not owner:
                raise ValueError(
                    "Delegated artifacts require an authorized ArtifactStore and owner"
                )
            parts = []
            for ref in refs:
                store.read(ref, owner=owner)  # Validate before handing data to the child.
                parts.append(
                    ImagePart(artifact=ref)
                    if ref.media_type.startswith("image/")
                    else FilePart(
                        artifact=ref,
                        filename=ref.id + (mimetypes.guess_extension(ref.media_type) or ".bin"),
                    )
                )
            task = HumanMessage(task, content_parts=parts)
        config = parent.config if parent is not None else {}
        worker = Agent(
            llm=llm,
            tools=sub_agent.tools or [],
            load_builtins=False,
            default_system_prompt=sub_agent.system_prompt,
            default_max_steps=sub_agent.max_steps,
            middleware=inherit_middleware(
                config.get("middleware"), sub_agent.middleware, exclude=self
            ),
            checkpointer=config.get("checkpointer"),
            event_bus=config.get("event_bus"),
            deps=(
                config["run_context"].deps
                if sub_agent.inherit_deps and config.get("run_context") is not None
                else None
            ),
        )
        session = None if resume else worker.create_session()
        session_id = resume["session_id"] if resume else session.session_id
        started = time.monotonic()
        try:
            if resume:
                decision = parent.metadata.get("continuation_decision", {}) if parent else {}
                result = await worker.resume(
                    resume,
                    approved=decision.get("approved", True),
                    reason=decision.get("reason", ""),
                )
            else:
                if sub_agent.output_model is not None:
                    _, result = await worker.run_structured(
                        task, sub_agent.output_model, session=session
                    )
                else:
                    result = await worker.run(task, session=session)
        except BaseException as exc:
            stopped = worker._sessions.get(session_id, session)
            exc.child_usage = dict(stopped.last_usage) if stopped else {}
            exc.child_llm_calls = stopped.last_llm_calls if stopped else 0
            exc.child_unknown = (
                {"session_id": session_id, "run_id": stopped.metadata["_blocked_run"]}
                if stopped and stopped.metadata.get("_blocked_run")
                else None
            )
            raise
        stopped = worker._sessions.get(session_id)
        from wuwei.runtime.handoff import DelegationOutput

        payload = {
            "sub_agent": sub_agent.name,
            "result": (
                result.output.model_dump(mode="json")
                if hasattr(result.output, "model_dump")
                else result.output
            )
            if sub_agent.output_model is not None and result.output is not None
            else result.content,
            "artifact_refs": [
                part.artifact.model_dump(mode="json")
                for part in result.content_parts
                if getattr(part, "artifact", None) is not None
            ],
            "status": result.status,
            "usage": result.usage,
            "llm_calls": result.llm_calls,
            "interrupt": getattr(result, "interrupt", None),
            "session_id": session_id,
            "reconciliation": (
                {"session_id": session_id, "run_id": stopped.metadata["_blocked_run"]}
                if stopped and stopped.metadata.get("_blocked_run")
                else None
            ),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "steps": getattr(result, "steps", 0),
        }
        return DelegationOutput.model_validate(payload).model_dump_json()


def inherit_middleware(parent=None, explicit=None, *, exclude=None):
    """Preserve parent guards and add child constraints without rebinding them.

    A fresh stack owns hook ordering; middleware instances retain their configured
    policies. The active dispatcher is omitted to prevent unbounded self-delegation.
    """
    from wuwei.middleware.stack import MiddlewareStack

    stack = MiddlewareStack()
    for source in (parent, explicit):
        for middleware in getattr(source, "middlewares", source or []):
            if middleware is not exclude and middleware not in stack.middlewares:
                stack.add(middleware)
    return stack
