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

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from wuwei.core.message import AIMessage, HumanMessage, SystemMessage, ToolMessage
from wuwei.llm.types import Message
from wuwei.graph.graph import END, CompiledGraph, StateGraph
from wuwei.graph.state import State
from wuwei.middleware.base import Middleware, MiddlewareContext
from wuwei.tools import Tool, ToolRegistry


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

    def to_tool_schema(self) -> dict:
        """生成 task 工具的 OpenAI function schema"""
        return {
            "type": "function",
            "function": {
                "name": f"task_{self.name}",
                "description": (
                    f"Delegate a task to the '{self.name}' sub-agent. "
                    f"{self.description}"
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

        注意：实际工具注入通过 tools 列表完成，
        这里作为 before_llm 钩子参与中间件生命周期。
        """
        return ctx

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
        result = await self._run_sub_agent(sub_agent, task_description, llm)

        return ToolMessage(
            content=result,
            tool_call_id="",
            name=tool_name,
        )

    async def _run_sub_agent(
        self,
        sub_agent: SubAgent,
        task: str,
        llm: Any,
    ) -> str:
        """执行子代理

        构建标准 Agent 循环图并执行。
        """
        from wuwei.middleware.stack import MiddlewareStack
        from wuwei.tools import ToolExecutor

        # 构建工具注册表
        registry = ToolRegistry()
        if sub_agent.tools:
            for tool in sub_agent.tools:
                registry.register(tool)
        executor = ToolExecutor(registry)

        # 构建中间件栈
        mw_stack = MiddlewareStack()
        if sub_agent.middleware:
            for mw in sub_agent.middleware:
                mw_stack.add(mw)

        all_tools = list(registry.list_tools())

        # 构建初始状态
        start_time = time.monotonic()
        state = State(
            messages=[
                SystemMessage(content=sub_agent.system_prompt),
                HumanMessage(content=task),
            ],
            metadata={},
            step=0,
        )

        # 构建子代理图
        graph = StateGraph(State)
        graph.set_max_steps(sub_agent.max_steps)

        async def agent_node(s: State, config: dict | None = None) -> State:
            messages = [m for m in s.messages]
            tools = list(all_tools)

            ctx = MiddlewareContext(state=s, config=config or {}, step=s.step)
            ctx = await mw_stack.execute_before_llm(ctx)
            if hasattr(ctx.state, 'messages'):
                messages = ctx.state.messages

            response = await llm.generate(messages, tools=tools)

            ai_msg = AIMessage(
                content=response.message.content or "",
                tool_calls=response.message.tool_calls or [],
            )

            ctx = await mw_stack.execute_after_llm(ctx, ai_msg)
            s.add_message(ai_msg)
            s.metadata["_has_tool_calls"] = bool(response.message.tool_calls)
            s.metadata["_tool_calls"] = response.message.tool_calls or []
            s.metadata["_last_content"] = response.message.content or ""
            return s

        async def tool_node(s: State, config: dict | None = None) -> State:
            tool_calls = s.metadata.get("_tool_calls", [])
            ctx = MiddlewareContext(state=s, config=config or {}, step=s.step)

            for tc in tool_calls:
                modified = await mw_stack.execute_before_tool(ctx, tc)
                if modified is None:
                    continue

                try:
                    msg = await executor.execute_one(modified)
                except Exception as exc:
                    msg = Message(
                        role="tool",
                        content=f"Tool error: {exc}",
                        tool_call_id=modified.id,
                        name=modified.function.name,
                    )

                tool_msg = ToolMessage(
                    content=msg.content or "",
                    tool_call_id=msg.tool_call_id or modified.id,
                    name=msg.name or modified.function.name,
                )
                modified_msg = await mw_stack.execute_after_tool(ctx, tool_msg)
                s.add_message(
                    ToolMessage(
                        role="tool",
                        content=modified_msg.content or "",
                        tool_call_id=modified_msg.tool_call_id,
                        name=modified_msg.name,
                    )
                )
            return s

        async def should_continue(s: State, config: dict | None = None) -> str:
            if s.metadata.get("_has_tool_calls"):
                return "continue"
            return "end"

        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
        graph.add_edge("tools", "agent")

        compiled = graph.compile()
        final_state = await compiled.invoke(state)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # 提取结果
        last_ai = final_state.get_last_ai_message()
        result_content = ""
        if last_ai and last_ai.content:
            result_content = last_ai.content

        # 构建完整的 ToolMessage 结果
        result = json.dumps({
            "sub_agent": sub_agent.name,
            "result": result_content,
            "elapsed_ms": elapsed_ms,
            "steps": final_state.step,
        }, ensure_ascii=False)

        return result
