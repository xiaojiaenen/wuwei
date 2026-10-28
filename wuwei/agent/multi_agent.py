"""多 Agent 协作模块 — StateGraph 驱动版

借鉴 LangGraph 的 Send API 和 DeepAgents 的 SubAgent 模式：

通信模式：
1. **Fan-out（扇出）**：Leader 分解任务后，多个 Member 并行执行
2. **Handoff（交接）**：Agent 直接交接给另一个 Agent（带上下文）
3. **Shared State（共享状态）**：成员通过 Leader 分派与结果汇总协作

架构：
    StateGraph:
        ┌──────────┐
        │  leader  │ ← 分解任务
        └────┬─────┘
             │ (条件边: fan_out)
    ┌────────┼────────┐
    ▼        ▼        ▼
  [A]      [B]      [C]   ← 并行执行（asyncio.gather）
    └────────┼────────┘
             │ (聚合边: fan_in)
        ┌────┴─────┐
        │ synthesis│ ← LLM 汇总
        └──────────┘
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field

from wuwei.agent.agent import Agent
from wuwei.core.message import AIMessage, HumanMessage, SystemMessage
from wuwei.graph.graph import END, CompiledGraph, StateGraph
from wuwei.graph.state import State
from wuwei.middleware.base import Middleware, MiddlewareContext

_delegate_depth = ContextVar("wuwei_delegate_depth", default=0)


class DelegationStoppedError(RuntimeError):
    """A member did not complete; result retains usage and approval payload."""

    def __init__(self, result):
        self.result = result
        self.status = result.status
        self.interrupt = getattr(result, "interrupt", None)
        super().__init__(f"Delegated agent stopped: {result.status}")


async def _governed_run(agent, task, *, system_prompt=None, middleware=None):
    from wuwei.agent.sub_agent import inherit_middleware
    from wuwei.runtime.context import current_context

    parent = current_context.get()
    depth = _delegate_depth.get()
    if depth >= 8:
        raise RuntimeError("Maximum delegation depth exceeded")
    token = _delegate_depth.set(depth + 1)
    try:
        worker = Agent(
            llm=agent.llm,
            tools=agent.tool_registry,
            load_builtins=False,
            default_system_prompt=system_prompt or agent.default_system_prompt,
            default_max_steps=agent.default_max_steps,
            middleware=inherit_middleware(
                parent.config.get("middleware") if parent else None,
                inherit_middleware(middleware, agent.middleware),
            ),
            checkpointer=agent.checkpointer,
            event_bus=agent.event_bus,
        )
        result = await worker.run(task, session=worker.create_session())
        if result.status != "completed":
            raise DelegationStoppedError(result)
        return result
    finally:
        _delegate_depth.reset(token)


@dataclass
class TeamMember:
    """团队成员 — 拥有独立 Agent 实例"""

    name: str
    agent: Agent
    role: str
    description: str = ""
    tools: list[str] = field(default_factory=list)


@dataclass
class HandoffSignal:
    """Agent 间交接信号

    借鉴 LangGraph 的 Command(goto=...) 模式。
    一个 Agent 完成任务后，可以将控制权交接给另一个 Agent。
    """

    source: str
    target: str
    context: str  # 交接时的上下文/结果摘要
    task: str  # 新 Agent 要执行的任务
    metadata: dict = field(default_factory=dict)


class MultiAgentGraph:
    """基于 StateGraph 的多 Agent 协作图

    替代旧的 Swarm 类，使用图驱动模式：
    - 任务分解和执行由 StateGraph 节点完成
    - 支持并行扇出（fan-out）
    - 支持 Agent 间 Handoff
    - 所有通信通过 State.messages 结构化传递
    - 支持检查点（可恢复）

    示例：
        graph = MultiAgentGraph()
        graph.add_worker("researcher", researcher_agent, role="研究员")
        graph.add_worker("writer", writer_agent, role="写手")
        graph.set_leader(leader_agent)

        result = await graph.run("写一篇关于AI的文章")
    """

    def __init__(self):
        self.leader: Agent | None = None
        self.members: dict[str, TeamMember] = {}
        self._max_steps: int = 20

    def set_leader(self, agent: Agent) -> MultiAgentGraph:
        """设置领导者 Agent"""
        self.leader = agent
        return self

    def add_worker(
        self,
        name: str,
        agent: Agent,
        role: str,
        description: str = "",
        tools: list[str] | None = None,
    ) -> MultiAgentGraph:
        """添加工作 Agent"""
        self.members[name] = TeamMember(
            name=name,
            agent=agent,
            role=role,
            description=description,
            tools=tools or [],
        )
        return self

    def set_max_steps(self, steps: int) -> MultiAgentGraph:
        """设置最大步数"""
        if steps <= 0:
            raise ValueError("steps must be positive")
        self._max_steps = steps
        return self

    async def run(self, task: str) -> str:
        """运行多 Agent 协作

        Returns:
            最终汇总结果
        """
        if not self.leader:
            raise ValueError("请先调用 set_leader() 设置领导者")
        if not self.members:
            # 没有成员，领导自己执行
            result = await _governed_run(self.leader, task)
            return result.content if hasattr(result, "content") else str(result)

        graph = self._build_graph()
        initial_state = State(
            messages=[
                SystemMessage(content=self._build_leader_system_prompt()),
                HumanMessage(content=task),
            ],
            metadata={},
            step=0,
        )

        final_state = await graph.invoke(initial_state)
        if not final_state.metadata.get("_team_completed"):
            raise RuntimeError("Team execution incomplete: graph step limit reached")
        last_ai = final_state.get_last_ai_message()
        return last_ai.content if last_ai else ""

    async def run_stream(self, task: str):
        """流式运行多 Agent 协作"""
        if not self.leader:
            raise ValueError("请先调用 set_leader() 设置领导者")

        graph = self._build_graph()
        initial_state = State(
            messages=[
                SystemMessage(content=self._build_leader_system_prompt()),
                HumanMessage(content=task),
            ],
            metadata={},
            step=0,
        )

        final_state = initial_state
        async for event in graph.stream(initial_state):
            final_state = event.get("state", final_state)
            yield event
        if not final_state.metadata.get("_team_completed"):
            raise RuntimeError("Team execution incomplete: graph step limit reached")

    def _build_leader_system_prompt(self) -> str:
        """构建领导者的系统提示（包含成员信息）"""
        member_lines = []
        for name, member in self.members.items():
            member_lines.append(
                f"- **{name}** ({member.role}): {member.description or 'No description'}"
            )

        return f"""你是一个多 Agent 团队的协调者。

## 团队成员
{chr(10).join(member_lines)}

## 工作流程
1. 分析任务，分解为可并行执行的子任务
2. 为每个子任务指定最合适的成员
3. 汇总各成员的执行结果，形成最终输出

## 委派方式
委派任务给成员时，在回复中使用以下格式：
```
[DECISION]
assignments:
  - member: <成员名>
    task: <子任务描述>
  - member: <成员名>
    task: <子任务描述>
[/DECISION]
```
"""

    def _build_graph(self) -> CompiledGraph:
        """构建多 Agent 协作图

        节点结构:
            leader → (有分配?) → fan_out → synthesis → END
        """
        graph = StateGraph(State)
        graph.set_max_steps(self._max_steps)

        # Node 1: Leader 分析任务
        async def leader_node(state: State, config: dict | None = None) -> State:
            response = await _governed_run(
                self.leader,
                state.messages[-1].content or "",
                system_prompt=self._build_leader_system_prompt(),
            )
            ai_msg = AIMessage(content=response.content or "")
            state.add_message(ai_msg)
            state.metadata["_leader_response"] = response.content or ""
            state.metadata["_has_assignments"] = "[DECISION]" in (response.content or "")
            state.metadata["_team_completed"] = not state.metadata["_has_assignments"]
            return state

        # Node 2: 并行执行分配
        async def fan_out_node(state: State, config: dict | None = None) -> State:
            leader_response = state.metadata.get("_leader_response", "")

            # 解析分配
            assignments = self._parse_assignments(leader_response)
            if not assignments:
                raise ValueError("Leader emitted an invalid or empty assignment block")
            for assignment in assignments:
                if (
                    not isinstance(assignment, dict)
                    or assignment.get("member") not in self.members
                    or not isinstance(assignment.get("task"), str)
                ):
                    raise ValueError("Invalid team assignment")

            # 并行执行
            async def execute_one(member_name: str, sub_task: str) -> tuple[str, str]:
                member = self.members.get(member_name)
                if not member:
                    return (member_name, f"Member '{member_name}' not found")
                result = await _governed_run(
                    member.agent, sub_task, middleware=self.leader.middleware
                )
                return (member_name, result.content)

            tasks = [asyncio.create_task(execute_one(a["member"], a["task"])) for a in assignments]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException as exc:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                exc.delegation_outcomes = [
                    {
                        "member": assignment["member"],
                        "status": "unknown"
                        if isinstance(outcome, asyncio.CancelledError)
                        else "failed"
                        if isinstance(outcome, BaseException)
                        else "completed",
                        "result": None if isinstance(outcome, BaseException) else outcome[1],
                        "error": str(outcome) if isinstance(outcome, BaseException) else None,
                    }
                    for assignment, outcome in zip(assignments, outcomes, strict=True)
                ]
                raise

            # 将成员结果写入 state
            results_text = []
            for r in results:
                if isinstance(r, Exception):
                    results_text.append(f"Error: {r}")
                else:
                    member_name, content = r
                    results_text.append(f"## {member_name} 的执行结果\n\n{content}\n")

            summary = "\n\n".join(results_text)
            state.add_message(HumanMessage(content=f"以下是各成员执行结果：\n\n{summary}"))
            state.metadata["_results_ready"] = True
            return state

        # Node 3: Leader 汇总
        async def synthesis_node(state: State, config: dict | None = None) -> State:
            response = await _governed_run(
                self.leader,
                "\n\n".join(m.content or "" for m in state.messages if m.role != "system"),
                system_prompt=self._build_leader_system_prompt(),
            )
            state.add_message(AIMessage(content=response.content or ""))
            state.metadata["_team_completed"] = True
            return state

        # 条件路由
        async def route_after_leader(state: State, config: dict | None = None) -> str:
            if state.metadata.get("_has_assignments"):
                return "fan_out"
            return "end"

        async def route_after_fan_out(state: State, config: dict | None = None) -> str:
            return "synthesis"

        graph.add_node("leader", leader_node)
        graph.add_node("fan_out", fan_out_node)
        graph.add_node("synthesis", synthesis_node)

        graph.set_entry_point("leader")
        graph.add_conditional_edges(
            "leader",
            route_after_leader,
            {"fan_out": "fan_out", "end": END},
        )
        graph.add_conditional_edges(
            "fan_out",
            route_after_fan_out,
            {"synthesis": "synthesis"},
        )
        graph.add_edge("synthesis", END)

        return graph.compile()

    def _parse_assignments(self, text: str) -> list[dict]:
        """从 Leader 响应中解析任务分配"""
        if "[DECISION]" not in text:
            return []

        try:
            import re

            import yaml

            # 提取 [DECISION]...[/DECISION] 块
            match = re.search(r"\[DECISION\](.*?)\[/DECISION\]", text, re.DOTALL)
            if not match:
                return []

            block = match.group(1).strip()
            # 尝试 YAML 解析
            try:
                data = yaml.safe_load(block)
            except Exception:
                return []

            if isinstance(data, dict) and "assignments" in data:
                return data["assignments"]
            return []
        except Exception:
            return []


class HandoffMiddleware(Middleware):
    """Agent 间 Handoff 中间件

    允许 Agent 通过特殊格式将控制权交接给另一个 Agent。

    使用方式：
        Agent A 输出:
        ```
        [HANDOFF to="agent_b"]
        请处理以下任务：...
        [/HANDOFF]
        ```

        HandoffMiddleware 拦截此输出，
        将上下文转发给 Agent B，并返回 B 的响应。
    """

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents

    async def after_llm(
        self,
        ctx: MiddlewareContext,
        response: AIMessage,
    ) -> MiddlewareContext:
        """检测 Handoff 信号并转发"""
        content = response.content or ""
        if "[HANDOFF" not in content:
            return ctx

        import re

        match = re.search(
            r'\[HANDOFF\s+to="(\w+)"\](.*?)\[/HANDOFF\]',
            content,
            re.DOTALL,
        )
        if not match:
            return ctx

        target_name = match.group(1)
        handoff_content = match.group(2).strip()

        target_agent = self.agents.get(target_name)
        if not target_agent:
            return ctx

        # 将当前上下文转发给目标 Agent
        try:
            result = await _governed_run(target_agent, handoff_content)
            result_text = result.content if hasattr(result, "content") else str(result)
            # 替换 Handoff 块为实际结果
            new_content = content.replace(match.group(0), result_text)
            response.content = new_content
        except DelegationStoppedError:
            raise
        except Exception as e:
            response.content = content.replace(
                match.group(0),
                f"[Handoff to {target_name} failed: {e}]",
            )

        return ctx


# 保持向后兼容
Swarm = MultiAgentGraph  # Swarm 现在是 MultiAgentGraph 的别名
