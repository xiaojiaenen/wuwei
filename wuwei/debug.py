"""Time-Travel 调试器 — 步骤级反事实重放

基于检查点创建独立调试分支（默认不执行真实工具）：

- `list_steps()`：浏览一次 run 的所有检查点（每步一个，含类型与内容摘要）
- `fork()`：从任意检查点分叉出新 run——可改工具参数、替换系统提示、
  甚至从 final 检查点重放（同上下文再跑一次，观察非确定性分歧）
- 轨迹 diff：fork 前后各自的消息序列对比（`diff_messages`）

用法：
    agent = Agent(llm=llm, checkpointer=SQLiteCheckpointer("./state"))
    result = await agent.run(task)          # 每步自动落检查点
    steps = await list_steps(agent.checkpointer)
    # 发现第 2 步工具参数错了：
    result2 = await agent.fork(
        steps[1]["id"],
        overrides={"tool_call_0": {"arguments": {"q": "修正后的查询"}}},
    )

CLI：
    python -m wuwei.debug ./state.db            # 列出检查点
    python -m wuwei.debug ./state.db --show <checkpoint_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from wuwei.core.message import FunctionCall, ToolCall
from wuwei.graph.state import State
from wuwei.runtime.interrupt import InterruptPayload


async def list_steps(checkpointer, limit: int = 100) -> list[dict[str, Any]]:
    """按时间顺序列出检查点摘要（新→旧）。"""
    records = await checkpointer.list_checkpoints(limit=limit)
    steps = []
    for record in records:
        metadata = record.get("metadata") or {}
        state = await checkpointer.load(record["id"])
        last_content = ""
        if state and state.messages:
            last = state.messages[-1]
            last_content = f"[{last.role}] {(getattr(last, 'content', '') or '')[:120]}"
        steps.append(
            {
                "id": record["id"],
                "kind": metadata.get("kind", "?"),
                "step": metadata.get("step", record.get("step", 0)),
                "run_id": metadata.get("run_id"),
                "pending_tools": [
                    tc.get("function", {}).get("name", "?")
                    if isinstance(tc, dict)
                    else getattr(getattr(tc, "function", None), "name", "?")
                    for tc in metadata.get("pending_tool_calls", [])
                ],
                "last_message": last_content,
                "created_at": record.get("timestamp"),
            }
        )
    return steps


def apply_overrides(state: State, overrides: dict[str, Any]) -> State:
    """把反事实修改应用到检查点状态。

    支持的 override 键：
    - "tool_call_<index>": {"name": ..., "arguments": {...}} 修改 pending 工具调用
    - "system_prompt": str 替换首条 system 消息
    - "append_user": str 在末尾追加一条用户消息（引导新方向）
    """
    metadata = dict(state.metadata or {})
    pending = [
        ToolCall(**tc) if isinstance(tc, dict) else tc
        for tc in metadata.get("pending_tool_calls", [])
    ]

    for key, value in overrides.items():
        if key.startswith("tool_call_") and isinstance(value, dict):
            index = int(key.rsplit("_", 1)[1])
            if 0 <= index < len(pending):
                tc = pending[index]
                name = value.get("name", tc.function.name)
                arguments = value.get("arguments", tc.function.arguments)
                pending[index] = ToolCall(
                    id=tc.id, type="function", function=FunctionCall(name=name, arguments=arguments)
                )
        elif key == "system_prompt" and state.messages:
            from wuwei.core.message import SystemMessage

            if state.messages[0].role == "system":
                state.messages[0] = SystemMessage(content=str(value))
            else:
                state.messages.insert(0, SystemMessage(content=str(value)))
        elif key == "append_user":
            from wuwei.core.message import HumanMessage

            state.messages.append(HumanMessage(content=str(value)))

    metadata["pending_tool_calls"] = [tc.model_dump() for tc in pending]
    replacements = {tc.id: tc for tc in pending}
    for message in state.messages:
        if message.tool_calls:
            message.tool_calls = [replacements.get(tc.id, tc) for tc in message.tool_calls]
    state.metadata = metadata
    return state


def diff_messages(a: list, b: list) -> dict[str, Any]:
    """对比两条轨迹的消息序列，返回分歧点。"""
    len_a, len_b = len(a), len(b)
    common = 0
    for x, y in zip(a, b):
        if x.role != y.role or (x.content or "") != (y.content or ""):
            break
        common += 1
    return {
        "length_a": len_a,
        "length_b": len_b,
        "diverge_at": common if common < min(len_a, len_b) else None,
        "rest_a": [f"[{m.role}] {(m.content or '')[:100]}" for m in a[common:]],
        "rest_b": [f"[{m.role}] {(m.content or '')[:100]}" for m in b[common:]],
    }


# ── Agent 扩展 ───────────────────────────────────────────────────────


async def agent_fork(
    agent,
    checkpoint_id: str,
    *,
    overrides=None,
    reason="time-travel fork",
    allow_side_effects=False,
):
    """Create an isolated run. Tool execution is simulated unless explicitly enabled."""
    from uuid import uuid4
    from wuwei.middleware import Middleware, MiddlewareStack
    from wuwei.core.message import ToolMessage

    if agent.checkpointer is None:
        raise ValueError("fork requires checkpointer")
    state = await agent.checkpointer.load(checkpoint_id)
    if overrides:
        state = apply_overrides(state, overrides)
    original_session = state.metadata.get("session") or {}
    session = agent.create_session(
        system_prompt=original_session.get("system_prompt"),
        max_steps=original_session.get("max_steps"),
        parallel_tool_calls=original_session.get("parallel_tool_calls"),
    )
    session.context.replace_messages([m.model_copy(deep=True) for m in state.messages])
    run_id = uuid4().hex
    old_run = state.metadata.get("run_state") or {}
    state.metadata.update(
        kind="fork",
        session_id=session.session_id,
        run_id=run_id,
        parent_checkpoint_id=checkpoint_id,
        session=session.to_dict(),
        step=0,
        run_state={
            "run_id": run_id,
            "session_id": session.session_id,
            "step_index": 0,
            "usage_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "llm_calls": 0,
            "latency_ms": 0,
            "permission_state": dict(old_run.get("permission_state", {})),
            "approval_decisions": {},
            "tool_states": {},
            "usage_records": [],
            "status": "running",
        },
    )
    state.step = 0
    state.metadata.pop("continuation", None)
    state.metadata.pop("approval_call", None)
    fork_id = await agent.checkpointer.save(state, metadata=state.metadata)

    class ReplayGuard(Middleware):
        async def before_tool(self, ctx, tc):
            return ToolMessage(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "type": "replay_simulated",
                            "message": "调试分支未执行真实工具；显式allow_side_effects=True才能执行",
                        },
                    }
                ),
                tool_call_id=tc.id,
                name=tc.function.name,
                status="error",
            )

    runner = agent.create_runner(session)
    if not allow_side_effects:
        runner.middleware = MiddlewareStack(*agent.middleware.middlewares, ReplayGuard())
    payload = InterruptPayload(
        checkpoint_id=fork_id,
        session_id=session.session_id,
        tool_name="fork",
        reason=reason,
        run_id=run_id,
    )
    return await runner.resume(payload, state=state)


def _print_steps(steps: list[dict[str, Any]]) -> None:
    for s in steps:
        pending = f" pending={','.join(s['pending_tools'])}" if s["pending_tools"] else ""
        print(f"{s['id']}  kind={s['kind']:<10} step={s['step']}{pending}")
        if s["last_message"]:
            print(f"    └─ {s['last_message']}")


def main(argv: list[str] | None = None) -> int:
    """CLI：python -m wuwei.debug <sqlite_path> [--show <id>]"""
    parser = argparse.ArgumentParser(prog="wuwei.debug", description="Time-Travel 检查点浏览器")
    parser.add_argument("db", help="SQLiteCheckpointer 数据库路径")
    parser.add_argument("--summary", help="只显示运行状态、用量与工具状态，不显示正文/参数")
    parser.add_argument("--show", help="展示某个检查点的完整消息序列")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    from wuwei.graph.checkpoint import SQLiteCheckpointer

    async def _run() -> int:
        cp = SQLiteCheckpointer(args.db)
        if args.summary:
            state = await cp.load(args.summary)
            meta = state.metadata or {}
            run = meta.get("run_state") or {}
            safe = {
                key: run[key]
                for key in (
                    "schema_version",
                    "run_id",
                    "session_id",
                    "parent_run_id",
                    "trace_id",
                    "status",
                    "step_index",
                    "llm_calls",
                    "latency_ms",
                    "usage_total",
                    "usage_available",
                    "tool_states",
                )
                if key in run
            }
            safe.update(checkpoint_id=args.summary, kind=meta.get("kind"))
            print(json.dumps(safe, ensure_ascii=False, indent=2))
            return 0
        if args.show:
            state = await cp.load(args.show)
            if state is None:
                print(f"检查点不存在: {args.show}")
                return 1
            meta = state.metadata or {}
            print(f"checkpoint {args.show}  kind={meta.get('kind')}  step={state.step}")
            for m in state.messages:
                preview = (m.content or "").replace("\n", " ")[:200]
                print(f"  [{m.role}] {preview}")
                for tc in getattr(m, "tool_calls", None) or []:
                    print(
                        f"    ↳ tool_call {tc.function.name}({json.dumps(tc.function.arguments, ensure_ascii=False)[:120]})"
                    )
            return 0
        steps = await list_steps(cp, limit=args.limit)
        if not steps:
            print("（无检查点）")
            return 0
        _print_steps(steps)
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
