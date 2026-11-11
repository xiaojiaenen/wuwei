"""受治理子代理的单一构建点。

sub_agent / multi_agent / planning（PlanExecutor 与内置规划）此前各自构造
``Agent``，继承规则（deps 通道、中间件排除、event_bus 透传、深度防护）各写
各的并已实际漂移——PlanExecutor 的任务事件对总线不可见，multi_agent 不传
deps，只有 sub_agent 排除自身防自委派。本工厂把构造收敛为一处，各调用方以
参数声明自己的策略；策略差异保留，实现差异清零。
"""

from __future__ import annotations

from typing import Any

UNSET = object()
"""区分「不传（用 Agent 内置默认）」与「显式传 None（清除默认）」。"""


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


def build_child_agent(
    llm,
    *,
    tools=None,
    system_prompt: Any = UNSET,
    max_steps: Any = UNSET,
    parent_middleware=None,
    child_middleware=None,
    exclude_middleware: Any = None,
    checkpointer=None,
    event_bus=None,
    deps=None,
) -> Any:
    """构建一个继承父级治理的子 ``Agent``。

    Args:
        system_prompt/max_steps: 哨兵 ``UNSET`` 表示沿用 Agent 内置默认；显式
            传 ``None`` 表示清除默认（保持 sub_agent 既有透传语义）。
        parent_middleware: 父级活动上下文里的中间件栈/列表（治理继承）。
        child_middleware: 子代理自身的额外中间件。
        exclude_middleware: 排除的实例（通常是发起委派的中间件，防自委派环）。
        deps: 传给子代理的 RunContext.deps；不传则子代理无 deps 通道。
        event_bus: 事件总线；计划任务此前丢失透传，此处统一接入。
    """
    from wuwei.agent.agent import Agent

    kwargs = {}
    if system_prompt is not UNSET:
        kwargs["default_system_prompt"] = system_prompt
    if max_steps is not UNSET:
        kwargs["default_max_steps"] = max_steps
    return Agent(
        llm=llm,
        tools=tools or [],
        load_builtins=False,
        middleware=inherit_middleware(
            parent_middleware, child_middleware, exclude=exclude_middleware
        ),
        checkpointer=checkpointer,
        event_bus=event_bus,
        deps=deps,
        **kwargs,
    )
