from typing import Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    """单个规划任务节点。"""

    id: int = Field(description="任务 ID，必须唯一。")
    description: str = Field(description="任务描述。")
    next: list[int] = Field(description="下游任务 ID 列表。")
    status: Literal[
        "pending",
        "in_progress",
        "completed",
        "failed",
        "blocked",
    ] = Field(description="任务状态。")
    result: str | None = Field(default=None, description="任务执行结果。")
    error: str | None = Field(default=None, description="任务失败信息。")


class TaskList(BaseModel):
    """任务列表包装对象。"""

    tasks: list[Task] = Field(default_factory=list, description="任务列表。")


class PlanRunResult(BaseModel):
    """一次 plan-and-execute 运行的汇总结果。"""

    tasks: list[Task] = Field(default_factory=list)
    usage: dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    latency_ms: int = 0
    llm_calls: int = 0
    planner_usage: dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    planner_latency_ms: int = 0
    planner_llm_calls: int = 0
    execution_usage: dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    execution_latency_ms: int = 0
    execution_llm_calls: int = 0
