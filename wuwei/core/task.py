from typing import Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    id:int=Field(description="任务ID（唯一标识）")
    description:str=Field(description="任务描述")
    next:list[int]=Field(description="后续任务ID列表")
    status:Literal[
        "pending",
        "in_progress",
        "completed",
        "failed",
        "blocked",
    ]=Field(description="任务状态")
    result: str | None = Field(default=None, description="任务执行结果")
    error: str | None = Field(default=None, description="任务失败信息")

class TaskList(BaseModel):
    tasks:list[Task]=Field(default_factory=list,description="任务列表")


