import asyncio
import json

from wuwei.core.task import Task, TaskList
from wuwei.llm import LLMGateway, Message, LLMResponse


class Planner:
    def __init__(self, goal: str, llm: LLMGateway):
        self.goal = goal
        self.task: list[Task] = []
        self.llm = llm


    def build_plan_prompt(self, goal: str):
        prompt = f"""
# Role
你是一个高级工作流规划引擎，擅长将复杂目标拆解为具有依赖关系的 DAG（有向无环图）任务流。
# Task
请为以下目标制定执行计划：{goal}
# Rules
1. 原子性：每个任务必须是一个独立、可执行的最小粒度动作。
2. 依赖逻辑：仔细分析任务间的先后顺序。如果任务 B 必须等任务 A 的结果，任务 A 的 "next" 必须包含 B 的 ID。相互独立的任务应该形成并行分支。
3. 状态初始化：默认所有任务 status 为 "pending"。只有计划中的第一个起始任务，其 status 设为 "in_progress"。
4. 不超过5个任务。
# Format
必须返回一个 JSON 字符串，外层key为'tasks'，内层为数组，每个元素严格遵循以下 Schema：
- `id` (integer): 任务ID，从1开始。
- `description` (string): 任务的具体执行指令。
- `next` (array of integers): 该任务依赖的下游任务ID列表。叶子节点（最后一步）必须为空数组 []。
- `status` (string): 枚举值，仅限 ["pending", "in_progress", "completed"]。
# Output Constraint
纯净输出 JSON，禁止输出任何非 JSON 字符。
"""
        return prompt


    async def plan_task(self) -> list[Task]:
        response:LLMResponse=await self.llm.generate(
            messages=[
                Message(role="user", content=self.build_plan_prompt(self.goal))
            ],
            stream=False,
            response_format={"type": "json_object"}
        )
        content = response.message.content
        wrapper = TaskList.model_validate_json(content)
        return wrapper.tasks


if __name__ == '__main__':
    llm_config = {
        "provider": "openai",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-542fb522c8e94185b99fb010307a00fe",  # 替换为实际的API密钥
        "model": "deepseek-chat",
        "temperature": 0.2
    }
    llm = LLMGateway(llm_config)
    planner = Planner("读取当前目录下的所有文件", llm)
    response = asyncio.run(planner.plan_task())

    print(response)


