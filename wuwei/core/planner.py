from wuwei.core.task import Task, TaskList
from wuwei.llm import LLMGateway, Message, LLMResponse


class Planner:
    def __init__(self, llm: LLMGateway):
        self.task: list[Task] = []
        self.llm = llm

    def _build_plan_prompt(self, goal: str):
        prompt = f"""
        # Role
        你是一个高级工作流规划引擎，擅长将复杂目标拆解为可执行的 DAG（有向无环图）任务流。
        
        # Goal
        请为以下目标制定执行计划：
        {goal}
        
        # Planning Rules
        1. 原子性
        每个任务必须是一个独立、可执行、可验证的最小动作，不要把多个动作合并成一个任务。
        
        2. DAG 约束
        任务之间必须是有向无环图，禁止循环依赖。
        
        3. 依赖表达
        使用 `next` 表示当前任务完成后可以进入的下游任务 ID 列表。
        如果任务 B 依赖任务 A 的结果，则任务 A 的 `next` 必须包含 B 的 ID。
        
        4. 并行分支
        只有在任务之间不存在结果依赖时，才允许并行分支。
        如果存在多个可并行分支，但系统只允许一个初始运行节点，请先创建一个唯一的根任务，由该根任务扇出到多个分支。
        
        5. 起始任务
        必须且只能有一个起始任务，其 `status` 设为 `"in_progress"`。
        其余任务的 `status` 必须为 `"pending"`。
        
        6. 初始化字段
        在规划阶段：
        - `result` 必须初始化为 `null`
        - `error` 必须初始化为 `null`
        
        7. 数量限制
        任务数必须在 2 到 5 个之间，除非目标极其简单，否则不要只生成 1 个任务。
        
        8. 描述要求
        `description` 必须是可直接交给执行器的明确指令，避免空泛表述，如“处理问题”“继续执行”。
        
        # Output Schema
        必须返回一个 JSON 对象，外层只有一个 key：`tasks`。
        `tasks` 是数组，每个元素严格遵循以下结构：
        
        - `id` (integer): 任务 ID，必须从 1 开始连续递增
        - `description` (string): 任务的具体执行指令
        - `next` (array of integers): 下游任务 ID 列表；叶子节点必须为 []
        - `status` (string): 仅允许以下值之一：
          `pending`, `in_progress`, `completed`, `failed`, `blocked`
        - `result` (null): 规划阶段固定为 null
        - `error` (null): 规划阶段固定为 null
        
        # Validation Rules
        生成结果前请自行检查：
        1. 是否只有一个 `in_progress`
        2. 是否所有任务 ID 唯一且连续
        3. 是否所有 `next` 中的 ID 都真实存在
        4. 是否至少有一个叶子节点
        5. 是否不存在环
        6. 是否所有 `result` 和 `error` 都是 null
        
        # Output Constraint
        只输出纯 JSON，不要输出 Markdown，不要输出解释，不要输出额外文本。

"""
        return prompt

    async def plan_task(self,goal:str) -> list[Task]:
        response: LLMResponse = await self.llm.generate(
            messages=[
                Message(role="user", content=self._build_plan_prompt(goal))
            ],
            stream=False,
            response_format={"type": "json_object"}
        )
        content = response.message.content
        wrapper = TaskList.model_validate_json(content)
        return wrapper.tasks

    @classmethod
    def create_planner(cls, llm: LLMGateway) -> "Planner":
        return cls(llm)
