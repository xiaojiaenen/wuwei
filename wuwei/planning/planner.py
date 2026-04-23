from wuwei.llm import LLMGateway, LLMResponse, Message
from wuwei.planning.task import Task, TaskList


class Planner:
    """负责把复杂目标拆解为 DAG 任务列表。"""

    def __init__(self, llm: LLMGateway) -> None:
        self.llm = llm
        self.last_usage = self._empty_usage()
        self.last_latency_ms = 0
        self.last_llm_calls = 0

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _build_plan_prompt(self, goal: str) -> str:
        """构造任务规划提示词。"""
        return f"""
# Role
你是一个高级工作流规划引擎，擅长把复杂目标拆解为具有依赖关系的 DAG（有向无环图）任务流。

# Goal
请为以下目标制定执行计划：
{goal}

# Planning Rules
1. 每个任务都必须是独立、可执行、可验证的最小动作。
2. 任务之间必须保持 DAG 结构，禁止循环依赖。
3. 使用 `next` 表示当前任务完成后可以进入的下游任务 ID 列表。
4. 如果任务 B 依赖任务 A 的结果，则任务 A 的 `next` 必须包含 B 的 ID。
5. 只有真正互不依赖的任务才允许并行。
6. 必须且只能有一个起始任务，它的 `status` 为 `"in_progress"`。
7. 其他任务的 `status` 必须为 `"pending"`。
8. 规划阶段所有任务的 `result` 和 `error` 都必须为 `null`。
9. 默认输出 2 到 5 个任务；只有目标非常简单时，才允许少于 2 个任务。
10. `description` 必须可以直接交给执行器执行，不要使用空泛表述。

# Output Schema
返回一个 JSON 对象，外层只有一个 key：`tasks`。

每个任务必须包含：
- `id`: 从 1 开始递增的整数
- `description`: 任务描述
- `next`: 下游任务 ID 列表
- `status`: 只能是 `pending`、`in_progress`、`completed`、`failed`、`blocked`
- `result`: 固定为 null
- `error`: 固定为 null

# Output Constraint
只输出纯 JSON，不要输出解释，不要输出 Markdown。
""".strip()

    async def plan_task(self, goal: str) -> list[Task]:
        """调用模型生成任务计划。"""
        self.last_usage = self._empty_usage()
        self.last_latency_ms = 0
        self.last_llm_calls = 0

        response: LLMResponse = await self.llm.generate(
            messages=[Message(role="user", content=self._build_plan_prompt(goal))],
            stream=False,
            response_format={"type": "json_object"},
        )
        self.last_usage = dict(response.usage)
        self.last_latency_ms = response.latency_ms
        self.last_llm_calls = 1
        content = response.message.content
        if not content:
            raise ValueError("Planner 未返回任何内容")

        wrapper = TaskList.model_validate_json(content)
        return wrapper.tasks

    @classmethod
    def create_planner(cls, llm: LLMGateway) -> "Planner":
        """基于给定 llm 创建默认 planner。"""
        return cls(llm)
