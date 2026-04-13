from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from wuwei.agent.session import AgentSession
from wuwei.llm import LLMGateway
from wuwei.runtime.hooks import RuntimeHook, HookManager
from wuwei.tools import Tool, ToolExecutor, ToolRegistry


class BaseAgent(ABC):
    """所有 agent 的最小抽象基类。"""

    @abstractmethod
    async def run(self, user_input: str, session: Any | None = None, stream: bool = False) -> Any:
        """运行一次 agent。"""


class BaseSessionAgent(BaseAgent):
    """
    带会话能力的公共基类。

    这个类负责收敛 Agent 和 PlanAgent 里重复的公共逻辑：
    - llm / tools / tool_executor 初始化
    - 默认 system_prompt / max_steps / parallel_tool_calls
    - session 的创建与复用
    """

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        default_system_prompt: str = "你是一个有用的助手",
        default_max_steps: int = 10,
        default_parallel_tool_calls: bool = False,
        hooks: list[RuntimeHook] | HookManager | None = None,
    ) -> None:
        """初始化公共依赖和默认会话配置。"""
        self.llm = llm
        self.default_system_prompt = default_system_prompt
        self.default_max_steps = default_max_steps
        self.default_parallel_tool_calls = default_parallel_tool_calls

        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        else:
            self.tool_registry = ToolRegistry()
            for tool in tools or []:
                self.tool_registry.register(tool)

        self.tool_executor = ToolExecutor(self.tool_registry)
        self.hooks = hooks if isinstance(hooks, HookManager) else HookManager(hooks)
        self._sessions: dict[str, AgentSession] = {}

    def create_session(
        self,
        session_id: str | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AgentSession:
        """创建一个新会话，并写入默认 system prompt。"""
        session = AgentSession(
            session_id=session_id or uuid4().hex,
            system_prompt=system_prompt or self.default_system_prompt,
            max_steps=max_steps if max_steps is not None else self.default_max_steps,
            parallel_tool_calls=(
                parallel_tool_calls
                if parallel_tool_calls is not None
                else self.default_parallel_tool_calls
            ),
        )
        self._sessions[session.session_id] = session
        return session

    def create_or_get_session(
        self,
        session_id: str | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AgentSession:
        """按 session_id 复用会话；如果不存在则新建。"""
        if session_id is not None and session_id in self._sessions:
            return self._sessions[session_id]

        return self.create_session(
            session_id=session_id,
            system_prompt=system_prompt,
            max_steps=max_steps,
            parallel_tool_calls=parallel_tool_calls,
        )

    @abstractmethod
    def create_runner(self, session: AgentSession) -> Any:
        """子类负责返回对应的 runner。"""
