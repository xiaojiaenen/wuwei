"""Agent 类 - 使用 Middleware + Plugin 系统"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Awaitable, Callable

from wuwei.agent.base import BaseSessionAgent
from wuwei.agent.session import AgentSession
from wuwei.llm import AgentEvent, LLMGateway
from wuwei.plugin import Plugin, PluginContext, PluginManager
from wuwei.plugin.builtin import load_all_builtin
from wuwei.runtime.agent_runner import AgentRunner
from wuwei.tools import Tool, ToolRegistry

ToolLike = Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]


class Agent(BaseSessionAgent):
    """普通单 agent 门面对象（使用 Middleware + Plugin）。"""

    def __init__(
        self,
        llm: LLMGateway,
        tools: list[Tool] | ToolRegistry | None = None,
        default_system_prompt: str = "你是一个有用的助手",
        default_max_steps: int = 10,
        default_parallel_tool_calls: bool = False,
        middleware=None,
        skill_manager=None,
        mcp_manager=None,
        plugins: list[Plugin] | None = None,
        plugin_dir: str | Path | None = None,
        load_builtins: bool = True,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            default_system_prompt=default_system_prompt,
            default_max_steps=default_max_steps,
            default_parallel_tool_calls=default_parallel_tool_calls,
            middleware=middleware,
        )

        # 保存供插件上下文使用
        self._skill_manager = skill_manager
        self._mcp_manager = mcp_manager

        # 初始化插件系统
        ctx = PluginContext(
            tool_registry=self.tool_registry,
            skill_manager=skill_manager,
            mcp_manager=mcp_manager,
            middleware_stack=self.middleware,
        )
        pm = PluginManager(ctx)

        # 仅在需要时加载内置插件
        # 当调用方已自行管理工具注册（如传入预填充的 ToolRegistry），可跳过
        if load_builtins:
            load_all_builtin(pm)

        if plugins:
            for p in plugins:
                pm.register(p)

        if plugin_dir:
            pm.load_directory(plugin_dir)

        self._plugin_manager = pm

    @property
    def plugin_manager(self) -> PluginManager:
        """返回插件管理器实例。"""
        return self._plugin_manager

    def create_runner(self, session: AgentSession) -> AgentRunner:
        """为普通 agent 会话创建执行器。"""
        return AgentRunner(
            llm=self.llm,
            tools=self.tool_registry.list_tools(),
            tool_executor=self.tool_executor,
            session=session,
            middleware=self.middleware,
        )

    async def run(
        self,
        user_input: str,
        session: AgentSession | None = None,
        stream: bool = False,
    ) -> Any:
        """运行一次普通 agent。"""
        current_session = session or self.create_or_get_session()
        runner = self.create_runner(current_session)
        return await runner.run(user_input, stream=stream)

    def stream_events(
        self,
        user_input: str,
        session: AgentSession | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """以结构化事件流运行一次普通 agent。"""
        current_session = session or self.create_or_get_session(session_id=session_id)
        runner = self.create_runner(current_session)
        return runner.stream_events(user_input)

    @classmethod
    def from_env(
        cls,
        *,
        tools: list[ToolLike] | None = None,
        system_prompt: str = "你是一个有用的助手",
        max_steps: int = 10,
        parallel_tool_calls: bool = False,
        middleware=None,
        skill_manager=None,
        mcp_manager=None,
        plugins: list[Plugin] | None = None,
        plugin_dir: str | Path | None = None,
        **llm_kwargs,
    ) -> "Agent":
        llm = LLMGateway.from_env(**llm_kwargs)
        registry = ToolRegistry()

        for item in tools or []:
            if isinstance(item, Tool):
                registry.register(item)
                continue
            if callable(item):
                registry.register_callable(item)
                continue
            raise TypeError(f"tools 只支持 Tool 或可调用对象，收到: {type(item).__name__}")

        return cls(
            llm=llm,
            tools=registry,
            default_system_prompt=system_prompt,
            default_max_steps=max_steps,
            default_parallel_tool_calls=parallel_tool_calls,
            middleware=middleware,
            skill_manager=skill_manager,
            mcp_manager=mcp_manager,
            plugins=plugins,
            plugin_dir=plugin_dir,
        )
