from typing import Any, AsyncIterator

from wuwei.core.base import BaseAgent, AgentConfig
from wuwei.core.context import Context
from wuwei.llm import LLMGateway, LLMResponse, LLMResponseChunk
from wuwei.tools import ToolRegistry, Tool


class Agent(BaseAgent):
    def __init__(self, llm: LLMGateway, tools: list | ToolRegistry | None = None,config: AgentConfig=None):
        self.llm = llm
        self.config = config
        # 处理工具
        # if tools:
        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
            self.tools = tools.list_tools()
        elif tools is None:
            self.tools = []
        else:
            # 创建ToolRegistry并注册工具
            self.tool_registry = ToolRegistry()
            for tool in tools:
                self.tool_registry.register(tool)
            self.tools = self.tool_registry.list_tools()

        # 初始化上下文
        self.context = Context()
        # 添加默认系统消息
        self.context.add_system_message(self.config.system_prompt)




    async def run(self, user_input: str, stream: bool=False) -> Any:
        if stream:
            # print("streaming...")
            return self._run_stream(user_input)
        else:
            return await self._run_non_stream(user_input)

    async def _run_non_stream(self, user_input):
        step_count=0
        self.context.add_user_message(user_input)
        while step_count < self.config.max_steps:
            response:LLMResponse = await self.llm.generate(
                self.context.get_messages(),
                tools=self.tools,
            )
            # 添加AI响应到上下文
            self.context.add_ai_message(
                response.message.content,
                response.message.tool_calls
            )

            if response.finish_reason=="tool_calls" and response.message.tool_calls:
                for tc in response.message.tool_calls:
                    tool_name=tc.function.name
                    tool_args=tc.function.arguments
                    tool:Tool=self.tool_registry.get(tool_name)
                    if tool:
                        try:
                            result = await tool.invoke(tool_args)
                            print(f"\n[工具执行结果] {tc.function.name}: {str(result)}\n")
                            self.context.add_tool_message(result,tc.id)
                        except Exception as e:
                            error_message=f"工具 {tool_name} 执行错误: {str(e)}"
                            self.context.add_tool_message(error_message,tc.id)
                    else:
                        error_message = f"工具 {tool_name} 不存在"
                        self.context.add_tool_message(error_message, tc.id)
            else:
                return response.message.content
            step_count+=1
        limit_message = "任务未完成，已达到最大步骤限制"
        self.context.add_ai_message(limit_message)
        return limit_message

    async def _run_stream(self, user_input):
        step_count = 0
        self.context.add_user_message(user_input)
        # print(self.context.get_messages())

        while step_count < self.config.max_steps:
            # print(f"第 {step_count} 步")
            full_content = ""
            full_tool_calls = None

            stream: AsyncIterator[LLMResponseChunk] = await self.llm.generate(
                self.context.get_messages(),
                tools=self.tools,
                stream=True,
            )

            # 1. 收集流式响应，同时逐块输出普通内容
            async for chunk in stream:
                full_content += chunk.content
                if chunk.tool_calls_complete:
                    full_tool_calls = chunk.tool_calls_complete
                # 普通文本块（非工具调用结束块）直接输出
                if chunk.content:
                    yield chunk

            # 2. 流式响应结束，将完整的 assistant 消息（含 tool_calls）加入历史
            self.context.add_ai_message(full_content, tool_calls=full_tool_calls)

            # 3. 处理工具调用（如果有）
            if full_tool_calls:
                for tc in full_tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    tool: Tool = self.tool_registry.get(tool_name)
                    if tool:
                        try:
                            result = await tool.invoke(tool_args)
                            result_str = str(result)
                            print(f"\n[工具执行结果] {tool_name}: {result_str}\n")
                            self.context.add_tool_message(result_str, tc.id)
                        except Exception as e:
                            error_msg = f"工具 {tool_name} 执行错误: {str(e)}"
                            yield f"\n[工具执行错误] {error_msg}\n"
                            self.context.add_tool_message(error_msg, tc.id)
                    else:
                        error_msg = f"工具 {tool_name} 不存在"
                        yield f"\n[工具错误] {error_msg}\n"
                        self.context.add_tool_message(error_msg, tc.id)
                # 有工具调用，继续循环让模型处理工具结果
                step_count += 1
                continue
            else:
                # 无工具调用，模型已给出最终答案，结束循环
                break