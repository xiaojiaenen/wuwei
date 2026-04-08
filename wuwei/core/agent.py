from typing import Any

from wuwei.core.base import BaseAgent
from wuwei.llm import LLMGateway
from wuwei.tools import ToolRegistry, Tool


class Agent(BaseAgent):
    def __init__(self,llm:LLMGateway,tools:ToolRegistry,context:Context):
        self.llm = llm
        self.tools = tools.to_schema()


    async def run(self, user_input: str) -> Any:
        pass

