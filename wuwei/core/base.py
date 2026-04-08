from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Agent(BaseModel):
    name: str="WUWEI AGENT"
    system_prompt="你是一个有用的助手"
    max_steps: int=10

class BaseAgent(ABC):
    @abstractmethod
    async def run(self,user_input:str)->Any:
        pass