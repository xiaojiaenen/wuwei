from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    async def run(self, user_input: str, session: Any | None = None, stream: bool = False) -> Any:
        pass
