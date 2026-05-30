"""沙箱基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    stdout: str
    stderr: str
    exit_code: int
    success: bool

    def __str__(self) -> str:
        if self.success:
            return self.stdout
        return f"Error (exit code {self.exit_code}):\n{self.stderr}"


class BaseSandbox(ABC):
    """沙箱基类"""

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: float = 30,
        cwd: str = None,
    ) -> SandboxResult:
        """执行命令"""
        ...

    @abstractmethod
    async def cleanup(self):
        """清理资源"""
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
