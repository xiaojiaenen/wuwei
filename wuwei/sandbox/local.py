"""本地沙箱"""

import asyncio
import os
from wuwei.sandbox.base import BaseSandbox, SandboxResult


class LocalSandbox(BaseSandbox):
    """本地沙箱

    在本地环境中执行命令。

    示例：
        async with LocalSandbox() as sandbox:
            result = await sandbox.execute("ls -la")
            print(result.stdout)
    """

    def __init__(self, workspace: str = None):
        """
        Args:
            workspace: 工作目录，默认当前目录
        """
        self.workspace = workspace or os.getcwd()

    async def execute(
        self,
        command: str,
        timeout: float = 30,
        cwd: str = None,
    ) -> SandboxResult:
        """执行命令"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or self.workspace,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return SandboxResult(
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                exit_code=process.returncode,
                success=process.returncode == 0,
            )

        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="",
                stderr=f"命令执行超时 ({timeout}s)",
                exit_code=-1,
                success=False,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                success=False,
            )

    async def cleanup(self):
        """清理资源"""
        pass
