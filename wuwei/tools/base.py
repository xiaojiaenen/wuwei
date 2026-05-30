"""工具基类 - 支持 Pydantic Schema 自动生成

借鉴 LangChain 的 Pydantic 自动生成 + Wuwei 的简洁性。
"""

from pydantic import BaseModel, Field, create_model
from typing import Any, Callable, Optional, get_type_hints
import inspect
import json
import asyncio


class Tool(BaseModel):
    """增强版工具，支持 Pydantic Schema 自动生成"""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    handler: Callable = Field(exclude=True)
    requires_approval: bool = False
    side_effect: bool = False
    timeout_seconds: float = 60.0

    # 注入参数支持（借鉴 LangChain）
    injected_params: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: str = None,
        description: str = None,
        injected_params: list[str] = None,
    ) -> "Tool":
        """从函数自动生成 Schema

        示例：
            @tool
            def search(query: str, max_results: int = 5) -> str:
                '''搜索文档'''
                ...

            tool = Tool.from_function(search)
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        # 自动生成 Pydantic 模型
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in (injected_params or []):
                continue  # 跳过注入参数
            param_type = hints.get(param_name, str)
            param_default = param.default if param.default != inspect.Parameter.empty else ...
            fields[param_name] = (param_type, param_default)

        schema_model = create_model(f"{func.__name__}_args", **fields)

        return cls(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            parameters=cls._pydantic_to_json_schema(schema_model),
            handler=func,
            injected_params=injected_params or [],
        )

    @staticmethod
    def _pydantic_to_json_schema(model: type[BaseModel]) -> dict:
        """Pydantic 模型转 JSON Schema"""
        schema = model.model_json_schema()
        # 转换为 OpenAI function calling 格式
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, args: dict, config: dict = None) -> str:
        """执行工具"""
        # 过滤注入参数
        filtered_args = {
            k: v for k, v in args.items()
            if k not in self.injected_params
        }

        # Pydantic 验证
        if self.parameters:
            self._validate_args(filtered_args)

        # 执行
        try:
            if asyncio.iscoroutinefunction(self.handler):
                result = await asyncio.wait_for(
                    self.handler(**filtered_args),
                    timeout=self.timeout_seconds,
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.handler(**filtered_args),
                )
            return str(result)
        except asyncio.TimeoutError:
            raise TimeoutError(f"工具 {self.name} 执行超时 ({self.timeout_seconds}s)")
        except Exception as e:
            raise ToolError(f"工具 {self.name} 执行失败: {e}")

    def _validate_args(self, args: dict):
        """参数验证"""
        required = self.parameters.get("required", [])
        for param in required:
            if param not in args:
                raise ValueError(f"缺少必需参数: {param}")


# 便捷函数
def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    injected_params: list[str] = None,
):
    """工具装饰器

    示例：
        @tool
        def search(query: str) -> str:
            '''搜索文档'''
            return "结果"

        @tool(name="custom_name", description="自定义描述")
        def my_func(x: int) -> str:
            return str(x)
    """
    def decorator(f: Callable) -> Tool:
        return Tool.from_function(
            f,
            name=name,
            description=description,
            injected_params=injected_params,
        )

    if func is not None:
        # @tool 无参数
        return decorator(func)
    else:
        # @tool(name="...", ...)
        return decorator
