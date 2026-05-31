"""Pydantic 输出解析器"""

import json
from typing import Any, Type
from pydantic import BaseModel, ValidationError
from wuwei.parsers.base import BaseOutputParser
from wuwei.parsers.json import JsonOutputParser


class PydanticOutputParser(BaseOutputParser):
    """Pydantic 输出解析器

    支持：
    - 自动验证 Pydantic 模型
    - 类型转换
    - 详细的错误信息

    示例：
        class User(BaseModel):
            name: str
            age: int

        parser = PydanticOutputParser(schema=User)
        result = parser.parse('{"name": "test", "age": 25}')
        # result = User(name="test", age=25)
    """

    def __init__(self, schema: Type[BaseModel]):
        """
        Args:
            schema: Pydantic 模型类
        """
        self.schema = schema
        self.json_parser = JsonOutputParser()

    def parse(self, output: str) -> Any:
        """解析并验证 Pydantic 模型"""
        # 先解析 JSON
        json_data = self.json_parser.parse(output)

        if isinstance(json_data, str):
            # 如果还是字符串，尝试再次解析
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError:
                raise ValueError(f"无法解析为 JSON: {output}")

        if isinstance(json_data, dict):
            # 验证 Pydantic 模型
            try:
                return self.schema.model_validate(json_data)
            except ValidationError as e:
                raise ValueError(f"验证失败: {e}")

        return json_data

    def get_format_instructions(self) -> str:
        """获取格式说明"""
        schema_json = json.dumps(
            self.schema.model_json_schema(),
            indent=2,
            ensure_ascii=False,
        )
        return f"请以以下 JSON 格式输出结果：\n```json\n{schema_json}\n```"
