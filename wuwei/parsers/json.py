"""JSON 输出解析器"""

import json
from typing import Any
from wuwei.parsers.base import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    """JSON 输出解析器

    支持：
    - 自动提取 JSON 内容
    - 验证 JSON 格式
    - 处理流式输出

    示例：
        parser = JsonOutputParser()
        result = parser.parse('{"name": "test", "value": 123}')
        # result = {"name": "test", "value": 123}
    """

    def parse(self, output: str) -> Any:
        """解析 JSON 输出"""
        if not output:
            return None

        # 尝试直接解析
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # 尝试从文本中提取 JSON
        json_str = self._extract_json(output)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 返回原始文本
        return output

    def _extract_json(self, text: str) -> str | None:
        """从文本中提取 JSON"""
        # 查找 JSON 块
        import re

        # 匹配 ```json ... ``` 或 ``` ... ```
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # 匹配 { ... } 或 [ ... ]
        json_object_pattern = r'\{[^{}]*\}'
        json_array_pattern = r'\[.*?\]'

        for pattern in [json_object_pattern, json_array_pattern]:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # 返回最长的匹配
                return max(matches, key=len)

        return None

    def get_format_instructions(self) -> str:
        """获取格式说明"""
        return "请以 JSON 格式输出结果。"
