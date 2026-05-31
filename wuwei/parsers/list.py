"""列表输出解析器"""

import json
from typing import Any
from wuwei.parsers.base import BaseOutputParser


class ListOutputParser(BaseOutputParser):
    """列表输出解析器

    支持：
    - JSON 数组解析
    - 逗号分隔列表
    - 换行分隔列表

    示例：
        parser = ListOutputParser()
        result = parser.parse("item1, item2, item3")
        # result = ["item1", "item2", "item3"]
    """

    def __init__(self, delimiter: str = None):
        """
        Args:
            delimiter: 分隔符（None 表示自动检测）
        """
        self.delimiter = delimiter

    def parse(self, output: str) -> list:
        """解析列表输出"""
        if not output:
            return []

        # 尝试解析 JSON 数组
        try:
            data = json.loads(output)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 使用分隔符分割
        if self.delimiter:
            return [item.strip() for item in output.split(self.delimiter) if item.strip()]

        # 自动检测分隔符
        if "\n" in output:
            return [item.strip() for item in output.split("\n") if item.strip()]
        elif "," in output:
            return [item.strip() for item in output.split(",") if item.strip()]
        else:
            return [output.strip()]

    def get_format_instructions(self) -> str:
        """获取格式说明"""
        return "请以列表格式输出结果，每行一个项目，或用逗号分隔。"
