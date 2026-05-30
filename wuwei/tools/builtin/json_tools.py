"""JSON 处理工具"""

import json
from wuwei.tools.base import Tool


@Tool.from_function
def json_parse(text: str) -> str:
    """解析 JSON 字符串

    Args:
        text: JSON 字符串
    """
    try:
        result = json.loads(text)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {e}"


@Tool.from_function
def json_extract(data: str, path: str) -> str:
    """从 JSON 中提取指定路径的值

    Args:
        data: JSON 字符串
        path: 点号分隔的路径，如 "user.name"
    """
    try:
        obj = json.loads(data)
        keys = path.split(".")
        for key in keys:
            if isinstance(obj, dict):
                obj = obj[key]
            elif isinstance(obj, list):
                obj = obj[int(key)]
            else:
                return f"无法访问路径: {path}"
        return json.dumps(obj, indent=2, ensure_ascii=False) if isinstance(obj, (dict, list)) else str(obj)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return f"提取失败: {e}"


JSON_TOOLS = [json_parse, json_extract]
