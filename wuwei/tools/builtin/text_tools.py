"""文本处理工具"""

from wuwei.tools.base import Tool


@Tool.from_function
def text_replace(text: str, old: str, new: str) -> str:
    """替换文本中的字符串

    Args:
        text: 原始文本
        old: 要替换的字符串
        new: 替换后的字符串
    """
    return text.replace(old, new)


@Tool.from_function
def text_split(text: str, delimiter: str = "\n") -> str:
    """按分隔符分割文本

    Args:
        text: 原始文本
        delimiter: 分隔符
    """
    parts = text.split(delimiter)
    return "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))


@Tool.from_function
def text_join(items: str, delimiter: str = "\n") -> str:
    """合并文本

    Args:
        items: 用换行分隔的文本
        delimiter: 合并分隔符
    """
    return delimiter.join(items.split("\n"))


@Tool.from_function
def text_upper(text: str) -> str:
    """转换为大写

    Args:
        text: 原始文本
    """
    return text.upper()


@Tool.from_function
def text_lower(text: str) -> str:
    """转换为小写

    Args:
        text: 原始文本
    """
    return text.lower()


@Tool.from_function
def text_trim(text: str) -> str:
    """去除首尾空白

    Args:
        text: 原始文本
    """
    return text.strip()


TEXT_TOOLS = [text_replace, text_split, text_join, text_upper, text_lower, text_trim]
