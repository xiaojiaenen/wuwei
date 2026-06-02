"""文本处理工具插件"""

from __future__ import annotations

from wuwei.plugin import PluginContext


def setup(ctx: PluginContext) -> None:
    @ctx.tool_registry.tool(
        name="text_replace",
        description="替换文本中的字符串",
        display_name="文本替换",
    )
    def text_replace(text: str, old: str, new: str) -> str:
        """替换文本中的字符串

        Args:
            text: 原始文本
            old: 要替换的字符串
            new: 替换后的字符串
        """
        return text.replace(old, new)

    @ctx.tool_registry.tool(
        name="text_split",
        description="按分隔符分割文本",
        display_name="文本分割",
    )
    def text_split(text: str, delimiter: str = "\n") -> str:
        """按分隔符分割文本

        Args:
            text: 原始文本
            delimiter: 分隔符
        """
        parts = text.split(delimiter)
        return "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))

    @ctx.tool_registry.tool(
        name="text_join",
        description="合并文本",
        display_name="文本合并",
    )
    def text_join(items: str, delimiter: str = "\n") -> str:
        """合并文本

        Args:
            items: 用换行分隔的文本
            delimiter: 合并分隔符
        """
        return delimiter.join(items.split("\n"))

    @ctx.tool_registry.tool(
        name="text_upper",
        description="转换为大写",
        display_name="转大写",
    )
    def text_upper(text: str) -> str:
        """转换为大写

        Args:
            text: 原始文本
        """
        return text.upper()

    @ctx.tool_registry.tool(
        name="text_lower",
        description="转换为小写",
        display_name="转小写",
    )
    def text_lower(text: str) -> str:
        """转换为小写

        Args:
            text: 原始文本
        """
        return text.lower()

    @ctx.tool_registry.tool(
        name="text_trim",
        description="去除首尾空白",
        display_name="去除空白",
    )
    def text_trim(text: str) -> str:
        """去除首尾空白

        Args:
            text: 原始文本
        """
        return text.strip()
