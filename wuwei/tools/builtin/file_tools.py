from markitdown import MarkItDown

from wuwei.tools import ToolRegistry


def register_file_tools(registry: ToolRegistry):
    @registry.tool(name="file_to_md", description="将文件转换为markdown")
    def file_to_md(path: str):
        try:
            md_converter = MarkItDown()
            # 转换文件
            result = md_converter.convert(path)
        except Exception as e:
            return str(e)
        if result.text_content.strip() is None or result.text_content.strip() == "" or result is None:
            return "转换失败，该文件无法转换"
        return result.text_content
