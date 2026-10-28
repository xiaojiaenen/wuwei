# 注册自定义工具

```python
from wuwei import ToolRegistry

registry = ToolRegistry()

@registry.tool(side_effect=False)
def join_names(names: list[str], separator: str = ", ") -> str:
    """连接姓名列表。"""
    return separator.join(names)

assert registry.get("join_names").parameters.properties["names"]["type"] == "array"
```

传 `Agent(llm=llm, tools=registry)`。`registry.register_callable(func)` 适合不能使用装饰器的函数；复杂远程 schema 可显式构造 `Tool` 和 `ToolParameters`。参数先验证再执行，同步 handler 在线程中执行。

写入类工具声明 `side_effect=True`，按需设置 `requires_approval=True`、`timeout_seconds`。超时不保证线程或远端动作停止；unknown 结果必须先对账。框架不会盲目重试已声明的副作用工具。

`load_builtins="minimal"` 为默认策略。可用 `load_builtins=["calc", "file"]` 按名选择，或 False 完全禁用内置插件。增加插件不等于授予业务权限。
