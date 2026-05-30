# 自定义工具

Wuwei 提供三种方式注册自定义工具。

## 方式一：装饰器（推荐）

```python
from wuwei.tools import ToolRegistry

registry = ToolRegistry()

@registry.tool(description="查询天气")
def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询指定城市的天气。

    :param city: 城市名称
    :param unit: 温度单位
    """
    return {"city": city, "temp": 25, "unit": unit}
```

装饰器自动完成：
1. 从 `func.__name__` 推断工具名称（可用 `name=` 覆盖）
2. 从 `func.__doc__` 提取描述（可用 `description=` 覆盖）
3. 从类型注解生成 JSON Schema
4. 从 docstring `:param:` 提取参数描述
5. 无默认值参数自动标记为 `required`

## 方式二：注册可调用对象

```python
from wuwei.tools import ToolRegistry

registry = ToolRegistry()

def my_tool(query: str, limit: int = 10) -> dict:
    """搜索数据库。"""
    return {"results": [...]}

# register_callable 返回注册后的 Tool 实例
tool = registry.register_callable(
    my_tool,
    name="search_db",
    description="搜索用户数据库",
)
```

适用于需要显式控制名称和描述的场景。

## 方式三：手动构造 Tool 实例

```python
from wuwei.tools import Tool, ToolParameters, ToolRegistry

registry = ToolRegistry()

tool = Tool(
    name="send_email",
    description="发送邮件",
    parameters=ToolParameters(
        properties={
            "to": {"type": "string", "description": "收件人"},
            "subject": {"type": "string", "description": "主题"},
            "body": {"type": "string", "description": "正文"},
        },
        required=["to", "subject", "body"],
    ),
    handler=lambda to, subject, body: {"sent": True, "to": to},
)

registry.register(tool)
```

适用于参数结构复杂或需要完全控制 Schema 的场景。

## 异步支持

所有方式均支持异步 handler：

```python
@registry.tool(description="调用外部 API")
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return {"status": resp.status_code, "body": resp.text}
```

`Tool.invoke()` 内部自动检测：
- `inspect.iscoroutinefunction(handler)` → 直接 `await handler(**args)`
- 普通函数返回值如果实现了 `__await__`，也会 `await`

## 类型映射表

注册时框架自动将 Python 类型映射为 JSON Schema 类型：

| Python 类型 | JSON Schema 类型 | 说明 |
|-------------|-----------------|------|
| `str` | `string` | 字符串 |
| `int` | `number` | 整数 |
| `float` | `number` | 浮点数 |
| `complex` | `number` | 复数（少见） |
| `bool` | `boolean` | 布尔值 |
| `list` | `array` | 列表 |
| `tuple` | `array` | 元组 |
| `dict` | `object` | 字典 |
| 其它 / 无注解 | `string` | 兜底类型 |

> :bulb: 如需更精确的类型控制（如 `enum`、`array` 元素类型），请使用方式三手动构造 `ToolParameters`。

## Agent.from_env 中注册

`Agent.from_env()` 和 `PlanAgent.from_env()` 的 `tools` 参数接受混合列表：

```python
from wuwei import Agent, Tool

agent = Agent.from_env(
    builtin_tools=["time", "calc"],
    tools=[my_tool, fetch_data],  # 可调用对象
    # tools 也接受 Tool 实例
)
```

框架会自动区分 `Tool` 实例和普通可调用对象，分别调用 `register()` 或 `register_callable()`。
