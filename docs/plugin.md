# 插件生命周期

插件目录包含 __init__.py 的 setup(ctx)，可选 plugin.yaml 描述 name/version/dependencies/tags。PluginManager 负责发现、依赖与重名检查、注册和卸载。Agent 接受 plugins 或 plugin_dir。

```python
from wuwei import PluginContext, PluginManager, ToolRegistry
registry = ToolRegistry()
manager = PluginManager(PluginContext(tool_registry=registry))

def setup(ctx):
    @ctx.tool_registry.tool()
    def hello(name: str) -> str:
        return f"你好，{name}"
```

在插件 setup 中用 ctx.middleware_stack.add(middleware) 注册生命周期行为；不要在插件另建绕过治理的工具执行循环。

minimal 为 calc/time/json/text；all 包括写入和脚本能力。加载代码不是授权，业务权限另外配置。
