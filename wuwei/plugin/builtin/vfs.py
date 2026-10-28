"""VFS 插件 — 虚拟文件系统工具组

加载方式：`Agent(load_builtins=["vfs"])`，默认挂载 LocalFS(".")（当前目录
作为 VFS 根）。自定义后端请直接使用 wuwei.fs.make_vfs_plugin。
"""

from wuwei.plugin import PluginContext


def setup(ctx: PluginContext) -> None:
    from wuwei.fs import LocalFS, make_vfs_tools

    backend = LocalFS(".")
    for tool in make_vfs_tools(backend, workspace_label="vfs"):
        ctx.tool_registry.register(tool)
