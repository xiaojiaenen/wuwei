"""沙箱驱动器工厂

按名称构造 :class:`~wuwei.sandbox.base.SandboxDriver`：

- ``local`` / ``local-sandbox``: :class:`~wuwei.sandbox.local.LocalSandbox`
  （旧版 shell 命令沙箱）
- ``local-python`` / ``python``: :class:`~wuwei.sandbox.local.LocalPythonDriver`
- ``docker``: :class:`~wuwei.sandbox.docker.DockerDriver`
- ``e2b``: :class:`~wuwei.sandbox.e2b.E2BDriver`
"""


from wuwei.sandbox.base import SandboxDriver
from wuwei.sandbox.docker import DockerDriver
from wuwei.sandbox.e2b import E2BDriver
from wuwei.sandbox.local import LocalPythonDriver, LocalSandbox

_REGISTRY: dict = {
    "local": LocalSandbox,
    "local-sandbox": LocalSandbox,
    "local-python": LocalPythonDriver,
    "python": LocalPythonDriver,
    "docker": DockerDriver,
    "e2b": E2BDriver,
}


def get_sandbox(name: str, **kwargs) -> SandboxDriver:
    """按名称构造沙箱驱动器。

    Args:
        name: 驱动器名称（local / local-python / docker / e2b，大小写不敏感）。
        **kwargs: 透传给对应驱动器构造函数的参数。

    Raises:
        ValueError: 未知的驱动器名称。
    """
    key = (name or "").strip().lower()
    cls: type = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"未知沙箱驱动器: {name!r}，可选值: {', '.join(sorted(set(_REGISTRY)))}"
        )
    return cls(**kwargs)
