from .file_tools import register_file_tools
from .skill_tools import register_skill_tools
from .time_tools import register_time_tools

__all__ = ["register_file_tools", "register_skill_tools", "register_time_tools"]

BUILTIN_TOOL_REGISTRARS = {
    "time": register_time_tools,
    "file": register_file_tools,
    "skill": register_skill_tools,
}
