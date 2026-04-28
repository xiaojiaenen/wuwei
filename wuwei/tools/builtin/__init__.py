from .calc_tools import register_calc_tools
from .file_tools import register_file_tools
from .git_tools import register_git_tools
from .npm_tools import register_npm_tools
from .python_tools import register_python_tools
from .skill_tools import register_skill_tools
from .time_tools import register_time_tools

__all__ = [
    "register_calc_tools",
    "register_file_tools",
    "register_git_tools",
    "register_npm_tools",
    "register_python_tools",
    "register_skill_tools",
    "register_time_tools",
]

BUILTIN_TOOL_REGISTRARS = {
    "calc": register_calc_tools,
    "time": register_time_tools,
    "file": register_file_tools,
    "git": register_git_tools,
    "npm": register_npm_tools,
    "python": register_python_tools,
    "skill": register_skill_tools,
}
