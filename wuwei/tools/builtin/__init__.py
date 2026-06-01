from .calc_tools import register_calc_tools
from .decision_tools import register_decision_tools
from .file_tools import register_file_tools
from .git_tools import register_git_tools
from .npm_tools import register_npm_tools
from .python_tools import register_python_tools
from .rag_tools import register_rag_tools
from .skill_tools import register_skill_tools
from .time_tools import register_time_tools
from .json_tools import JSON_TOOLS
from .http_tools import HTTP_TOOLS
from .text_tools import TEXT_TOOLS

__all__ = [
    "register_calc_tools",
    "register_decision_tools",
    "register_file_tools",
    "register_git_tools",
    "register_npm_tools",
    "register_python_tools",
    "register_rag_tools",
    "register_skill_tools",
    "register_time_tools",
    "JSON_TOOLS",
    "HTTP_TOOLS",
    "TEXT_TOOLS",
]

BUILTIN_TOOL_REGISTRARS = {
    "calc": register_calc_tools,
    "time": register_time_tools,
    "file": register_file_tools,
    "git": register_git_tools,
    "npm": register_npm_tools,
    "python": register_python_tools,
    "skill": register_skill_tools,
    "rag": register_rag_tools,
    "decision": register_decision_tools,
    "json": lambda registry: [registry.register(t) for t in JSON_TOOLS],
    "http": lambda registry: [registry.register(t) for t in HTTP_TOOLS],
    "text": lambda registry: [registry.register(t) for t in TEXT_TOOLS],
}
