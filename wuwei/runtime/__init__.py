"""运行时模块

注意：旧 Hook 系统（RuntimeHook, HookManager）已弃用，请使用 Middleware 系统。
"""

# 新版：使用 Middleware
from wuwei.runtime.agent_runner_v2 import AgentRunner

# 旧版：Hook 系统（已弃用，保留用于向后兼容）
from wuwei.runtime.hooks import HookManager, RuntimeHook
from wuwei.runtime.console_hook import ConsoleHook
from wuwei.runtime.context_hook import ContextCompressionHook
from wuwei.runtime.hitl import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalProvider,
    ApprovalRequest,
    ConsoleApprovalProvider,
    ToolApprovalRejected,
)
from wuwei.runtime.hitl_hook import HitlHook
from wuwei.runtime.memory_hook import MemoryExtractionHook, MemoryRetrievalHook
from wuwei.runtime.planner_executor_runner import PlannerExecutorRunner
from wuwei.runtime.rag_hook import RagRetrievalHook
from wuwei.runtime.skill_hook import SkillHook
from wuwei.runtime.storage_hook import StorageHook

__all__ = [
    # 新版
    "AgentRunner",
    # 旧版（已弃用）
    "HookManager",
    "RuntimeHook",
    "ConsoleHook",
    "ContextCompressionHook",
    "MemoryExtractionHook",
    "MemoryRetrievalHook",
    "RagRetrievalHook",
    "SkillHook",
    "StorageHook",
    "PlannerExecutorRunner",
    "ApprovalDecision",
    "ApprovalPolicy",
    "ApprovalProvider",
    "ApprovalRequest",
    "ConsoleApprovalProvider",
    "HitlHook",
    "ToolApprovalRejected",
]
