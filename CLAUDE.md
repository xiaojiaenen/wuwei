# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Wuwei 是一个轻量 Python Agent 框架，核心循环：**用户输入 → LLM → 工具调用 → 结果回传 → 循环直到完成**。通过 Hook 机制扩展持久化、审批、上下文压缩等能力。

## 常用命令

```bash
# 安装
pip install -e ".[dev]"

# 测试
pytest                                    # 全部
pytest tests/test_builtin_tools.py        # 单文件
pytest tests/test_builtin_tools.py -v     # 详细输出

# 代码质量
ruff check wuwei/ tests/                  # lint
black wuwei/ tests/                       # 格式化
```

## 架构

```
wuwei/
├─ agent/       # Agent、PlanAgent — 用户直接使用的门面对象
├─ runtime/     # AgentRunner — 执行循环；HookManager — 生命周期钩子
├─ planning/    # Planner — 任务分解；Task — 任务模型
├─ memory/      # Context — 消息容器；ContextWindow — 滑动窗口；Storage — 持久化协议
├─ llm/         # LLMGateway — 模型调用；adapters/ — 适配器（目前仅 OpenAI）
├─ tools/       # Tool — 工具定义；ToolRegistry — 注册；ToolExecutor — 执行
└─ skill/       # Skill — 技能定义；SkillManager — 管理；SkillProvider — 加载
```

### 核心流程

```
Agent.run(input)
  → AgentRunner._run_non_stream(input)
    → loop (最多 max_steps 轮):
        1. HookManager.before_llm(messages, tools)   # 上下文裁剪、注入记忆等
        2. LLMGateway.generate(messages, tools)       # 调用模型
        3. HookManager.after_llm(response)
        4. 如果有 tool_calls:
             HookManager.before_tool(tool_call)       # 审批检查
             ToolExecutor.execute_one(tool_call)       # 执行工具
             HookManager.after_tool(tool_call, result) # 持久化结果
        5. 否则: 结束，返回结果
```

### 扩展方式：Hook

继承 `RuntimeHook`，重写需要的回调：

```python
class MyHook(RuntimeHook):
    async def before_llm(self, session, messages, tools, *, step, task=None):
        # 修改 messages 或 tools
        return messages, tools

    async def after_tool(self, session, tool_call, tool_message, *, step, task=None, tool=None):
        # 工具执行后的副作用，如持久化、通知
        ...
```

注册到 Agent：

```python
agent = Agent(llm=llm, tools=tools, hooks=[MyHook(), StorageHook(storage)])
```

### 添加自定义工具

```python
# 方式一：装饰器
registry = ToolRegistry()

@registry.tool(name="my_tool", description="做某件事")
def my_tool(param: str) -> dict:
    return {"result": param.upper()}

# 方式二：register_callable
def my_func(x: int) -> int:
    return x * 2
registry.register_callable(my_func)
```

### 添加内置工具

在 `wuwei/tools/builtin/` 下新建文件，写一个 `register_xxx(registry, **kwargs)` 函数，然后在 `__init__.py` 的 `BUILTIN_TOOL_REGISTRARS` 中注册。`from_builtin` 会通过 inspect 自动转发匹配的 kwargs（如 `skill_manager`）。

## 开发规范

- Python >=3.10，使用 `list[...]`、`dict[...]`、`X | None` 等现代语法
- 异步优先：所有 Hook 回调、工具执行、LLM 调用都是 async
- 工具函数返回 `dict`，错误抛异常（ToolExecutor 会捕获并转为结构化错误消息）
- 测试用 pytest + pytest-asyncio，`asyncio_mode = "auto"` 已配置
