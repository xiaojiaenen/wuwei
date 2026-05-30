# Changelog

## v0.1.8

### 新增

- **Hook 系统重构**
  - `RuntimeHook` 基类：`before_llm`、`after_llm`、`after_ai_message`、`before_tool`、`after_tool`、`on_task_start`、`on_task_end` 共 7 个钩子点
  - `HookManager`：管理多个 Hook 的注册与链式调用
  - `ConsoleHook`：控制台调试日志 Hook
  - `ContextCompressionHook`：轮次过多时自动压缩上下文
  - `StorageHook`：消息增量持久化 Hook
  - `SkillHook`：技能使用说明自动注入 Hook
  - `HitlHook`：人机协作审批 Hook

- **流式事件流**
  - `AgentEvent` 模型：`text_delta`、`reasoning_delta`、`tool_start`、`tool_end`、`done`、`error` 六种事件类型
  - `Agent.stream_events()` / `PlanAgent.stream_events()` 统一事件流接口
  - `AgentRunner.stream_events()` / `PlannerExecutorRunner.execute_events()` 底层事件流实现

- **技能系统增强**
  - `run_skill_python_script` 工具支持 `load_token` 安全校验
  - `list_skills` 返回技能目录下可用 Python 脚本列表

- **上下文窗口管理**
  - `SimpleContextWindow`：构建发给模型的精简消息窗口
  - `ContextWindowConfig`：配置 `max_recent_turns`、`max_tool_chars`、`include_summary`
  - `split_turns()`：消息列表按轮次拆分

- **LLM 网关**
  - `LLMGateway.from_env()` 自动搜索 `.env` 文件（最多 3 层父目录）
  - 流式响应自动拼接 tool call 增量为完整结构

### 变更

- `Agent` 和 `PlanAgent` 构造函数统一接受 `hooks` 参数
- `BaseSessionAgent` 提取公共逻辑：session 创建/复用、tool_registry 初始化
- `ToolExecutor.execute()` 支持 `concurrent=True` 并行执行
- `FileStorage.save_meta()` 使用原子写入（`.tmp` + `os.replace()`）

### 修复

- 修复流式模式下 `tool_calls_complete` 未正确传递的问题
- 修复 `PlannerExecutorRunner` 中任务统计未正确累加的问题
- 修复 `ContextCompressionHook` 重复压缩的边界条件

---

本文件记录 Wuwei Agent 框架的重要变更。
格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。
