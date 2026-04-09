# Examples

`examples/` 目录只放可直接运行的示例，`tests/` 目录只放测试文件。

## 运行前准备

离线示例不需要模型配置：

```bash
python examples/tool_executor_minimal.py
```

在线示例需要先设置 API Key：

```powershell
$env:WUWEI_API_KEY="your_key"
```

如果你使用兼容 OpenAI 协议的服务，也可以设置：

```powershell
$env:WUWEI_BASE_URL="https://api.deepseek.com"
$env:WUWEI_MODEL="deepseek-chat"
```

## 示例列表

- `examples/tool_executor_minimal.py`
  - 纯本地离线运行
  - 演示 `ToolRegistry + ToolExecutor`

- `examples/agent_minimal.py`
  - 最小 `Agent` 示例
  - 演示工具调用和流式输出

- `examples/agent_session_minimal.py`
  - 演示 `AgentSession` 多轮复用
  - 第二轮问题会读取第一轮上下文

- `examples/plan_agent_minimal.py`
  - 最小 `PlanAgent` 示例
  - 演示先规划再执行
  - 打印 task 列表、工具调用和最终 task 结果
