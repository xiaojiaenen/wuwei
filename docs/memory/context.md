# 会话上下文

Context 保存消息序列，AgentSession 将它与会话配置关联。get_messages() 读取历史，add_user_message/add_system_message/add_ai_message/add_tool_message 添加相应消息，replace_messages() 替换视图。

```python
from wuwei import Context
context = Context()
context.add_system_message("请保持简洁")
context.add_user_message("你好")
assert context.get_last_message().content == "你好"
```

应用通常通过 Agent.run 操作上下文，不必手动追加同一用户输入。不要跨用户共用默认 session；参见[会话](../core/session.md)。
