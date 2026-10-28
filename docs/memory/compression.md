# 压缩与原文召回

```python
from wuwei.middleware import MiddlewareStack, ContextCompressionMiddleware
stack = MiddlewareStack()
# 已配置 llm 后：
# stack.add(ContextCompressionMiddleware(llm=llm, trigger_tokens=6000,
#                                       keep_recent_turns=3))
```

压缩在模型调用前检查窗口，将旧内容变成摘要并保留最近轮次。摘要模型调用也需要费用与失败策略。

LedgerMiddleware 将大块工具结果外置，模型通过 read_context(entry_id) 召回。使用 ContextLedger(persist_dir=...) 保存持久原文，并将 ledger_tools(ledger) 注册到同一工具注册表。pin 状态随条目持久化，内存缓存淘汰后仍可补读。

非持久账本为保留引用而保留原文，总内存不保证有界；部署时明确保留期限与磁盘容量。压缩降低窗口成本，不保证模型永不遗漏信息。
