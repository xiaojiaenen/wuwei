# 控制上下文窗口

模型上下文窗口是请求限制，不是永久存储。长对话先保留 system 约束、完整工具调用/结果配对和最近轮次，再压缩较早内容。

Wuwei 使用 ContextCompressionMiddleware 处理长历史；它不代表所有消息原文都保留。需要可召回原文时配置持久 Ledger 或独立存储，参见[压缩](compression.md)与[存储](storage.md)。

不要按字符随意切断工具 JSON，也不要把 summary 当事实数据库。token 估算应保留模型窗口余量。
