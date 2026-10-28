# 存储会话和检查点

FileStorage 提供 save_meta(session)、append_message(session_id, message)、load(session_id)、delete(session_id)。StorageMiddleware 按运行的 session_id 归档；不要把所有用户写进默认目录。

会话 JSON 保存历史与配置；checkpoint 保存恢复所需的 run 状态，两者用途不同。MemoryCheckpointer 仅进程内；SQLiteCheckpointer 用于本地持久化与原子 claim。多进程/自定义后端须提供等价的原子领取契约。

含密钥或业务敏感内容的历史、账本、检查点需要访问控制、保存期限和备份。加载 session 后仍需按当前权限验证操作。
