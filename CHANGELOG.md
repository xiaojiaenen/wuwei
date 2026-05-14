# 更新日志

## [Unreleased]

### 新增

- **工具中文名**：`Tool` 模型新增 `display_name` 字段，支持为工具设置中文展示名，方便下游 UI 展示。`ToolRegistry.tool()` 和 `ToolRegistry.register_callable()` 均新增 `display_name` 参数。所有内置工具已配置中文名。

### 修复

- 修复 Union 类型注解（`X | None`）导致 `__name__` 报错的问题。

## [1.0.0] - 2026-05-08

### 新增

- 首个正式版本发布。
- 记忆衰减与自动清理机制。
- 长期记忆与 RAG 知识库功能。
- Skill 引用资源与缓存机制。
