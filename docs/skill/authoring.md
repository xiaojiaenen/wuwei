# 技能编写指南

## SKILL.md 格式

每个技能由一个 `SKILL.md` 文件定义，包含 YAML frontmatter 和 Markdown 正文。

```markdown
---
name: code-review
description: 代码审查技能，检查代码质量、安全性和最佳实践
---

# 代码审查指南

## 审查维度

1. **代码质量**：命名规范、函数长度、重复代码
2. **安全性**：SQL 注入、XSS、敏感信息泄露
3. **性能**：时间复杂度、内存使用、I/O 操作
4. **可维护性**：注释、测试覆盖、错误处理

## 审查流程

1. 先阅读整体结构
2. 逐文件检查
3. 输出审查报告

## 输出格式

```markdown
## 审查结果

### 严重问题
- ...

### 建议改进
- ...
```
```

### Frontmatter 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | `string` | 是 | 技能唯一名称 |
| `description` | `string` | 否 | 技能简要描述 |

### 正文

Markdown 正文即为技能的指令内容（`instruction`），会完整传递给 Agent。

## 目录结构

```
skills/
├── code-review/
│   ├── SKILL.md                    # 技能定义
│   └── scripts/                    # 可选：技能脚本
│       ├── check_security.py
│       └── analyze_complexity.py
├── data-analysis/
│   └── SKILL.md
└── deployment/
    ├── SKILL.md
    └── scripts/
        └── validate_config.py
```

### 规则

- 每个技能一个目录
- `SKILL.md` 必须在目录根下
- `scripts/` 目录存放可执行的 Python 脚本（可选）
- 支持嵌套目录，`FileSystemSkillProvider` 会递归扫描

## 完整示例

### 简单技能（无脚本）

```markdown
---
name: commit-message
description: 生成规范的 Git commit message
---

# Commit Message 生成规范

## 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Type 类型

- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- style: 格式调整
- refactor: 重构
- test: 测试
- chore: 构建/工具

## 规则

1. subject 不超过 50 字符
2. 使用祈使句
3. body 说明 why 而非 what
```

### 带脚本的技能

```markdown
---
name: python-lint
description: Python 代码静态分析
---

# Python Lint 技能

## 使用方法

1. 先调用 `scripts/run_flake8.py` 对目标文件运行 flake8
2. 根据输出结果给出修改建议
3. 如果有严重问题，直接指出文件和行号

## 注意事项

- 只分析 `.py` 文件
- 忽略 `__pycache__` 和 `.venv` 目录
```

对应的脚本 `scripts/run_flake8.py`：

```python
import subprocess
import sys

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    result = subprocess.run(
        ["flake8", "--max-line-length=120", target],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

if __name__ == "__main__":
    main()
```

## 最佳实践

1. **描述要精准**：`description` 是 Agent 判断是否使用技能的依据
2. **指令要可执行**：正文应明确告诉 Agent 具体步骤
3. **脚本要自包含**：脚本应处理异常并输出有用信息
4. **避免副作用**：脚本默认只读，写入操作应在指令中明确说明
