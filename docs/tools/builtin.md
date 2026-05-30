# 内置工具

Wuwei 提供 7 组内置工具，通过名称注册到 `ToolRegistry`。

## 注册方式

```python
from wuwei.tools import ToolRegistry

# 注册全部内置工具
registry = ToolRegistry.from_builtin(["time", "file", "git", "calc", "python", "npm", "skill"])

# 或只注册需要的
registry = ToolRegistry.from_builtin(["time", "calc"])
```

内置工具名称映射表：

| 名称 | 注册函数 | 工具数量 |
|------|---------|---------|
| `time` | `register_time_tools` | 1 |
| `file` | `register_file_tools` | 6 |
| `git` | `register_git_tools` | 6 |
| `calc` | `register_calc_tools` | 1 |
| `python` | `register_python_tools` | 1 |
| `npm` | `register_npm_tools` | 3 |
| `skill` | `register_skill_tools` | 3 |

---

## time — 时间工具

### `get_now`

获取当前时间，支持时区。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timezone` | `str` | `"Asia/Shanghai"` | IANA 时区名 |

```python
# LLM 调用示例
{"timezone": "Asia/Shanghai"}
# 返回: {"timezone": "Asia/Shanghai", "iso": "2025-04-29T22:57:00+08:00"}
```

---

## file — 文件工具

所有路径参数相对于 `workspace`（默认为 `"."`），框架自动校验路径安全性。

### `file_to_md`

将文件转换为 Markdown（依赖 `markitdown` 库）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` | 文件路径 |

### `read_text_file`

读取文本文件，默认最多 20000 字符。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | — | 文件路径 |
| `max_chars` | `int` | `20000` | 最大返回字符数 |
| `workspace` | `str` | `"."` | 工作区根目录 |

### `write_text_file`

写入文本文件，默认不覆盖已有文件。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | — | 文件路径 |
| `content` | `str` | — | 写入内容 |
| `overwrite` | `bool` | `False` | 是否覆盖 |
| `workspace` | `str` | `"."` | 工作区根目录 |

### `append_text_file`

追加内容到文件末尾，不存在时自动创建。

### `replace_text_in_file`

替换文件中的文本。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | — | 文件路径 |
| `old_text` | `str` | — | 查找文本 |
| `new_text` | `str` | — | 替换文本 |
| `count` | `int` | `-1` | 最大替换次数，`-1` 为全部 |

### `delete_file`

删除单个文件（不删除目录）。

---

## git — Git 工具

默认超时 30 秒，最大输出 12000 字符。

| 工具 | 说明 | 关键参数 |
|------|------|---------|
| `git_status` | 查看仓库状态 | `workspace`, `short` |
| `git_diff` | 查看 diff | `path`, `staged`, `workspace` |
| `git_log` | 查看提交日志 | `limit` (1-100), `workspace` |
| `git_show` | 查看 revision 内容 | `revision`, `stat_only` |
| `git_add` | 暂存文件 | `path`, `workspace` |
| `git_commit` | 创建 commit | `message`, `workspace` |

> :warning: `git_commit` 和 `git_add` 会修改仓库状态，建议配合 HITL 审批使用。

---

## calc — 计算工具

### `calculate`

安全计算数学表达式。基于 AST 解析，**不执行任意代码**。

支持：`+ - * / // % **`、括号、所有 `math` 模块函数和常量。

```python
# 示例
{"expression": "sqrt(16) + sin(pi / 2)"}  # → 5.0
{"expression": "round(10 / 3, 2)"}         # → 3.33
{"expression": "2 ** 10"}                  # → 1024
```

---

## python — Python 脚本工具

### `run_python_script`

执行 workspace 内的 `.py` 脚本。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `script_path` | `str` | — | 相对 workspace 的脚本路径 |
| `args_json` | `str` | `"[]"` | 命令行参数，JSON 数组 |
| `timeout_seconds` | `int` | `10` | 超时时间 |
| `max_output_chars` | `int` | `4000` | 最大输出字符数 |
| `workspace` | `str` | `"."` | 工作区根目录 |

```python
{"script_path": "scripts/analyze.py", "args_json": "[\"--input\", \"data.csv\"]"}
```

---

## npm — Node.js 工具

默认超时 120 秒，最大输出 12000 字符。

### `npm_list_scripts`

读取 `package.json` 中的 `scripts` 列表。

### `npm_run_script`

运行 npm script。

| 参数 | 类型 | 说明 |
|------|------|------|
| `script_name` | `str` | scripts 中的脚本名 |
| `args_json` | `str` | 额外参数，JSON 数组 |
| `workspace` | `str` | 包含 package.json 的目录 |

### `npm_install_package`

安装 npm 包。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `package` | `str` | — | 包名，如 `lodash` 或 `typescript@latest` |
| `dev` | `bool` | `False` | 是否安装到 devDependencies |

> :warning: 会修改 `package.json`，建议配合 HITL 审批。

---

## skill — 技能工具

与 [技能系统](../skill/overview.md) 配合使用。

| 工具 | 说明 |
|------|------|
| `list_skills` | 列出所有可用技能的摘要 |
| `load_skill` | 加载指定技能的指令正文 |
| `run_skill_python_script` | 执行技能目录下的 Python 脚本 |

需要先 `load_skill` 获取 `load_token`，才能调用 `run_skill_python_script`。
