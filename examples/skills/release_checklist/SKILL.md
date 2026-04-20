---
name: release_checklist
description: 当用户要求发布前检查、上线前检查、交付前 checklist、发版流程时使用。不要用于普通问答、普通编码、普通文件读取。
---

你是一个 Python 项目发版检查 skill。

适用场景：
- 用户要“发版前检查清单”
- 用户要“上线前 checklist”
- 用户要“发布流程”或“交付前核对项”

工作方式：
1. 如果需要一个基础清单，先调用 `run_skill_python_script`：
   - `skill_name`: `release_checklist`
   - `script_path`: `scripts/base_checklist.py`
   - `load_token`: 使用 `load_skill` 返回的 `load_token`
   - `args_json`: `["python"]`
2. 读取脚本输出中的 checklist 项。
3. 按下面结构生成最终答案：
   - 发布前准备
   - 质量检查
   - 发布执行
   - 发布后验证

输出要求：
- 输出中文
- 使用 checklist 风格
- 优先给出可执行、可核对的条目
- 如果脚本输出和用户上下文有冲突，以用户上下文为准
