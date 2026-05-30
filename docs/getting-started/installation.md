# 安装

## 环境要求

- Python >= 3.10, < 3.14
- 操作系统：macOS / Linux / Windows

## 使用 pip 安装

```bash
pip install wuwei
```

## 使用 uv 安装（推荐）

```bash
uv pip install wuwei
```

## 从源码安装

```bash
git clone https://github.com/xiaojiaenen/wuwei.git
cd wuwei
pip install -e .
```

## 安装开发依赖

```bash
pip install -e ".[dev]"
```

开发依赖包括：

| 包 | 用途 |
|---|---|
| `pytest` | 测试框架 |
| `pytest-asyncio` | 异步测试支持 |
| `black` | 代码格式化 |
| `ruff` | 代码检查 |

## 依赖说明

Wuwei 的核心依赖非常精简：

| 包 | 版本 | 用途 |
|---|---|---|
| `openai` | >= 1.0.0 | LLM 调用 (OpenAI 兼容协议) |
| `pydantic` | >= 2.0.0 | 数据模型和验证 |
| `pyyaml` | >= 6.0.3 | 配置文件解析 |
| `markitdown` | >= 0.1.5 | 文档解析 (PPTX/DOCX/XLSX/PDF) |

## 验证安装

```python
import wuwei
print(wuwei.__version__)  # 应输出 0.1.8
```

## 下一步

- [快速上手](quickstart.md) — 5 分钟运行第一个 Agent
- [配置](configuration.md) — 了解 LLM 和工具配置