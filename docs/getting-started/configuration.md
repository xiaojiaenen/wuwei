# 配置

## LLM 配置

Wuwei 通过环境变量配置 LLM，支持自动查找 `.env` 文件。

### 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | API 密钥 |
| `OPENAI_BASE_URL` | ❌ | API 地址，默认 `https://api.openai.com/v1` |
| `OPENAI_MODEL` | ❌ | 模型名称，默认 `gpt-5.4` |

### .env 文件

框架会自动在当前目录和最多 3 层父目录中查找 `.env` 或 `env` 文件：

```bash
# .env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

### 使用不同的环境变量前缀

```python
# 读取 ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
agent = Agent.from_env(env_prefix="ANTHROPIC")
```

### 显式配置

```python
from wuwei import LLMGateway

llm = LLMGateway({
    "provider": "openai",
    "api_key": "sk-xxx",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 4096,
})
```

## Agent 配置

```python
agent = Agent.from_env(
    # 内置工具
    builtin_tools=["time", "file", "git"],
    
    # 自定义工具
    tools=[my_tool],
    
    # 系统提示词
    system_prompt="你是一个代码助手",
    
    # 最大步骤数（防止无限循环）
    max_steps=15,
    
    # 并行执行工具调用
    parallel_tool_calls=True,
    
    # 生命周期钩子
    hooks=[my_hook],
    
    # LLM 配置覆盖
    model="gpt-4o",
    temperature=0.1,
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `builtin_tools` | `list[str]` | `None` | 内置工具名称列表 |
| `tools` | `list[Tool]` | `None` | 自定义工具列表 |
| `system_prompt` | `str` | `"你是一个有用的助手"` | 系统提示词 |
| `max_steps` | `int` | `10` | 最大执行步骤数 |
| `parallel_tool_calls` | `bool` | `False` | 是否并行执行工具 |
| `hooks` | `list[RuntimeHook]` | `None` | 生命周期钩子列表 |

## 内置工具

| 名称 | 说明 |
|---|---|
| `time` | 获取当前时间 |
| `file` | 文件读写操作 |
| `git` | Git 操作 |
| `calc` | 数学计算 |
| `python` | Python 代码执行 |
| `npm` | npm 包管理 |
| `skill` | 技能管理工具 |

## 重试和超时

```python
llm = LLMGateway({
    "provider": "openai",
    "api_key": "sk-xxx",
    "retry": {"max_attempts": 3},
    "timeout": 60,  # 秒
})
```