# 模型与 Agent 配置

```python
from wuwei import LLMGateway

llm = LLMGateway.from_env(provider="openai")
# provider="responses" 使用 Responses API；provider="anthropic" 使用 Claude。
# 兼容端点：from_env(provider="openai", env_prefix="DEEPSEEK")
```

环境变量使用 `<PREFIX>_API_KEY`、`<PREFIX>_BASE_URL`、`<PREFIX>_MODEL`。显式配置可传 `LLMGateway({...})`，不要把认证信息写进源文件。

需要自定义认证、代理或连接池时，优先传 `client` 或 `client_factory`，而不是修改消息转换器：

```python
from openai import AsyncOpenAI
from wuwei import LLMGateway

client = AsyncOpenAI(max_retries=0)  # 从环境变量读取认证
llm = LLMGateway({"provider": "openai", "client": client,
                  "model": "gpt-4o", "temperature": 0.2,
                  "retry": {"max_attempts": 2}, "timeout": 30})
# 使用完 await llm.aclose()；借入 client 默认由调用方 await client.close()。
```

框架自建 SDK client 默认关闭 SDK 重试，让网关负责分类重试。借入客户端也应明确 retry 所有权；否则 SDK 与框架尝试次数可能相乘。`set_fallback(other_gateway)` 借用完整 fallback 策略，调用方管理 other_gateway 生命周期。已发布流式内容、工具或 usage 元数据后不重放请求。确定性参数错误不做传输重试。
