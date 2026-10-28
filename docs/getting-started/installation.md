# 安装

需要 Python 3.11+。

```bash
pip install wuwei
pip install 'wuwei[anthropic,mcp]'  # 按需扩展
# 开发：
pip install -e '.[dev]'
```

核心包含图、中间件、内存存储和本地执行驱动。graph/middleware/memory/sandbox 兼容 extra 保留为空。可选扩展：anthropic、mcp、skill、gateway、observability、markitdown、e2b、chroma、qdrant、pgvector、milvus、vectorstore、serve；all 包含所有可选 Python 扩展，外部服务需另配。

模型接入依赖范围：OpenAI SDK >=3.23、Anthropic SDK >=1.11（可选）；MCP >=2.2,<3；工具 schema 验证需要 jsonschema>=4。具体声明以项目 pyproject.toml 为准，版本下限不是所有上游版本的互操作承诺。

验证安装不需要读取密钥：

```python
from importlib.metadata import version
from wuwei import Agent, LLMGateway, ToolRegistry
print(version("wuwei"))
```

测试默认离线，`pytest --live` 显式开启需要账号与远程服务的测试。
