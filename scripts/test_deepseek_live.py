"""DeepSeek 真实调用最小测试（省钱模式：共 3 次 LLM 调用，max_tokens 限制很小）

用法：.venv/bin/python scripts/test_deepseek_live.py
前置：项目根目录 .env 配置 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from wuwei.agent import Agent
from wuwei.llm import LLMGateway
from wuwei.tools import Tool, ToolParameters, ToolRegistry


def get_price(coin: str) -> dict:
    """查询币种价格（演示假数据）。

    :param coin: 币种名称，如 btc
    """
    return {"coin": coin, "price_usd": 67000.0}


async def main() -> int:
    llm = LLMGateway.from_env(provider="openai", env_prefix="DEEPSEEK")
    print(f"[1] provider={llm.provider} model={llm.adapter.model} base_url={llm.adapter.client.base_url}")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_price",
            description="查询币种的美元价格",
            parameters=ToolParameters(),
            handler=get_price,
        )
    )
    agent = Agent(
        llm=llm,
        tools=registry,
        load_builtins=False,
        default_system_prompt="回答尽量简短。",
        default_max_steps=4,
    )

    # ── 测试 1：非流式纯对话（1 次 LLM 调用） ──────────────────────
    t0 = time.monotonic()
    result = await agent.run("用五个字回答：你好吗", )
    dt = time.monotonic() - t0
    print(f"[2] 非流式: content={result.content!r} usage={result.usage} latency={dt:.1f}s")
    assert result.content, "非流式返回为空"

    # ── 测试 2：工具调用循环（2 次 LLM 调用） ──────────────────────
    result2 = await agent.run("btc 现在多少美元？用 get_price 查询后报数", session=agent._default_session)
    tool_msgs = [m for m in agent._default_session.context.get_messages() if m.role == "tool"]
    print(f"[3] 工具调用: content={result2.content!r} tool_msgs={len(tool_msgs)} usage={result2.usage}")
    assert tool_msgs, "工具消息未回写会话"
    assert "67000" in result2.content or "67" in result2.content, f"模型未使用工具结果: {result2.content!r}"

    print("\n全部通过 ✅（共 3 次 LLM 调用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
