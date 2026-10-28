# 模型层独立复核

复核范围：`wuwei/llm/**`、`wuwei/core/types.py`、`wuwei/core/message.py`，对照 `USER_FEEDBACK_AND_API_REVIEW.md`。只读源码与针对反例的本地探针；复用模型 owner 已有 51 条通过结果，没有重复执行该套测试。未修改模型实现。

## 需要修复的三个问题

### P1：显式 Responses 输出上限被默认值覆盖

位置：`wuwei/llm/gateway.py` 构造 `adapter_kwargs["max_tokens"]`；`wuwei/llm/adapters/responses.py:43`。

探针：`LLMGateway({'provider':'responses', 'client':fake_client, 'max_output_tokens':99}).adapter.build_request([])['max_output_tokens']` 实际为 **4096**，不是 99。Gateway 无条件补充 max_tokens 默认值，Adapter 又让这个旧别名优先于显式 max_output_tokens。会增加请求输出范围与费用，且可能打破模型上限。

修复：provider 边界只生成一个规范 token 字段；显式 max_output_tokens 优先，或者冲突时明确报错，不能让隐式默认覆盖显式值。补 Gateway → Adapter 请求回归，覆盖只提供新字段以及两字段冲突。

### P1：fallback 重置整体 deadline

位置：`wuwei/llm/gateway.py:325–331`、`412–436`。

每个 gateway 在自己的 asyncio.timeout 内执行；主链超时后，fallback 在原 timeout 外重新开始完整 timeout。探针主 timeout=0.02s、fallback timeout=0.2s，两 Adapter 各等待 0.04s，最终 **0.062s 返回成功**，超过主调用整体预算。流式路径结构相同。

修复：顶层请求生成绝对 deadline，并让全部 primary attempts、退避、fallback 链共享剩余时间。不要把同一 deadline 超时当作仍有新预算的 fallback 条件。

### P2：能力开关打开后 canonical 消息仍丢 reasoning_content

位置：`wuwei/llm/adapters/openai.py:62–68`；`wuwei/core/message.py:73–80`。

探针：`OpenAIAdapter(..., supports_reasoning_content=True).build_request([BaseMessage(role='assistant', content='answer', reasoning_content='reason')])['messages']` 得到 `[{'role':'assistant','content':'answer'}]`。Canonical response 使用 BaseMessage；只有 AIMessage 覆盖的 to_openai 会加入 reasoning。Adapter 的开关只是不删除字段，却没有在规范消息上补入它。

修复：在 provider capability 分支显式序列化 canonical reasoning_content；默认不开启时继续省略。补从 parse_response → canonical message → build_request 的往返测试，而不是只拿 AIMessage 构造测试。

## 其余关键路径的判断

- JSON 损坏/非对象参数：流式在完整正常终止后才解析 JSON，对非对象拒绝；非流式 FunctionCall.arguments 的 Pydantic dict 契约阻止非对象。length/incomplete 不暴露可执行完整调用。未发现此前“坏 JSON 替换为空对象”的路径。
- 流结束：EOF 没有终止标记抛错；终止后新增内容抛错；底层 stream 在 finally 关闭。已发布 usage/tool metadata 同样算 observed，阻止重试/fallback。代码支持这一安全边界。
- 重试：构造的 SDK client 默认 max_retries=0，确定性异常不在网关重试白名单；没有发现框架默认情况下主 attempts 与 fallback attempts 相乘的递归。显式传 max_retries 或借入 SDK client 可以另有 SDK 重试；当前不会验证/关闭借入客户端的策略，应在公开 API 文档明确由谁拥有重试预算，避免将框架尝试数等同于实际 HTTP 次数。
- Provider 注入和资源：支持 adapter/factory、client/factory；借用 client 默认不关闭，自建 client 默认关闭。借入 fallback 保留其策略和生命周期所有权。
- Canonical 兼容：LLMResponse 保留 content/tool_calls/reasoning_content 旧构造入口与只读访问属性；与显式 message 混用会报错。usage BaseModel 可转换，缺失 usage 与明确零 usage 可区分。

上述判断是本地实现复核，不替代真实 provider 兼容验收。三个反例均未发送网络请求。
