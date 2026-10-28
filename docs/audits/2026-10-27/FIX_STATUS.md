# 修复与验收状态（2026-10-28）

候选：`codex/framework-hardening-20261028`。范围为最初审查的本地3.2工作树，连同其尚未提交的新模块；没有丢弃这些原有改动。用户已授权完成后推送，本次只推送Git分支，不发布PyPI或部署在线服务。

## 原18组问题的落点

| 问题 | 修复与实际验收入口 |
|---|---|
| 1 跨步权限/配额丢失 | JSON run governance；跨步及同批读后写、配额和审批恢复不重复扣额回归 |
| 2 路径边界 | 规范路径/目录匹配，拒绝相邻前缀、..和相对规则符号链接逃逸；未知工具默认拒绝，显式capability/tool_categories/allowed_tools |
| 3 多工具中断/重复执行 | 完整剩余批次；每个调用独立审批；Memory/SQLite原子claim和最新head检查；重启续跑/重复resume/并发批次崩溃回归 |
| 4 恢复绕过治理 | 不再skip_before_tool；决策绑定id/name/参数；先完成改写再执行guard；保存run/session/累计消耗/祖先预算 |
| 5 安全配置失败继续启动 | 构造错误直接终止；YAML支持嵌套agent、permission/budget与SQLite配置；未知字段拒绝 |
| 6 Responses默认不可用 | provider参数映射；显式max_output_tokens优先；SDK签名/请求抓取回归 |
| 7 中间件结果协议 | ToolMessage终止before_tool链；dispatcher排在guard之后；子代理使用相同协议 |
| 8 动态工具不可见 | 每次模型调用取registry；MCP发现缓存与每次概览注入分离、同一视图去重 |
| 9 会话连续性 | run/stream默认会话一致；AG-UI线程映射；MCP任务默认隔离、显式session_id复用 |
| 10 graph流式reducer | invoke/stream同样归并；公开END导出；文档离线图示例实际执行 |
| 11 Anthropic统计/系统指令 | 官方SDK形状usage、输入输出/cache聚合；保留全部system及signed reasoning；截断不当成功 |
| 12 存储/序列化/分页 | 注入session_id；ActionLog显式JSON化；保留恢复/unknown保护标记；SQLite稳定游标；异步IO |
| 13 账本/CodeMode参数 | 生成必需参数schema；Pydantic+JSON Schema验证；list[str]修复；nested CodeMode工具走治理回调 |
| 14 子代理旁路 | 统一Runner；继承权限/预算/wrapper；父子实际调用汇总；嵌套审批续跑、取消不确定副作用传到父级 |
| 15 structured旁路 | 同会话、同middleware/工具、同预算/统计/checkpoint；校验纠正属于同run；返回typed output+AgentRunResult |
| 16 未完成报成功/计划图 | incomplete/max_steps；DAG验证；planner与执行费用共同计入；PlanExecutor.resume仅调度未完成下游、不盲重试 |
| 17 fork修改原会话 | 新session/run/lineage；改写同步到历史；默认模拟工具，实际副作用显式opt-in；不消费原检查点 |
| 18 账本全量保真 | 缓存与持久原文/内存原文分离；淘汰后仍可召回；pin持久化 |

## 下游3.0.2反馈

P0-1/2的专用旧适配器通过显式弃用别名迁移到统一兼容端点；响应模型统一并兼容旧构造。P0-3 usage-only、P1-6缺usage、P1-7参数碎片旧问题保留回归。其余有效反馈均已落实到模型/运行时修复；原反馈的P1-9“连续user必然400”和P2-8“正常Chat输出list必崩”已按官方契约纠正，不为错误前提增加补丁。

额外修复：确定性错误不重试；整体deadline跨fallback共享；SDK/client参数分离及client/factory注入/资源所有权；thinking按capability保留；TypedDict兼容Python3.11；生成参数不再忽略；内部记忆/压缩模型调用计费，缓存按session隔离；追踪资源不写入检查点并在退出时清理；EventBus有序投递与drain/close。

## 协议与文档

- A2A明确实现0.3，而不是伪装同时兼容1.0；标准message/send、AgentCard/响应/错误/状态由官方a2a-sdk0.3.26类型验证。tasks/send保留为旧接口。
- MCP使用官方SDK2.x握手；legacy SSE与Streamable HTTP分开；失败发现清理连接；MCP出口返回interrupt/usage_available等，恢复审批仍由宿主可信控制面处理。
- AG-UI流式与降级入口均映射会话/真实终止状态，并清理消费中的流。
- README/USAGE/MIGRATION与公开文档已改成当前API，离线示例执行；旧设计方案标明历史性质。
- `pytest`默认离线；真实模型集成需显式`--live`。CI不再把lint错误吞掉。打包包含测试conftest，避免sdist误跑线上测试。

## 验收记录

最终集成验收（2026-10-28）：

| 环境/检查 | 结果 |
|---|---|
| Python3.11.15，隔离新安装依赖 | 411 passed、9 skipped，5.69s |
| Python3.12.13，原项目环境/测试下限SDK | 411 passed、9 skipped，5.63s |
| Python3.13.14，隔离新安装依赖 | 411 passed、9 skipped，5.64s |
| SDK跨环境 | 下限环境OpenAI3.23/Anthropic1.11/MCP2.2；新环境OpenAI3.40/Anthropic2.3/MCP2.3；官方A2A0.3.26 |
| Ruff关键错误 + git diff --check | 通过 |
| MkDocs --strict | 通过，输出/tmp/wuwei-doc-build-20261028 |
| uv build | wheel与sdist通过；核对sdist包含离线测试开关和文档示例，不包含内部审查记录/密钥 |
| 独立wheel导入/Agent/类型化工具smoke | 通过，未使用源码路径替代wheel |
| CLI | evals/debug --help通过 |

此前局部证据分别见MODEL_FIX_REPORT、BOUNDARY_FIX_REPORT、DELEGATION_FIX_REPORT、MIDDLEWARE_FIX_REPORT，以及独立复核报告。独立复核发现的并发批次重放、审批后参数改写、后台不确定副作用和跨版本配置问题均补了回归。9项跳过为未提供相应可选SDK/执行环境的测试，不计为已通过。真实模型集成文件默认不收集，需显式--live且另行准备凭据/业务验收。

最终附加验收：MCP出口保留恢复payload；记忆提取跨同session的多次run计数，并隔离不同session；内部记忆/压缩调用与主调用共享预算，未知usage不再冒充零成本。

## 运行边界与使用条件

- 不确定副作用为unknown，阻止自动重放和同会话继续；线程取消无法保证外部动作撤销。先核对目的端结果，再由应用重置/重建会话。没有宣称跨系统exactly-once。
- CodeMode仍是可信进程内代码执行，不是OS沙箱。嵌套需要审批的动作返回direct_approval_required，要求改为直接工具调用；不重播整段代码假装可恢复。需要真正隔离时选择Docker/E2B。
- 后台handle清单是进程内能力；child审批检查点可持久化，父运行退出会清理其后台任务。MultiAgentGraph保留字符串API，未完成成员通过带result/interrupt的DelegationStoppedError交给宿主，不宣称团队级持久化作业平台。
- 官方协议类型、离线SDK/MockTransport、内存/SQLite及进程崩溃模拟已验证；真实提供商、远端MCP、Docker/E2B、外部向量数据库和业务平台仍由部署环境验收。没有调用用户真实模型密钥或线上写服务。
