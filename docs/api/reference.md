# API 模块索引

| 模块 | 入口 |
|---|---|
| wuwei.agent | Agent、AgentSession、SubAgent、AsyncSubAgent、MultiAgentGraph |
| wuwei.core | BaseMessage/AIMessage/ToolMessage、Runnable、错误类型 |
| wuwei.llm | LLMGateway、LLMResponse、LLMResponseChunk、AgentEvent、AgentRunResult |
| wuwei.tools | ToolRegistry、Tool、ToolParameters、ToolExecutionPolicy、ToolExecutor |
| wuwei.runtime | AgentRunner、EventBus、ActionLog、InterruptPayload |
| wuwei.graph | StateGraph、State、END、MemoryCheckpointer、SQLiteCheckpointer |
| wuwei.graph.reducers | overwrite、extend、merge_dict、add、first_non_empty |
| wuwei.middleware | MiddlewareStack、HITL、预算、Ledger、Memory、RAG、Storage |
| wuwei.security | PermissionConfig、PermissionMiddleware |
| wuwei.planning | Planner、PlanExecutor、Task、TaskList、PlanRunResult |
| wuwei.plugin | Plugin、PluginContext、PluginManager |
| wuwei.skill | Skill、SkillManager、SkillProvider；fs_provider.FileSystemSkillProvider |
| wuwei.memory | Context、FileStorage、MemoryStore、KnowledgeStore、embedder |
| wuwei.vectorstore | 内存与可选远程向量后端 |
| wuwei.fs | InMemoryFS、LocalFS、make_vfs_tools、make_vfs_plugin |
| wuwei.sandbox | LocalPythonDriver、DockerDriver、E2BDriver、get_sandbox |
| wuwei.serve | build_a2a_app、build_ag_ui_app、build_mcp_server、serve_mcp |
| wuwei.gateway | 多平台消息适配器与 Webhook |
| wuwei.decision | DecisionClient、decision_approval_provider |
| wuwei.evals / wuwei.debug | 评估与检查点调试 CLI |
| wuwei.parsers | JSON/Pydantic/列表输出解析器 |
| wuwei.config | load_agent_config、create_agent_from_config |

返回值与状态见[Agent](../core/agent.md)，实际执行示例见[离线示例](../examples/offline.md)。
