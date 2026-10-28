# 公开文档修复

## 交付

重写 README、USAGE、MIGRATION 与公开 MkDocs 页面：以当前 API 与执行契约组织入门、模型资源所有权、工具注册、会话、结构化输出、权限/审批、恢复、委派、图、存储、插件/技能、协议及运维。保留完整能力地图和历史 1.0 发布记录，删除已失效示例、旧架构类名、固定模块/测试计数与无依据的差异化宣传。

新增 protocols/governance/operations 页面以及四段可执行离线示例；导航覆盖新增页面。旧报告与 superpowers 记录未改。

迁移明确：provider 别名弃用兼容；canonical/旧响应构造共存；gateway.adapter 仍可用但新集成优先 client/factory；空兼容 extras；默认 minimal；run/stream 同会话；structured tuple 及挂起 output=None；fork 默认隔离模拟；unknown 不重试；自定义检查点需要原子 claim；计划/后台审批恢复与进程内句柄边界；MultiAgentGraph 非完成异常结果；A2A 0.3 message/send 与 tasks/send 弃用兼容；旧 MCP SSE 独立于 HTTP。

## 验证

`pytest tests/test_documentation_hardening.py -q`：4 passed。

测试直接执行文档四个完整代码段（工具参数、同会话 run/stream、graph reducer、结构化输出），逐条校验公开 Python 示例的 wuwei imports、Markdown 相对链接与 MkDocs 导航文件，并阻止删除 API 的示例回流。没有网络请求。

`python -m mkdocs build --strict --site-dir /tmp/wuwei-doc-build` 未执行成功：当前虚拟环境没有 mkdocs。静态链接/导航检查已通过；最终站点构建仍需安装文档构建工具的环境执行，尚无站点渲染验收。

依赖声明、CI 与在线 --live 机制由主执行 owner 集成；文档注明测试 SDK 下限及外部服务需要单独验收，不将离线例子当远端互操作证据。

## 主执行最终验收补充
已安装声明的docs依赖，修正离线代码围栏的渲染及排除内部审查记录后，2026-10-28 `mkdocs build --strict`已通过；此前未构建状态已关闭。
