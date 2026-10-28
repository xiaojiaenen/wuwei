# 持久会话、后台任务与安全恢复

## 会话跨进程重启保留历史

```python
from wuwei import Agent, SQLiteSessionStore

store = SQLiteSessionStore('.wuwei/sessions.db', namespace='tenant-a', ttl_seconds=86400)
agent = Agent(llm=llm, session_store=store, load_builtins=False)
# 在 async 函数中：
# session = await agent.load_session('chat-1') or agent.create_session('chat-1')
# result = await agent.run('继续之前的讨论', session=session)
# await agent.release_session('chat-1')
```

`release_session` 只淘汰进程内缓存；`delete=True` 才删除持久历史。有审批或未知动作的会话不能直接删除；没有持久store时，也不能淘汰它的唯一内存历史。默认会话仍复用，显式的新session仍隔离。无store时保留原有进程内行为。

store的checkout原子领取会话租约并在执行中续期，提交时校验owner/token/revision，过期owner不能覆盖新状态。同一Agent内按session排队；不同worker争抢会话时抛`SessionBusy`。worker无正常提交就丢失租约，后续执行会保留unknown阻塞，不把执行前的旧历史当作新任务重跑。

TTL只清理正常闲置会话的可加载历史，不自动抹掉待审批/unknown事实。使用`Agent.aclose()`或`async with Agent(...)`清理本地缓存；执行中的Agent不能关闭。借入LLM默认由调用方管理；`Agent.from_env()`创建的LLM由Agent关闭。数据库连接逐操作关闭，不持有后台连接池。

namespace由经过鉴权的宿主选择，不要接收模型传来的tenant_id就直接用作namespace。这是存储分区，不是用户鉴权组件。ArtifactStore的owner也遵循相同规则。

## 一个任务只由一个worker领取

```python
from wuwei import JobService, SQLiteJobStore

jobs = SQLiteJobStore('.wuwei/jobs.db', namespace='tenant-a')
service = JobService(jobs, agent_factory)  # 每次返回新Agent，工具/依赖由宿主注入
# 在 async 函数中：
# job = await jobs.submit('分析报告', idempotency_key='business-request-123')
# record = await service.execute(job.id)
# record = await jobs.get(job.id)
```

同一幂等键复用原任务；输入/会话不一致则报错。执行中的任务有续租和取消检查；已完成任务重复execute只返回原结果。崩溃后过期的running任务转为unknown，不自动重新调用工具。`cancel`对排队/等待审批的任务立即生效，对执行中的任务写入取消请求，由worker清理当前运行；已发生的外部动作不能靠取消撤销。

审批挂起后，可在新进程创建同一store和checkpointer，再提交信号：

```python
# await jobs.signal(job.id, approved=True, operator='reviewer-id',
#                   signal_id='decision-123', reason='已核对申请')
# record = await service.execute(job.id)
```

信号由可信宿主确认审核人权限后调用。相同signal_id和决定重复提交不会重复审批，复用id却改变决定会报错。完整中断状态由Agent检查点恢复，不能只持久化一句任务描述后从头再跑。

`JobService.reconcile(job_id, checkpoint_id=..., tool_call_id=..., outcome=ToolOutcome(...), operator=..., evidence=...)`将权威外部结果记入Agent检查点并继续，更新job真值。unknown任务不会通过普通execute自动重试；仍不确定时不要提交假成功。多个未知动作逐个对账，使用返回的replacement checkpoint继续。详情见[运行契约](../runtime-contracts.md)。

SQLiteJobStore/JobService是轻量单任务执行与持久真值接口，不是另造一个分布式调度平台。

## DBOS负责持久排队

安装`pip install 'wuwei[dbos]'`。当前适配DBOS 3.3.x的官方workflow/step与DBOSClient接口；应用负责数据库配置、启动和销毁。

```python
from dbos import DBOS, DBOSClient
from wuwei import DBOSJobBackend, build_dbos_workflow

url = 'sqlite:///.wuwei/dbos.sqlite'
DBOS(config={'name': 'my-worker', 'system_database_url': url, 'enable_otlp': False})
workflow = build_dbos_workflow(service)  # launch前注册workflow和step
DBOS.launch()
DBOS.register_queue('wuwei')           # DBOS 3.3：launch后注册队列
client = DBOSClient(system_database_url=url)
backend = DBOSJobBackend(service, client)
# 在 async 函数中：
# job = await backend.submit('分析报告', idempotency_key='business-request-123')
# print((await backend.get(job.id)).status)
# await backend.cancel(job.id)
# 退出应用时 client.destroy(); DBOS.destroy()
```

Agent运行位于DBOS step中，不把模型/数据库调用直接放在确定性workflow函数里。每个审批续跑使用独立delivery identity；enqueue失败后可使用同一业务幂等键重提。DBOS并不让外部动作自动获得exactly-once：本地未知状态仍阻止重放。

离线验收使用真实DBOS SDK、SQLite队列及fake LLM，验证投递、执行、重复提交；没有连接生产DBOS/Postgres或真实模型。

## 不显示正文的诊断

```bash
wuwei-debug .wuwei/checkpoints.db --summary CHECKPOINT_ID
```

输出run/session/trace标识、状态、步数、用量和工具状态，不输出提示词、工具参数或任意扩展metadata。需要查看历史正文时才显式使用已有`--show`。
