# 流式会话锁租约

状态：已实现（`wuwei/runtime/stream_bridge.py`）。对应问题清单 R-9。

## 现象

`AgentRunner._iter_events` 在 `async with self.session.lock` 内部 `yield` 事件。消费方 `async for` 中途 `break`、所在任务被取消、或框架丢弃生成器时，并不会自动触发 `aclose()`。只要还有引用持有这个生成器，锁就一直被占用，该 `session_id` 后续所有 `run()` 无限阻塞，日志里没有任何线索。

## 根因

锁的生命周期绑定在**生成器对象**上，而业务上需要它绑定在**一次运行**上。Python 的异步生成器只有被回收或被 `aclose()` 时才在 `yield` 点抛 `GeneratorExit` 并执行 `finally`；这两件事都不由框架控制。因此"在生成器里 `with lock`"这种写法，把正确性交给了调用方的自觉。

先把话说清楚：`contextlib.aclosing` 只能修自己代码里的调用点，治不了三方框架、`break`、以及被丢弃的生成器。它属于约定，不属于机制。

## 三个方案

| 方案 | 做法 | 能解决什么 | 代价 |
|---|---|---|---|
| A. 消费契约 | 文档要求调用方 `async with aclosing(stream)` | 只覆盖自家代码 | 无强制力，线上照旧挂死 |
| B. 有界等待 + 诊断 | 锁获取加超时，报出持有者 `run_id` 与持续时长 | 把"永久挂死"变成"可诊断失败" | 不释放锁，会话仍需人工或进程重启 |
| C. 生产者任务 + 队列 | 锁由后台 task 持有，生成器只消费队列；消费方消失时 task 自毁并释放 | 根治，且能区分"消费者走了"和"模型卡住" | 改动集中在 runner，需要新增超时参数 |

**推荐 B + C 分两次落地**：先上 B（一个 patch，风险极低），再上 C（结构性修复）。只做 A 等于没做。

## C 的目标形态

```text
stream_events()
   └─ _RunProducer(task)  ── async with session.lock ──> _execute_events()
          │  事件写入 queue(maxsize=64)
          └─ 心跳：put 阻塞超过 consumer_idle_timeout → 判定消费者消失
   _iter_events()  ← queue.get()   ← 消费者
```

关键点：

1. **锁在 producer 的 `async with` 里**，`finally` 一定执行，与消费者生死无关。
2. **背压用队列**，`maxsize=64`。慢消费者只是让 producer 阻塞在 `put`，不丢事件。
3. **消费者消失的判定是 `put` 超时**，不是锁超时。默认 `consumer_idle_timeout=30s`；设 `None` 关闭（退回当前行为）。
4. **收尾语义**：消费者消失 → producer 取消 `_execute_events` → 落 `cancelled` 检查点 → 释放锁。**不要标 `unknown`**：`unknown` 的语义是"外部副作用结果不明"，而消费者断连时工具调用要么没开始、要么已有结果，谎报 unknown 会让会话进入 `_blocked_run`，反而需要人工对账才能恢复。
5. **正常路径零变化**：`run()` 走 `_collect` 全量消费，行为与现在一致；顺序由单队列保证。

## 需要明确的边界

- `consumer_idle_timeout` 与 `LLMGateway.timeout` 是两回事：前者衡量"消费者还活着吗"，后者衡量"模型还活着吗"。不要合并成一个参数。
- producer 不进入 `_close_background_tasks` 的清单，它是运行本身，不是委派任务。
- 取消传播：消费者任务被取消 → 关闭队列 → producer 收到取消 → 沿用现有 `except CancelledError` 分支落 `cancelled` 检查点。
- 与 `SQLiteSessionStore` 的 lease 不要混淆：那是跨进程的存储租约，这里只是进程内 `asyncio.Lock`。进程内锁不需要 token，因为只有一个事件循环。
- 若将来支持多 worker，进程内锁一律不成立，必须走 `SessionBusy` + 租约，这一层已经由持久化模块提供。

## 验收标准

1. 消费 3 个事件后 `break` 且不 `aclose`，在 `consumer_idle_timeout=0.05` 下，另一个 run 能在 200ms 内取到锁。
2. 正常消费到 `run_end`，锁释放，无残留 producer task。
3. `run()` 全量收集路径的返回结果与改造前逐字段一致。
4. 取消消费者任务：producer 被取消，session 的 `_blocked_run` 未被设置，检查点 `kind="cancelled"`。
5. 超时判定只影响生产者，不影响正在执行的工具：工具执行到一半时消费者断连，等工具结果提交后再取消，避免制造新的 unknown。

第 5 条是实现的难点，也是不能简单用"锁超时"代替的原因：锁超时会在工具执行中途释放锁，让第二个 run 进入同一会话，比挂死更糟。

## 实现

`bridge()` 把生产者放进任务，事件经 `queue(maxsize)` 传给消费者：

| 位置 | 变化 |
|---|---|
| `AgentRunner._iter_events` | 改为 `bridge(self._locked_events(...))` |
| `AgentRunner._locked_events` | 持锁运行，并在 `finally` 中 `aclose()` 内层生成器 |
| `Agent._stream_session` | 同样改为 `bridge(...)`，让 store 锁与租约也归属任务 |
| 配置 | `Agent(stream_idle_timeout=30.0)` / `AgentRunner(consumer_idle_timeout, event_queue_size)` |

两处实现细节值得记住：

1. **`async for` 不会关闭内层生成器**。`_locked_events` 在 yield 点被关闭时，`_execute_events`
   只是被丢弃；若不在 `finally` 里显式 `aclose()`，它的取消分支会在锁释放之后才跑，与下一个
   run 撞车。这是原实现就存在的隐患，桥接层顺手修掉了。
2. **"准备中"不等于"已派发"**。`_execute_events` 在 `tool_states` 里先写 `preparing` 再落
   `tool_running` 检查点，此时工具可能尚未派发。中断时若一律标 `unknown`，用户会为一个从未
   执行过的工具被要求对账。实现里用 `dispatched_calls` 记录真正进入执行的调用：已派发的标
   `unknown`（需对账），未派发的补一条 `not_executed` 工具结果（可安全重试），会话不被封死。

## 未覆盖

`consumer_idle_timeout=None` 退回旧的直接透传模式（锁仍由生成器持有），用于需要逐字节
零缓冲的场景；该模式下 R-9 的风险仍然存在。
