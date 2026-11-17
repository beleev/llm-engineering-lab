# 异步运行 · 事件、取消与恢复

[![异步运行 · 事件、取消与恢复](../../docs/screenshots/agent-async-runtime-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/async-runtime)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/async-runtime)

## 直觉

用户点了取消, 工具究竟停在哪一步? 它刚提交的那笔钱怎么办?

demo 用真实的 asyncio 任务模拟一个会往账本里记 10 元的工具, 分三种时序跑: 跑完、提交前被取消、提交后被取消。

- 取消是两段式: 用户发出 cancel-request 后, 要等执行器回一个 cancel-ack 才算真的停了。
- 已提交的副作用不回滚: 提交后再取消 (cancel-after), 账本里仍是 10 元; 关掉账本重开、再提交一次, 还是 1 笔。靠的是 SQLite 里以 invoice-1 为主键的幂等写入 (同一个键写两次只算一次)。

## 核心原理

### 从调用栈变成事件生命周期

本例把工具从同步函数改成事件驱动的任务, 每一步都记成一条事件。同步写法里, 工具是一个函数, 从头跑到尾, 中途谁也插不进来。改成事件驱动后:

1. 任务由外部事件唤醒。

2. 跑到 await release.wait() 这一行挂起 (让出控制权, 等别人把 release 这个开关打开)。

3. 再由结果唤醒继续跑。

Runtime.record 把每一步记成一条事件, 每条事件都标着它属于哪个任务、哪一代 (generation, 见下面 "流式输出也需要代际" 那条)。状态有:

- working: 被唤醒开始干活。
- committed: 已往账本提交。
- stream: 流式输出了一个 token。
- completed: 正常结束。
- cancel-request: 用户要求取消。
- cancel-ack: 执行器确认已停。

complete 模式 (不打断) 的事件序列是 working → committed → stream × 2 → completed。每条事件在记进内存的同时也写进 SQLite 的 events 表, 进程没了也能查。

### 取消要等执行器确认

asyncio 里取消一个任务是调 Task.cancel(), 但它不会立刻停下对方。它只会在对方下一次 await (可取消点) 时在那里抛出 CancelledError。调用方要是发完 cancel 就当对方停了, 对方可能还在跑。

本例的 interrupt 分三步:

1. 把 generation 加 1 (让旧一代作废)。

2. task.cancel()。

3. await 这个任务, 直到执行器在 except CancelledError 里记下 cancel-ack 才返回。

有了这个确认, 旧 generation 的 token 不会再输出: cancel-after 模式里 streamed 为 0。

已结束任务的取消是幂等的: task.done() 时 interrupt 只返回之前有没有 ack 过, 不伪造一条新确认。这种在 await 处被自动打断的保证只对 asyncio 协程成立; 阻塞线程、外部服务和物理设备没有, 要自己设计确认机制。

### 提交状态独立持久化

提交过的副作用记在独立的账本里: 取消不回滚它, 重试也不多记。Journal 是这个账本, 底下是 SQLite。operations 表以 key 为主键, 本例的键是 invoice-1, 绑定金额 10。commit 分两步:

1. 用 INSERT OR IGNORE 写入: 同一个键第二次写会被忽略, 不会多记一笔。

2. 写完再读回这个键的金额, 和这次要写的不一样就抛 "幂等键对应的参数变了"。

这种键叫幂等键: 同一笔操作不管重试几次, 账本里只有一条。

三种模式的结果:

- cancel-after: 执行器在挂起前就提交了 10 元, 之后才被取消。取消确认后账本仍是 10 元, 因为提交过的东西不回滚。
- 然后 demo 模拟 "响应丢了、进程重开": 关掉账本再打开, 按同一个键 invoice-1 再提交一次。恢复前后都是 10 元、1 笔。
- cancel-before: 还没提交就被取消, 账本 0 元 0 笔。

恢复后要做的事是先查账本里有没有这笔收据, 不能凭 "超时了" 就断定没执行过。

### 流式输出也需要代际

流式输出的每一块也要核对 generation, 过期的不再输出。模型吐出的文字块、工具的进度、语音输出都是一条条事件。本例把 "已"、"处理" 两个 token 当流式事件: 每出一个之前, 先比对自己启动时记下的 generation 和运行时当前的 generation, 不一样就说明中途被打断过, 直接 return, 不再输出。

界面上展示的是已经发生的进度, 不把局部 token 当最终承诺。

预算也要防打断。write 分三步:

1. 先检查 generation 没过期、used 没超过 budget=2。

2. 然后 used += 1。

3. 之后才真的提交。

这几行中间没有 await, 是一个无法被打断的临界段, 所以不会出现 "刚检查完预算就被切走, 回来时预算已被别人用掉" 的情况。

### 长期任务需要更多恢复信息

本例只展示单进程里的三样东西: generation、幂等键和事件表。生产环境里跑几小时的任务还要:

- 持久化检查点 (跑到哪了)、租约 (谁在跑、多久不续就算死)、输入版本和审批状态。
- 分清 "外部通知" 和 "用户新指令"。

虚拟身份和工具凭证要按执行身份隔离, 任务换人跑不能带着别人的凭证。

### 同步模型怎样接异步事件

大多数模型 API 是同步的: 发一个请求, 等一个回答, 中间插不进事件。运行时可以这样接:

- 把事件排队, 等到安全的轮次边界再注入上下文。
- 或者中止旧请求, 换一个新 generation 重发。

就算模型支持原生的事件接口, 运行时仍要负责:

- 任务归属 (这条结果属于哪个任务)。
- 迟到结果过滤 (旧 generation 的丢掉)。
- 预算。

后台通知不能直接升级为用户授权: 一条 "任务完成了" 的事件不等于用户说了 "可以继续"。

### 读代码的顺序

- `demo.py:Journal`
- `demo.py:Runtime`
- `demo.py:run_case`

## 运行

```bash
python -m llm_agent.m29_async_runtime.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `complete, cancel-before, cancel-after` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

用的是真实的 asyncio 任务和磁盘上的 SQLite 文件。竞态 (两件事谁先谁后) 靠 asyncio.Event 屏障精确复现 (ready 和 release 两个开关), 不靠 sleep 猜时序。

没有网络分区、分布式租约或进程强杀测试; "重开" 重开的是账本文件, 不是自动恢复任意 Python 调用栈。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

先提交再收到取消确认, 账本里的那笔写入怎样处理?

<details><summary>答案</summary>

仍存在, 补偿需要另一个操作。取消停止未来动作, 不撤销过去提交。补偿要有自己的授权和审计。

</details>
