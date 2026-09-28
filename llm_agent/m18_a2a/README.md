# M18 — A2A (Agent2Agent): agent 找 agent

两个进程内的 agent 用 A2A 的形状协作:
1. 出差助理读 Agent Card 找到报销 agent, 用 `message/send` 交任务。
2. 对方缺出差日期, 就进入 `input-required` 反问。
3. 助理补上后, 任务 `completed` 并交回 artifact。

任务状态机写死, 非法转移在本地和线上都被拒绝。

## 直觉

m09 的 MCP 让 agent 调用别人提供的**工具**: 参数一次给齐, 一次调用拿回结果, 无状态。

但有些"别人"本身就是 agent。以报销 agent 为例:
- 它会发现你没给日期, 反问你。
- 它会跑一阵子 (审批、查额度)。
- 它内部用什么工具、跑几轮 loop, 也不该暴露给调用方。

把它硬包成一个 MCP 工具, 要么 schema 里把所有可能追问的字段都列成必填, 要么丢掉多轮能力。

A2A 解决的是这个: agent 之间以**任务**为单位协作。任务有 id、有状态、可以多轮来回, 最终产物放在 artifact 里。调用方只看到消息和产物, 看不到对方内部。

**MCP 是 agent 调工具, A2A 是 agent 找 agent。**

## 核心数据结构与控制流

全部在 `m18_a2a/demo.py`:

| 部分 | 符号 | 做什么 |
|---|---|---|
| 状态机 | `TRANSITIONS` / `Task.to` / `InvalidTransition` | 转移表写死; 终态 (completed / failed / canceled) 的出边为空 |
| 任务 | `Task.to_json` | `{kind:"task", id, contextId, status:{state, message?}, history, artifacts}` |
| 服务端 | `ExpenseAgent` | `card` (Agent Card); `handle(wire)` 是 JSON-RPC 分发: `message/send` / `tasks/get` / `tasks/cancel` |
| 服务端内部 | `ExpenseAgent._work` | 缺日期 → `input-required`; 否则用 `core` 的 `Agent + CalculatorTool` 算合计 → 超限 `failed`, 否则 `completed` + artifact |
| 传输 | `Network.get_card` / `post` | 进程内的 "HTTP": 只收发字符串, 记录线上每条 JSON (`wire`) |
| 客户端 | `TripAssistant.discover` / `send` / `delegate` | 按 skill tag 挑 agent; `input-required` 时查自己的日程回答, 用同一个 `taskId` 续上 |
| 错误码 | `TASK_NOT_FOUND` / `TASK_NOT_CANCELABLE` / `UNSUPPORTED_OPERATION` | -32001 / -32002 / -32004 (A2A 规范的 JSON-RPC 错误表) |

```
TripAssistant                                   ExpenseAgent
  GET card ─────────────────────────────────►  {name, skills:[{id:"file_expense", tags:["报销",...]}], ...}
  message/send "报销上海出差: 机票 1280..." ──►  submitted → working → input-required  "请提供出差日期"
  message/send taskId=task-1 "出差日期 ..." ──►  working → (内部 Agent + calculator) → completed
  ◄──────────────────────────────────────────  artifacts:[{parts:[{kind:"data", data:{claim_id, date, total}}]}]
```

状态机:

```
submitted ──► working ──► completed
    │            │ ▲
    │            ▼ │
    │       input-required
    └──────► failed / canceled  (submitted、working、input-required 都可以到这两个终态)
终态没有出边: completed → working、failed → canceled 都是非法的
```

## MCP 与 A2A 对照

| | MCP (m09) | A2A (本章) |
|---|---|---|
| 对方是什么 | 工具 / 资源的提供者 | 另一个 agent |
| 交互单位 | 一次 `tools/call` | 一个 Task (多轮消息) |
| 状态 | 无状态, 调用即返回 | submitted / working / input-required / 终态 |
| 缺信息时 | 本地 schema 校验失败, 或工具返回 `isError` | 进入 `input-required`, 对方主动反问 |
| 发现 | `tools/list` 列出每个工具的 inputSchema | Agent Card 列出 skills、tags、输入输出模式 |
| 对方内部 | 工具就是一个函数 | 不透明: 用什么模型、工具、几轮 loop 都不暴露 |
| 结果 | `content` + `isError` | 状态消息 + artifacts (text / data parts) |
| 两者关系 | 一个 A2A agent 的内部完全可以用 MCP 调工具 | |

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m18_a2a.demo
```

```
[1] 发现: 读 Agent Card, 按 skill tag 挑人
  card                    : {'name': 'expense-agent', 'url': 'a2a://expense', 'defaultOutputModes': ['application/json']}
  skills                  : [('file_expense', ['expense', 'reimbursement', '报销'])]

[2] 多轮任务: 对方缺信息 → input-required → 补充 → completed + artifact
    input-required: 请提供出差日期 (YYYY-MM-DD)。 → 查日程回答 2026-09-15
    completed: 已提交报销单 EXP-task-1, 合计 2630 元。
  artifact                : {'claim_id': 'EXP-task-1', 'date': '2026-09-15', 'total': 2630}
  状态轨迹                    : submitted → working → input-required → working → completed

[3] 不透明: 对方内部跑了 agent loop + calculator, 调用方的线路上一个字也没有
  对方内部工具调用                : ['calculator']
  线路上出现 calculator        : False

[4] failed 也是正常结局: 超过报销上限
    input-required: 请提供出差日期 (YYYY-MM-DD)。 → 查日程回答 2026-09-15
    failed: 合计 7700 元, 超过单次报销上限 5000 元, 请走特批流程。

[5] 非法状态转移一律拒绝
    本地  completed -> working: rejected
    本地  submitted -> completed: rejected
    本地  input-required -> completed: rejected
    本地  failed -> canceled: rejected
    线上  message/send  -> error -32004: task task-1 is completed; cannot continue (completed -> working)
    线上  tasks/cancel  -> error -32002: task task-1 is completed (completed -> canceled)
    线上  tasks/get     -> error -32001: task task-404 not found
    线上  message/stream -> error -32601: Method not found
```

断言验证的内容:

- [1] 按 tag `报销` 挑中的 URL 是 `a2a://expense`。
- [2] 多轮任务:
  - 状态轨迹恰为 `submitted → working → input-required → working → completed`。
  - artifact 的 data 恰为 `{claim_id, date: 2026-09-15, total: 2630}` (1280 + 3 × 450)。
  - 线上每一条都是可解析的 JSON 字符串。
- [3] 对方内部 transcript 里有一次 `calculator` 调用, 线上记录里一次也没出现 `calculator`。
- [4] 超限任务结局是 `failed`, 没有 artifact。
- [5] 非法转移:
  - 四个本地非法转移都抛 `InvalidTransition`。
  - 四个线上请求分别得到 -32004 / -32002 / -32001 / -32601。最后一个是服务端没实现的 `message/stream`, 回的是 JSON-RPC 规范本身的 "Method not found"。
  - 被拒绝之后任务仍是 `completed`, 状态轨迹仍是 5 步 (拒绝没有产生任何副作用)。

## 与真实系统的差距

- **没有 HTTP**: 真实 A2A 走 JSON-RPC 2.0 over HTTPS, Agent Card 在 `https://<host>/.well-known/agent-card.json` (旧版本叫 `agent.json`)。这里用字符串收发模拟, 形状一致, 但没有鉴权、TLS、超时。
- **没有流式和推送**: 真实 A2A 有 `message/stream` (SSE 推 `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent`) 和 push notification (长任务回调 webhook)。这里 `message/send` 同步跑完才返回, 发 `message/stream` 得到 -32601。
- **状态不全**: 规范里还有 `rejected`、`auth-required`、`unknown`。转移表是本模块自己写死的教学版本, 规范本身没有给出这么严格的表。
- **Agent Card 没有鉴权声明和签名**: 真实 card 里有 `securitySchemes`, 可以签名防伪造。按 tag 选人也很粗: 真实系统会用描述做语义匹配, 或者有注册中心。
- **不透明是双刃剑**: 调用方看不到对方内部, 也就没法审计对方做了什么。跨组织协作时要靠合同、日志和 artifact 本身的可验证性, 这里没有涉及。
- **追问的回答是写死的**: `TripAssistant` 只会从 `calendar` 查日期。真实系统里调用方通常是 LLM, 要理解追问、决定自己答还是转问用户。
- 错误码取自 A2A 规范的 JSON-RPC 错误表。规范仍在演进, 以官方规范为准。

## 常见误区

- **"A2A 和 MCP 是竞争关系, 二选一。"** 两者解决不同层次的问题。[3] 里报销 agent 内部就在调工具 (这里是本地 calculator, 换成 MCP 工具也一样), 对外则以 A2A agent 的身份接任务。
- **"input-required 是一种失败。"** 它是正常的中间状态: 任务没结束, 同一个 `taskId` 还能继续。失败是 `failed`, 而 `failed` 也是正常结局 (超限是业务结果, 不是协议错误)。
- **"任务结束了还可以发消息改一下。"** 终态不可再动 ([5] 的 -32004)。要改就开一个新任务; 否则调用方拿到的 artifact 可能在它不知道的时候被改掉。

## 自测题

1. 为什么报销能力用 A2A 暴露, 而不是做成一个 `file_expense(amount, date)` 的 MCP 工具?

<details><summary>答案</summary>

做成 MCP 工具的问题:
- 调用方必须一次给齐所有参数。缺了日期, 只能拿到一个 `isError` 或校验失败, 再从头调一次。
- 对方的业务规则 (哪些字段必填、超限怎么办) 要么泄漏进 schema, 要么调用方根本不知道。

A2A 让对方以任务为单位工作: 缺信息就进入 `input-required` 反问, 调用方在同一个 `taskId` 上补充。对方内部怎么算、用了什么工具完全不暴露 ([3])。

代价是有状态: 要管理任务生命周期、处理非法转移。

</details>

2. `rpc_message_send` 收到带 `taskId` 的消息时, 先 `task.to("working")` 再 `task.history.append(msg)`。为什么这个顺序重要?

<details><summary>答案</summary>

如果任务已经在终态, `to("working")` 会抛 `InvalidTransition`, 被转成 -32004 返回。函数到此为止, 这条消息没有写进 history。

反过来先 append, 一个被拒绝的请求也会改掉已完成任务的历史。[5] 最后的断言 (状态仍是 completed、轨迹仍是 5 步) 就是在验证"被拒绝的请求没有副作用"。

</details>

3. [4] 的超限任务也先经历了 `input-required`。如果服务端先算合计、发现超限直接 `failed`, 需要什么前提?

<details><summary>答案</summary>

前提是合计只依赖机票和住宿金额, 不依赖日期。这样服务端可以在追问日期之前就判定超限, 直接 `working → failed`, 省掉一轮往返。

本模块的 `_work` 先查日期, 是为了让两个场景走同一条路径。真实 agent 应该先检查"不可能成功"的条件再追问, 否则用户补完信息才被告知失败。

这是 agent 内部的策略问题, 协议两种都允许 (`working` 可以到 `failed`, `input-required` 也可以)。

</details>
