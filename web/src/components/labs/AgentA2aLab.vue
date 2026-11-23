<!--
  A2A 任务状态机 (对应 llm_agent/m18_a2a/demo.py 的 TRANSITIONS / Task.to / ExpenseAgent / TripAssistant.delegate)。
  只讲一件事: A2A 的交互单位是一个有状态的 Task; input-required 能转回 working (回边), 终态没有出边。
  有回边, 所以不用 DagView, 手写一张 8 个节点的小状态图 (A2A v1.0 的 8 个状态, 不含 UNSPECIFIED)。
-->
<template>
  <LabFrame
    title="A2A — 一个任务怎么在两个 agent 之间来回, 结束后为什么不能再改"
    sub="- 模拟什么: 两个 Agent 之间的跨服务协作流程。出差助理 (调用方) 将差旅报销委托给报销 Agent (服务方)。A2A (Agent2Agent) 协议将交互基本单元定义为带状态的任务 Task, 而非单次无状态函数调用: 每个 Task 拥有唯一 id 与生命周期状态, 支持多轮追问补充信息, 并将最终产出存入 artifact。本实验台完整回放 demo 中的一次报销任务交互, 并支持手动触发状态转移以检验协议约束。
- 图上是什么: 上方为协议状态转移图, 下方为步骤说明与通信报文:
  - 八个圆角框: 对应 A2A v1.0 的八种任务状态, 包括 submitted (已提交)、working (处理中)、两个中断态 input-required (等调用方补信息) 与 auth-required (等调用方补凭据), 以及四个终态 completed (已完成)、failed (执行失败)、canceled (已取消) 与 rejected (对方决定不接)。
  - 有向箭头: 表示协议允许的状态转移边; 右侧四个粗边框节点为终态, 终态不存在任何出边。
  - 高亮标记: 高亮节点代表任务当前所处状态, 高亮箭头代表本次任务已走过的转移路径。
  - 灰色说明文字: 图下方的灰字概述当前步骤发生的交互事件。
  - 报文代码块: 底部展示线上实际收发的 JSON-RPC 报文, 其中「→」表示助理发出的请求, 「←」表示报销 Agent 返回的响应。
- 可以做什么: 提供三类交互操作:
  - 「机票」「住宿晚数」「每晚」三个滑杆: 调整报销明细金额, 报销合计等于机票加住宿晚数乘每晚单价; 当合计金额超过 5000 元上限时, 任务终态会由 completed 变为 failed。
  - 「改成私人消费」: 把请求换成 demo [6] 的私人聚餐, 对方在 working 之后直接 rejected, 不追问也不计算。
  - 「线上写法」: 在 demo 沿用的 v0.3 写法 (message/send、kebab-case 状态名、kind 字段) 与 v1.0 写法 (SendMessage、TASK_STATE_* 枚举名、去掉 kind、ROLE_ 前缀) 之间切换报文显示, 状态机本身不变。
  - 步骤播放器: 通过「上一步 / 下一步」或时间轴回放三步交互。第 1 步助理读取服务方的 Agent Card 并按 skill 标签选中报销 Agent; 第 2 步发送首条 message/send, 服务方发现缺少出差日期后停在 input-required 状态发起追问; 第 3 步助理携带同一 taskId 补交日期, 任务推进至 completed 或 failed。
  - 点击状态图节点: 在任意时刻点击图中某个状态框, 即可尝试将当前任务强制推向该状态; 合法转移会直接生效, 非法转移则会闪红并返回标准错误码。
- 应该看到: 默认停在最后一步 (任务已处于 completed 终态), 可观察以下现象与右侧读数:
  - 点击 working 或 canceled: 转移均被拒绝, 右侧「最近一次被拒的错误码」分别显示 -32004 或 -32002, 「当前状态」保持不变。
  - 「报销合计 / 单次上限」: 显示本次申报总额与 5000 元制度上限, 超限时标红。
  - 「轨迹长度」: 记录任务自创建以来经历的状态总数; demo 正常完成路径共包含 5 个状态 (submitted, working, input-required, working, completed), 底部列出完整状态序列。
  - 将总金额拖至超过 5000: 第 3 步终态变为 failed, 响应报文中的 artifacts 列表为空。
  - 切到 v1.0 写法: 同一份报文的方法名、状态值和 part 结构变了, 轨迹与错误码不变。"
    module="llm_agent/m18"
    run="python -m llm_agent.m18_a2a.demo"
    :challenge="{
      ask: '将时间轴回放至最后一步 (任务已进入 completed 状态), 随后分别点击 working 与 canceled 节点。请先预测: 服务端会分别返回什么 JSON-RPC 错误码? 当前任务状态是否会发生改变?',
      answer: '两次操作均会被拒绝, 错误码分别为 -32004 与 -32002, 且任务状态保持不变。\n- 点击 working: 请求被拒绝并返回错误码 -32004 (UnsupportedOperation)。该操作等价于携带同一 taskId 再次发送 message/send 试图继续推进已结束的任务, 而 A2A 协议规定终态任务不再接受后续消息输入。\n- 点击 canceled: 请求被拒绝并返回错误码 -32002 (TaskNotCancelable)。该操作等价于调用 tasks/cancel, 已执行完毕的任务不存在可取消的在途工作。\n- 状态与轨迹不变: completed 属于终态, 状态转移表中其出边集合为空。demo 中的 Task.to 在修改状态前会先查验合法转移表, 未命中时直接抛出 InvalidTransition 异常, 因此当前状态与历史记录均不会被篡改; demo 断言验证了连续三次非法请求后, 状态轨迹仍保持原有的 5 个状态。\n- 工程意义: 若需修改申报金额, 调用方必须新建独立任务。正因为终态不可逆转, 调用方才能将已完成的 artifact 作为确定性交付物安全消费, 无需担心结果在后台被静默改写。',
    }"
  >
    <template #controls>
      <LabSlider v-model="flight" label="机票" :min="0" :max="5000" :step="10" unit=" 元" />
      <LabSlider v-model="nights" label="住宿晚数" :min="1" :max="7" unit=" 晚" />
      <LabSlider v-model="rate" label="每晚" :min="100" :max="1000" :step="10" unit=" 元" />
      <div class="row">
        <button type="button" :class="{ active: personal }" :aria-pressed="personal" @click="personal = !personal">改成私人消费</button>
        <button v-for="w in ['v0.3', 'v1.0']" :key="w" type="button" :class="{ active: dialect === w }" @click="dialect = w">线上写法 {{ w }}{{ w === 'v0.3' ? ' (demo)' : '' }}</button>
      </div>
      <StepPlayer :stepper="stepper" />
    </template>

    <svg viewBox="0 0 440 235" class="sm" role="group" aria-label="A2A 任务状态机">
      <defs>
        <marker id="a2a-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah" /></marker>
      </defs>
      <line v-for="e in edges" :key="e.id" :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2" class="edge" :class="{ taken: e.taken }" marker-end="url(#a2a-arrow)" />
      <line v-if="flash" :x1="POS[flash.from].x" :y1="POS[flash.from].y" :x2="POS[flash.to].x" :y2="POS[flash.to].y" class="edge bad" />
      <g v-for="(p, s) in POS" :key="s" class="node focus-inset" :class="{ now: s === state, bad: flash && flash.to === s, term: !TRANSITIONS[s].length, seen: path.includes(s) }"
        tabindex="0" role="button" :aria-label="`试着把任务推到 ${s}`" @click="tryMove(s)" @keydown.enter="tryMove(s)">
        <rect :x="p.x - 50" :y="p.y - 15" width="100" height="30" rx="15" />
        <text :x="p.x" :y="p.y + 4" text-anchor="middle">{{ s }}</text>
      </g>
    </svg>

    <p class="note">{{ note }}</p>
    <pre class="mono wire">{{ frame.wire }}</pre>

    <template #stats>
      <div class="kv"><span>当前状态</span><b :class="state === 'completed' ? 'good' : ['failed', 'canceled', 'rejected'].includes(state) ? 'bad' : ''">{{ state || '(还没有任务)' }}</b></div>
      <div class="kv"><span>报销合计 / 单次上限 (元)</span><b :class="total > LIMIT ? 'bad' : 'good'">{{ total }} / {{ LIMIT }}</b></div>
      <div class="kv"><span>轨迹长度 (经过了几个状态)</span><b>{{ path.length }}</b></div>
      <div class="kv"><span>最近一次被拒的错误码</span><b :class="flash ? 'bad' : ''">{{ flash ? flash.code : '—' }}</b></div>
      <p class="lab-note">完整轨迹: {{ path.join(' → ') || '—' }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

// 与 demo 的 TRANSITIONS 相同 (教学版, 比规范原文更严格)
const TRANSITIONS = {
  submitted: ['working', 'failed', 'canceled', 'rejected'],
  working: ['input-required', 'auth-required', 'completed', 'failed', 'canceled', 'rejected'],
  'input-required': ['working', 'failed', 'canceled'],
  'auth-required': ['working', 'failed', 'canceled'],
  completed: [], failed: [], canceled: [], rejected: [],
}
const POS = { submitted: { x: 55, y: 120 }, working: { x: 190, y: 40 }, 'input-required': { x: 150, y: 205 }, 'auth-required': { x: 270, y: 205 }, completed: { x: 385, y: 25 }, failed: { x: 385, y: 85 }, canceled: { x: 385, y: 145 }, rejected: { x: 385, y: 205 } }
const LIMIT = 5000
const flight = ref(1280), nights = ref(3), rate = ref(450) // demo [2] 的默认请求; [4] 是 3200 + 5 × 900 = 7700
const total = computed(() => flight.value + nights.value * rate.value)
const personal = ref(false), dialect = ref('v0.3')
// v1.0: 枚举序列化成名字, 方法名改成驼峰, 去掉 kind 判别字段 (与 demo 的 v1_state / V1_METHODS 一致)
const st = (s) => dialect.value === 'v1.0' ? 'TASK_STATE_' + s.toUpperCase().replace('-', '_') : s
const M = (m) => dialect.value === 'v1.0' ? { 'message/send': 'SendMessage', 'tasks/cancel': 'CancelTask' }[m] : m
const part = (k, v) => dialect.value === 'v1.0' ? { [k]: v } : { kind: k, [k]: v }

// 线上报文按 demo 的 Network.wire 裁剪 (去掉 history); 金额和结局在前端现算
const frames = computed(() => {
  const ask = personal.value ? '报销私人聚餐: 餐费 600 元' : `报销上海出差: 机票 ${flight.value} 元, 住宿 ${nights.value} 晚每晚 ${rate.value} 元`
  const ok = total.value <= LIMIT
  const end = ok ? 'completed' : 'failed'
  const endMsg = ok ? `已提交报销单 EXP-task-1, 合计 ${total.value} 元。` : `合计 ${total.value} 元, 超过单次报销上限 ${LIMIT} 元, 请走特批流程。`
  const j = (o) => JSON.stringify(o, null, 1)
  const send = (id, text, taskId) => j({ jsonrpc: '2.0', id, method: M('message/send'), params: { message: { role: dialect.value === 'v1.0' ? 'ROLE_USER' : 'user', parts: [part('text', text)], ...(taskId ? { taskId } : {}) } } })
  const P1 = ['submitted', 'working', 'input-required']
  const card = { path: [], note: '发现: GET /.well-known/agent-card.json, 按 skill tag "报销" 挑中 expense-agent。相当于 MCP 的 tools/list, 但列的是技能 (skill), 没有函数签名。', wire: j({ name: 'expense-agent', url: 'a2a://expense', skills: [{ id: 'file_expense', tags: ['expense', 'reimbursement', '报销'] }] }) }
  const reply = (id, state, text, artifacts = []) => j({ id, result: { id: 'task-1', status: { state: st(state), message: { parts: [part('text', text)] } }, artifacts } })
  if (personal.value) return [card, { path: ['submitted', 'working', 'rejected'], note: '私人消费不在业务范围: 对方在动手之前决定不接, working 之后直接 rejected。它和 failed 一样是终态, 但没有追问、没有计算。', wire: `→ ${send(2, ask)}\n← ${reply(2, 'rejected', '私人消费不在差旅报销范围, 不受理。')}` }]
  return [
    card,
    { path: P1, note: '第一条请求没带 taskId: 新建 task-1。对方发现没有日期, 停在 input-required 反问。input-required 是中断态, 任务还活着。', wire: `→ ${send(2, ask)}\n← ${reply(2, 'input-required', '请提供出差日期 (YYYY-MM-DD)。')}` },
    { path: [...P1, 'working', end], note: ok ? `助理查日程答 2026-09-15, 带同一个 taskId 续上。对方内部用 calculator 算出 ${total.value}, 线上看不到这次调用; 交回 artifact。` : `合计 ${total.value} 超过上限 ${LIMIT}: failed 也是正常结局, 没有 artifact。`, wire: `→ ${send(4, '出差日期 2026-09-15', 'task-1')}\n← ${reply(4, end, endMsg, ok ? [{ parts: [part('data', { claim_id: 'EXP-task-1', date: '2026-09-15', total: total.value })] }] : [])}` },
  ]
})
const stepper = useStepper(() => frames.value.length, { interval: 1600 })
stepper.step.value = 2 // 默认停在任务结束: 读者可以直接试非法转移
const frame = computed(() => frames.value[Math.min(stepper.step.value, frames.value.length - 1)])
const manual = ref([]) // 读者手动点出来的转移, 接在当前帧的轨迹后面
const flash = ref(null)
watch(() => [stepper.step.value, total.value, personal.value], () => { manual.value = []; flash.value = null })

const path = computed(() => [...frame.value.path, ...manual.value])
const state = computed(() => path.value[path.value.length - 1] || '')

// ★ 与 Task.to 同一条规则: 不在出边表里就拒绝, 状态不动。线上再按方法映射成 JSON-RPC 错误码
function tryMove(to) {
  const from = state.value
  if (!from) { flash.value = null; return }
  if (TRANSITIONS[from].includes(to)) { manual.value = [...manual.value, to]; flash.value = null; return }
  const terminal = !TRANSITIONS[from].length
  const code = terminal && to === 'working' ? '-32004' : terminal && to === 'canceled' ? '-32002' : 'InvalidTransition'
  flash.value = { from, to, code }
}
const note = computed(() => {
  const f = flash.value
  if (f) return `拒绝 ${f.from} → ${f.to}。本地: InvalidTransition("${f.from} -> ${f.to}")${f.code.startsWith('-') ? `; 线上: ${M(f.to === 'working' ? 'message/send' : 'tasks/cancel')} → error ${f.code}` : ''}。状态保持 ${f.from}。`
  if (!state.value) return frame.value.note + ' 还没有任务, 点状态图不会有反应。'
  return manual.value.length ? `手动转移: ${manual.value.join(' → ')}。合法, 状态已更新。` : frame.value.note
})

const edges = computed(() => {
  const taken = new Set(path.value.slice(1).map((s, i) => `${path.value[i]}>${s}`))
  const out = []
  for (const [a, tos] of Object.entries(TRANSITIONS)) for (const b of tos) {
    const pa = POS[a], pb = POS[b]
    const off = TRANSITIONS[b].includes(a) ? (a === 'working' ? -9 : 9) : 0 // 回边与去边并排
    const dx = pb.x - pa.x, dy = pb.y - pa.y
    const cut = (sign) => { const t = Math.min(dx ? 50 / Math.abs(dx) : 1, dy ? 15 / Math.abs(dy) : 1); return { x: sign * dx * t, y: sign * dy * t } }
    const s = cut(1), e = cut(-1)
    out.push({ id: `${a}>${b}`, taken: taken.has(`${a}>${b}`), x1: pa.x + off + s.x, y1: pa.y + s.y, x2: pb.x + off + e.x, y2: pb.y + e.y })
  }
  return out
})
</script>

<style scoped>
.sm { width: 100%; max-width: 560px; }
.edge { stroke: var(--border-strong); stroke-width: 1.2; }
.edge.taken { stroke: var(--accent); stroke-width: 2; }
.edge.bad { stroke: var(--danger); stroke-width: 2; stroke-dasharray: 5 4; }
.ah { fill: var(--text-dim); }
.node { cursor: pointer; }
.node rect { fill: var(--bg-elev); stroke: var(--border-strong); }
.node.term rect { stroke-width: 2.5; }
.node.seen rect { stroke: var(--accent); }
.node.now rect { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 2.5; }
.node.bad rect { fill: color-mix(in srgb, var(--danger) 25%, transparent); stroke: var(--danger); }
.node:focus-visible rect { stroke: var(--warn); stroke-width: 3; }
.node text { font-size: 12px; fill: var(--text); font-family: "SF Mono", Menlo, monospace; pointer-events: none; }
.note { margin-top: 8px; font-size: 12px; color: var(--text-muted); line-height: 1.7; min-height: 40px; }
.wire { margin-top: 6px; font-size: 11px; line-height: 1.45; background: var(--code-bg); color: var(--code-text); border-radius: var(--radius-sm); padding: 8px 10px; overflow-x: auto; max-height: 260px; white-space: pre; }
</style>
