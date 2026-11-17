<!--
  A2A 任务状态机 (对应 llm_agent/m18_a2a/demo.py 的 TRANSITIONS / Task.to / ExpenseAgent / TripAssistant.delegate)。
  只讲一件事: A2A 的交互单位是一个有状态的 Task; input-required 能转回 working (回边), 终态没有出边。
  有回边, 所以不用 DagView, 手写一张 6 个节点的小状态图。
-->
<template>
  <LabFrame
    title="A2A — 一个任务怎么在两个 agent 之间来回, 结束后为什么不能再改"
    sub="- 模拟什么: 两个 agent 协作。出差助理 (调用方) 把「报销」这件事交给报销 agent (服务方)。A2A (Agent2Agent) 协议规定它们之间的交互单位不是一次函数调用, 而是一个有状态的任务 Task: 任务有 id、有状态, 可以多轮来回 (对方发现缺信息会反问), 最终产物放在 artifact 里。这个台子把 demo 里的一次报销任务回放一遍, 并让你自己去推状态机。
      - 图上是什么: 一张状态图, 下面是说明和报文:
        - 六个圆角框: Task 可能处于的六个状态, submitted (已提交)、working (对方正在处理)、input-required (对方缺信息, 停下来反问)、completed (成功)、failed (失败)、canceled (取消)。
        - 箭头: 框之间的箭头是协议允许的转移方向; 右边三个边框加粗的是终态, 终态没有向外的箭头。
        - 高亮: 高亮的框是任务现在所处的状态, 高亮的箭头是这次任务已经走过的路。
        - 灰字: 图下面的灰字说明当前这一步发生了什么。
        - 代码块: 再下面的代码块是线上实际收发的 JSON-RPC 报文, → 是助理发出的请求, ← 是报销 agent 的回复。
      - 可以做什么: 三种操作:
        - 「机票」「住宿晚数」「每晚」三个滑杆: 改报销金额, 合计 = 机票 + 晚数 × 每晚; 合计超过 5000 元的上限, 结局会从 completed 变成 failed。
        - 播放器: 用「上一步 / 下一步」或时间轴回放三步。第 1 步助理读对方的 Agent Card, 按 skill 标签挑中报销 agent; 第 2 步发第一条 message/send, 对方发现没有出差日期, 停在 input-required 反问; 第 3 步助理带着同一个 taskId 补上日期, 任务走到 completed 或 failed。
        - 点状态图上的一个框: 任何时候点, 就是试着把任务推到那个状态。合法的转移会走过去, 非法的会闪红并给出错误码。
      - 应该看到: 默认停在最后一步, 任务已经 completed。现象和读数:
        - 点 working 或 canceled: 都会被拒绝, 右侧「最近一次被拒的错误码」显示 -32004 或 -32002, 「当前状态」不变。
        - 「报销合计 / 单次上限」: 本次金额和 5000 元上限, 超了就标红。
        - 「轨迹长度」: 任务到现在经过了几个状态, demo 的正常结局是 5 个 (submitted, working, input-required, working, completed), 最下面一行是完整轨迹。
        - 把金额拖到超过 5000: 第 3 步的结局变成 failed, 报文里 artifacts 是空的。"
    module="llm_agent/m18"
    run="python -m llm_agent.m18_a2a.demo"
    :challenge="{
      ask: '回放到最后一步 (任务已 completed), 再点 working 或 canceled。先猜: 线上各回什么错误码? 任务状态会变吗?',
      answer: '- 点 working: 被拒, 错误码 -32004 (UnsupportedOperation)。点 working 相当于再发一条带同一个 taskId 的 message/send, 想让已经结束的任务继续干活; 协议规定终态任务不支持这个操作。\n- 点 canceled: 被拒, 错误码 -32002 (TaskNotCancelable)。它相当于发 tasks/cancel; 一个已经完成的任务没有东西可取消。\n- 状态不变: completed 是终态, 转移表里它的出边集合是空的。demo 里 Task.to 在改状态之前先查转移表, 查不到就抛 InvalidTransition, 所以状态和历史都没被碰过。demo 的断言验证了这一点: 连拒三次之后, 轨迹仍然是原来的 5 个状态。\n- 要改金额只能开一个新任务。终态不能「复活」, 所以调用方才敢把拿到的 artifact 当成最终结果, 不用担心它在自己不知道的时候被改掉。',
    }"
  >
    <template #controls>
      <LabSlider v-model="flight" label="机票" :min="0" :max="5000" :step="10" unit=" 元" />
      <LabSlider v-model="nights" label="住宿晚数" :min="1" :max="7" unit=" 晚" />
      <LabSlider v-model="rate" label="每晚" :min="100" :max="1000" :step="10" unit=" 元" />
      <StepPlayer :stepper="stepper" />
    </template>

    <svg viewBox="0 0 440 200" class="sm" role="group" aria-label="A2A 任务状态机">
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
      <div class="kv"><span>当前状态</span><b :class="state === 'completed' ? 'good' : state === 'failed' || state === 'canceled' ? 'bad' : ''">{{ state || '(还没有任务)' }}</b></div>
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
  submitted: ['working', 'failed', 'canceled'],
  working: ['input-required', 'completed', 'failed', 'canceled'],
  'input-required': ['working', 'failed', 'canceled'],
  completed: [], failed: [], canceled: [],
}
const POS = { submitted: { x: 55, y: 110 }, working: { x: 185, y: 60 }, 'input-required': { x: 185, y: 160 }, completed: { x: 380, y: 25 }, failed: { x: 380, y: 100 }, canceled: { x: 380, y: 175 } }
const LIMIT = 5000
const flight = ref(1280), nights = ref(3), rate = ref(450) // demo [2] 的默认请求; [4] 是 3200 + 5 × 900 = 7700
const total = computed(() => flight.value + nights.value * rate.value)

// 线上报文按 demo 的 Network.wire 裁剪 (去掉 history); 金额和结局在前端现算
const frames = computed(() => {
  const ask = `报销上海出差: 机票 ${flight.value} 元, 住宿 ${nights.value} 晚每晚 ${rate.value} 元`
  const ok = total.value <= LIMIT
  const end = ok ? 'completed' : 'failed'
  const endMsg = ok ? `已提交报销单 EXP-task-1, 合计 ${total.value} 元。` : `合计 ${total.value} 元, 超过单次报销上限 ${LIMIT} 元, 请走特批流程。`
  const j = (o) => JSON.stringify(o, null, 1)
  const send = (id, text, taskId) => j({ jsonrpc: '2.0', id, method: 'message/send', params: { message: { role: 'user', parts: [{ kind: 'text', text }], ...(taskId ? { taskId } : {}) } } })
  const P1 = ['submitted', 'working', 'input-required']
  return [
    { path: [], note: '发现: GET /.well-known/agent-card.json, 按 skill tag "报销" 挑中 expense-agent。相当于 MCP 的 tools/list, 但列的是技能 (skill), 没有函数签名。', wire: j({ name: 'expense-agent', url: 'a2a://expense', skills: [{ id: 'file_expense', tags: ['expense', 'reimbursement', '报销'] }] }) },
    { path: P1, note: '第一条 message/send 没带 taskId: 新建 task-1。对方发现没有日期, 停在 input-required 反问。input-required 是正常中间态, 任务还活着。', wire: `→ ${send(2, ask)}\n← ${j({ id: 2, result: { id: 'task-1', status: { state: 'input-required', message: { parts: [{ text: '请提供出差日期 (YYYY-MM-DD)。' }] } }, artifacts: [] } })}` },
    { path: [...P1, 'working', end], note: ok ? `助理查日程答 2026-09-15, 带同一个 taskId 续上。对方内部用 calculator 算出 ${total.value}, 线上看不到这次调用; 交回 artifact。` : `合计 ${total.value} 超过上限 ${LIMIT}: failed 也是正常结局, 没有 artifact。`, wire: `→ ${send(4, '出差日期 2026-09-15', 'task-1')}\n← ${j({ id: 4, result: { id: 'task-1', status: { state: end, message: { parts: [{ text: endMsg }] } }, artifacts: ok ? [{ parts: [{ kind: 'data', data: { claim_id: 'EXP-task-1', date: '2026-09-15', total: total.value } }] }] : [] } })}` },
  ]
})
const stepper = useStepper(() => frames.value.length, { interval: 1600 })
stepper.step.value = 2 // 默认停在任务结束: 读者可以直接试非法转移
const frame = computed(() => frames.value[stepper.step.value])
const manual = ref([]) // 读者手动点出来的转移, 接在当前帧的轨迹后面
const flash = ref(null)
watch(() => [stepper.step.value, total.value], () => { manual.value = []; flash.value = null })

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
  if (f) return `拒绝 ${f.from} → ${f.to}。本地: InvalidTransition("${f.from} -> ${f.to}")${f.code.startsWith('-') ? `; 线上: ${f.to === 'working' ? 'message/send' : 'tasks/cancel'} → error ${f.code}` : ''}。状态保持 ${f.from}。`
  if (!state.value) return frame.value.note + ' 还没有任务, 点状态图不会有反应。'
  return manual.value.length ? `手动转移: ${manual.value.join(' → ')}。合法, 状态已更新。` : frame.value.note
})

const edges = computed(() => {
  const taken = new Set(path.value.slice(1).map((s, i) => `${path.value[i]}>${s}`))
  const out = []
  for (const [a, tos] of Object.entries(TRANSITIONS)) for (const b of tos) {
    const pa = POS[a], pb = POS[b]
    const off = (a === 'working' && b === 'input-required') ? -12 : (a === 'input-required' && b === 'working') ? 12 : 0 // 回边与去边并排
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
