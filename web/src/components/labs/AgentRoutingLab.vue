<!--
  模型路由与级联的成本模型 (对应 llm_agent/m61_model_routing/demo.py 的 tasks / router_noise / attempt / run / policies)。
  只讲一件事: 先过完成率门槛, 再比每个完成任务的成本; 每次请求的成本会把级联算得过于便宜。
  任务难度按分位数摆放、路由误差用黄金分割序列, 都不用随机数, 所以数字与 Python 逐项一致。
-->
<template>
  <LabFrame
    title="模型路由 — 小模型先试真的省钱吗, 按每个完成的任务算一遍"
    sub="- 模拟什么: 一批 200 个难易不一的 agent 任务, 用 7 种策略分配给模型。级联 (cascade) 指先让便宜的小模型试, 验收失败再交给大模型; 路由 (routing) 指先用一个分类器预测难度, 预测简单的走级联, 其余直接交给大模型。每次尝试是 6 轮的 agent 循环, 每轮重读共享前缀和已有历史, 缓存读按 0.1 倍、写按 1.25 倍计价, 缓存按模型隔离。价格比例取自 Claude 当前价目表 (大模型按 Opus 5.5, 小模型按 Haiku 5.5 或 Sonnet 5.5), 各模型的能力阈值是假设值。
- 图上是什么: 每行一种策略, 横条长度是每个完成任务的成本 (总花费含失败尝试, 除以完成的任务数), 行末是完成率:
  - 实色横条: 完成率达到 80% 门槛, 可以参与比价。
  - 浅色横条: 完成率不到门槛, 成本再低也不可比。
  - 竖虚线: 单模型 medium 的每任务成本, 作为参照。
- 可以做什么: 按界面顺序:
  - 「简单任务占比」滑杆: 改变任务集里简单任务 (难度低于 0.3) 的比例。
  - 「小模型价位」按钮: Haiku 5.5 价位约为大模型的 1/40, Sonnet 5.5 价位为 1/2。
  - 「路由预测噪声」滑杆: 分类器预测难度时的误差标准差 σ, 0 表示完美路由。
  - 点击任意一行: 下方显示该策略的每次请求成本、每任务成本、缓存命中率与平均轮数。
- 应该看到: 默认设置 (简单任务 30%, Haiku 价位, σ=0.15) 与 m61 demo [1] 的输出一致:
  - 点「级联 (交给大模型)」: 每次请求比单模型 medium 便宜, 每个完成任务却更贵, 右侧第三个读数大于 1。
  - 切到 Sonnet 价位并把简单任务拖到 10%: 右侧第二个读数变成大于 1, 级联输给单模型 medium。
  - 把 σ 拖到 0: 路由的平均轮数降到 6.0, 但 Haiku 价位下它的每任务成本仍高于从头换的级联。"
    module="llm_agent/m61"
    run="python -m llm_agent.m61_model_routing.demo"
    :challenge="{
      ask: '默认设置下, 「级联 (交给大模型)」和「级联 (从头换大模型)」完成率相同、调用次数也相同。先预测: 两者每个完成任务的成本谁高, 差多少?',
      answer: '把记录交给大模型接着做的那种更贵: 每个完成任务约 0.268 美元, 从头换的只要 0.169 美元。\n- 两者的小模型尝试完全一样, 区别只在升级后的那一段。\n- 缓存按模型隔离: 小模型 6 轮留下的约 17400 token 历史在大模型的缓存里没有, 大模型第一轮要按 1.25 倍写入价全部重读。\n- 之后大模型的每一轮都要再读这段历史, 上下文比从头开始长, 输入花费随之增加。\n- 从头换时大模型只读已缓存的共享前缀和自己的新历史, 花费与直接把任务交给它相同。\n所以级联省不省钱, 不只看小模型多便宜, 还要看升级时怎么交接上下文。',
    }"
  >
    <template #controls>
      <LabSlider v-model="easy" label="简单任务占比" :min="0" :max="90" :step="10" unit=" %" />
      <div class="row">
        <span class="tip">小模型价位:</span>
        <button v-for="s in SMALLS" :key="s.id" type="button" :class="{ active: small === s.id }" @click="small = s.id">{{ s.label }}</button>
      </div>
      <LabSlider v-model="sigma" label="路由预测噪声 σ" :min="0" :max="0.5" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <div class="bars">
      <button v-for="r in rows" :key="r.name" type="button" class="bar-row" :class="{ sel: sel === r.name, below: r.rate < FLOOR }" @click="sel = r.name">
        <span class="name">{{ r.name }}</span>
        <span class="track">
          <i class="fill" :style="{ width: Math.min(100, r.per_done / scale * 100) + '%' }" />
          <i class="ref" :style="{ left: Math.min(100, ref_.per_done / scale * 100) + '%' }" />
        </span>
        <span class="mono val">{{ r.per_done.toFixed(3) }}</span>
        <span class="mono rate">{{ (r.rate * 100).toFixed(1) }}%</span>
      </button>
    </div>
    <p class="detail mono">{{ sel }}: 每次请求 {{ cur.per_call.toFixed(4) }} 美元 · 每个完成任务 {{ cur.per_done.toFixed(3) }} 美元 · 缓存命中 {{ (cur.hit * 100).toFixed(1) }}% · 平均 {{ cur.turns.toFixed(1) }} 轮{{ cur.rate < FLOOR ? ' · 完成率不到门槛, 不参与比价' : '' }}</p>

    <template #stats>
      <div class="kv"><span>过门槛的最便宜策略</span><b>{{ best.name }} · {{ best.per_done.toFixed(3) }}</b></div>
      <div class="kv"><span>级联 (从头换) / 单模型 medium</span><b :class="ratioRestart < 1 ? 'good' : 'bad'">{{ ratioRestart.toFixed(2) }}</b></div>
      <div class="kv"><span>级联 (交给大模型) / 单模型 medium</span><b :class="ratioCarry < 1 ? 'good' : 'bad'">{{ ratioCarry.toFixed(2) }}</b></div>
      <div class="kv"><span>选中策略的平均轮数 (单模型为 6.0)</span><b :class="cur.turns <= TURNS ? 'good' : 'bad'">{{ cur.turns.toFixed(1) }}</b></div>
      <p class="lab-note">比值按每个完成任务的成本算, 小于 1 表示比单模型 medium 便宜。全部数字来自确定性成本模型 (能力阈值与每轮 token 数是假设值), 不是实测; 级联的验收器按完美处理, 这是对级联最有利的假设。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

// 与 m61 demo 的常量相同
const PRICE = { big: [4, 20], mid: [2, 10], small: [0.1, 0.5] } // 美元 / 1M token
const CAP = { 'big:high': 0.88, 'big:medium': 0.8, 'big:low': 0.7, 'mid:medium': 0.6, 'small:medium': 0.35 }
const OUT = { low: 400, medium: 900, high: 2000 }
const PREFIX = 6000, STEP = 2000, TURNS = 6, FLOOR = 0.8, N = 200
const SMALLS = [{ id: 'small', label: 'Haiku 5.5 (约 1/40)' }, { id: 'mid', label: 'Sonnet 5.5 (1/2)' }]

const easy = ref(30), small = ref('small'), sigma = ref(0.15), sel = ref('级联 (交给大模型)')

function tasks(share) {
  const k = Math.round(N * share)
  return [...Array(k)].map((_, i) => 0.3 * (i + 0.5) / k).concat([...Array(N - k)].map((_, j) => 0.3 + 0.7 * (j + 0.5) / (N - k)))
}
const noise = (i, s) => s * Math.sqrt(12) * (((i * 0.6180339887) % 1) - 0.5)

// ★ 一次 6 轮尝试: 第一轮里接过来的历史 carried 在本模型缓存里没有, 按写入价算
function attempt(model, effort, warm, carried) {
  const [pin, pout] = PRICE[model]
  let cost = 0, read = 0, written = 0, history = carried
  for (let k = 0; k < TURNS; k++) {
    const hit = k === 0 ? (warm ? PREFIX : 0) : PREFIX + history
    const write = k === 0 ? (warm ? 0 : PREFIX) + carried + STEP : STEP
    cost += (hit * 0.1 + write * 1.25) * pin / 1e6 + OUT[effort] * pout / 1e6
    read += hit; written += write; history += STEP + OUT[effort]
  }
  return { cost, read, written, history }
}

function run(plan, ts, carry) {
  let cost = 0, read = 0, written = 0, done = 0, calls = 0
  const warm = new Set() // 缓存按模型隔离
  ts.forEach((d, i) => {
    let carried = 0
    for (const [model, effort] of plan(i, d)) {
      const a = attempt(model, effort, warm.has(model), carried)
      warm.add(model); cost += a.cost; read += a.read; written += a.written; calls += TURNS
      if (d <= CAP[`${model}:${effort}`]) { done++; break }
      carried = carry ? a.history : 0
    }
  })
  return { rate: done / ts.length, per_call: cost / calls, per_done: cost / Math.max(done, 1), hit: read / (read + written), turns: calls / ts.length }
}

const rows = computed(() => {
  const ts = tasks(easy.value / 100), s = small.value
  const cascade = () => [[s, 'medium'], ['big', 'medium']]
  const P = [
    ['单模型 high', () => [['big', 'high']]], ['单模型 medium', () => [['big', 'medium']]], ['单模型 low', () => [['big', 'low']]],
    ['只用小模型', () => [[s, 'medium']]], ['级联 (从头换大模型)', cascade], ['级联 (交给大模型)', cascade],
    ['路由 (按预测难度)', (i, d) => (d + noise(i, sigma.value) < 0.3 ? cascade() : [['big', 'medium']])],
  ]
  return P.map(([name, plan]) => ({ name, ...run(plan, ts, name === '级联 (交给大模型)') }))
})
const pick = (name) => rows.value.find((r) => r.name === name)
const ref_ = computed(() => pick('单模型 medium'))
const cur = computed(() => pick(sel.value))
const scale = computed(() => Math.max(...rows.value.map((r) => r.per_done)))
const best = computed(() => rows.value.filter((r) => r.rate >= FLOOR).reduce((a, b) => (b.per_done < a.per_done ? b : a)))
const ratioRestart = computed(() => pick('级联 (从头换大模型)').per_done / ref_.value.per_done)
const ratioCarry = computed(() => pick('级联 (交给大模型)').per_done / ref_.value.per_done)
</script>

<style scoped>
.bars { display: flex; flex-direction: column; gap: 4px; }
.bar-row { display: grid; grid-template-columns: minmax(96px, 150px) 1fr 48px 48px; align-items: center; gap: 8px; padding: 3px 6px; border: 1px solid transparent; border-radius: var(--radius-sm); background: none; color: var(--text); font-size: 12px; text-align: left; min-height: 0; }
.bar-row.sel { border-color: var(--accent); background: var(--accent-soft); }
.name { overflow-wrap: anywhere; }
.track { position: relative; height: 14px; background: var(--bg); border-radius: 2px; }
.fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); border-radius: 2px; }
.below .fill { background: var(--border-strong); }
.below .name, .below .rate { color: var(--text-dim); }
.ref { position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed var(--warn); }
.val, .rate { font-size: 11px; text-align: right; }
.detail { margin-top: 8px; font-size: 11px; color: var(--text-muted); line-height: 1.6; }
.tip { font-size: 12px; color: var(--text-muted); }
</style>
