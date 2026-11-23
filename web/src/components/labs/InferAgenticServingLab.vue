<!-- Agent 多轮工具循环的前缀命中实验台 (对应 llm_infer/m41_agentic_serving): build_session + render + history_hits 的 JS 移植。 -->
<template>
  <LabFrame
    title="Agent 工具循环的前缀命中 — 历史里的思考块删不删, 决定缓存从哪里断开"
    sub="- 模拟什么: 一个 Agent 会话里用户发言 2 轮, 每轮模型调用若干次工具再给出回答; 每次调用先写一段思考, 再输出工具调用。每次请求的 prompt 是整段历史加上新的工具结果, 引擎的前缀缓存逐 token 比较本次 prompt 和之前算过的序列, 公共前缀直接复用。对话模板如果删掉之前的思考块, 断点之后的 token 都要重算。system prompt 假设已被别的会话缓存。
- 图上是什么: 上图每根柱子是一次模型调用, 柱高是 prompt 长度, 绿色是命中、红色是要算的部分。下面的长条是选中那次调用的 prompt, 按块着色 (system、用户消息、思考、工具调用、工具结果、回答), 竖线是缓存断开的位置。
- 可以做什么: 切换三种历史拼法; 拖动思考长度、工具结果长度和每轮工具调用数; 点击柱子 (或聚焦后按左右方向键) 选中一次调用。
- 应该看到: 全部保留时每次只算新的工具结果; 新用户消息到来时删思考, 第 2 轮第一次调用的柱子大半变红; 每次都删时每次多算上一次的可见输出, 但上下文短得多。右侧是要算的 token (比全部保留多时标红)、命中率、最后一次调用的上下文长度和思考占输出的比例。"
    module="llm_infer/m41"
    run="python -m llm_infer.m41_agentic_serving.demo"
    :challenge="{
      ask: '选「新用户消息到来时删思考」, 把每轮工具调用数从 5 拖到 10。第 2 轮第一次调用要重算的 token 怎么变? 为什么这比「每次都删」的代价更集中?',
      answer: '大约翻一倍。删掉的是第 1 轮所有调用的思考块, 第一个被删的思考块紧跟在第 1 轮用户消息后面, 缓存从那里断开; 之后第 1 轮的每次工具调用、每段工具结果和回答都要重算。工具调用数从 5 到 10, 这些可见内容也大约多一倍。\n「每次都删」时断点总在上一次调用的思考块处, 每次只多算上一次的可见输出 (几十个 token), 代价摊在每次调用上; 「新用户消息到来时删」在工具循环内部全部命中, 代价集中在新一轮的第一次调用, 工具结果越长、循环越长, 这一次越慢。全部保留没有这个断点, 代价是上下文更长, 每个会话占的 KV 更多。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(name, p) in POLICIES" :key="p" type="button" :class="{ active: policy === p }" @click="policy = p">{{ name }}</button>
      </div>
      <LabSlider v-model="think" label="每次思考" :min="0" :max="4000" :step="100" unit=" token" />
      <LabSlider v-model="result" label="每段工具结果" :min="100" :max="4000" :step="100" unit=" token" />
      <LabSlider v-model="steps" label="每轮工具调用数" :min="1" :max="10" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H + 74}`" role="group" aria-label="每次调用的命中与重算">
      <g
        class="bars" tabindex="0" role="slider" aria-label="选中的调用" :aria-valuemin="1" :aria-valuemax="sim.rows.length" :aria-valuenow="pickI + 1"
        @keydown.left.prevent="pick = Math.max(0, pickI - 1)" @keydown.right.prevent="pick = Math.min(sim.rows.length - 1, pickI + 1)"
      >
        <g v-for="(r, i) in sim.rows" :key="i" @mouseenter="pick = i" @click="pick = i">
          <rect :x="i * bw" y="0" :width="bw" :height="H" class="bg" />
          <rect :x="i * bw + 2" :y="H - r.prompt * ky" :width="bw - 4" :height="r.hit * ky" class="hit" />
          <rect :x="i * bw + 2" :y="H - (r.prompt - r.hit) * ky" :width="bw - 4" :height="(r.prompt - r.hit) * ky" class="miss" />
          <rect v-if="i === pickI" :x="i * bw" y="0" :width="bw" :height="H" class="pick" />
        </g>
      </g>
      <line :x1="(steps + 1) * bw" :x2="(steps + 1) * bw" y1="0" :y2="H" class="turn" />
      <text :x="(steps + 1) * bw + 4" y="12" class="ax">第 2 轮</text>
      <g :transform="`translate(0, ${H + 18})`">
        <rect v-for="(b, j) in strip" :key="j" :x="b.x0 * kx" y="0" :width="Math.max(0.5, b.n * kx)" height="18" :fill="COLORS[b.kind]" />
        <line :x1="sel.hit * kx" :x2="sel.hit * kx" y1="-4" y2="24" class="cut" />
        <text x="0" y="36" class="ax">调用 {{ pickI + 1 }} 的 prompt: {{ sel.prompt.toLocaleString() }} token, 命中 {{ sel.hit.toLocaleString() }}, 要算 {{ (sel.prompt - sel.hit).toLocaleString() }}</text>
      </g>
    </svg>
    <p class="legend"><span v-for="(c, k2) in COLORS" :key="k2"><i :style="{ background: c }" />{{ KIND_NAME[k2] }}</span></p>

    <template #stats>
      <div class="kv"><span>要算的 token (全部保留: {{ base.miss.toLocaleString() }})</span><b :class="sim.miss > base.miss ? 'bad' : 'good'">{{ sim.miss.toLocaleString() }}</b></div>
      <div class="kv"><span>命中率</span><b>{{ (sim.hitRate * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>最后一次调用的上下文 (KV)</span><b>{{ sim.ctx.toLocaleString() }}</b></div>
      <div class="kv"><span>思考占输出</span><b>{{ (sim.thinkFrac * 100).toFixed(1) }}%</b></div>
      <div class="lab-note">
        <p>默认参数与 Python demo [1] 相同: 三种拼法要算的 token 依次是 8,200 / 12,800 / 9,100, 最终上下文 25,800 / 18,600 / 12,600。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { range, sum } from '@/utils/labmath.js'

const W = 640, H = 150
const POLICIES = { keep_all: '全部保留', strip_on_user: '新用户消息到来时删思考', strip_all: '每次都删思考' }
const KIND_NAME = { system: 'system', user: '用户消息', think: '思考', call: '工具调用', result: '工具结果', answer: '回答' }
const COLORS = { system: 'var(--text-dim)', user: 'var(--accent)', think: 'var(--eye)', call: 'var(--left)', result: 'var(--warn)', answer: 'var(--right)' }
const SHAPE = { system: 2000, user: 100, call: 60, answer: 300, turns: 2 }
const policy = ref('strip_on_user'), think = ref(1200), result = ref(800), steps = ref(5), pick = ref(6)

function session(nThink, nResult, nSteps) {                                   // 同 build_session: 块 = {kind, turn, id, n}
  let id = 0
  const blk = (kind, turn, n) => ({ kind, turn, id: id++, n })
  const out = [blk('system', -1, SHAPE.system)], calls = []
  for (let t = 0; t < SHAPE.turns; t++) {
    out.push(blk('user', t, SHAPE.user))
    for (let s = 0; s <= nSteps; s++) {
      const last = s === nSteps
      const produced = [blk('think', t, nThink), blk(last ? 'answer' : 'call', t, last ? SHAPE.answer : SHAPE.call)]
      calls.push({ start: out.length, produced })
      out.push(...produced)
      if (!last) out.push(blk('result', t, nResult))
    }
  }
  return { blocks: out, calls }
}
const render = (blocks, p, turn) => blocks.filter((b) => !(b.kind === 'think' && (p === 'strip_all' || (p === 'strip_on_user' && b.turn < turn))))   // ★ 删掉的思考块
const lcp = (a, b) => { let n = 0; for (let i = 0; i < Math.min(a.length, b.length) && a[i].id === b[i].id; i++) n += a[i].n; return n }  // 块内 token 唯一, 按块比即可

function hits(p, nThink, nResult, nSteps) {
  const { blocks, calls } = session(nThink, nResult, nSteps)
  const cached = [[blocks[0]]], rows = []
  let out = 0, th = 0, ctx = 0
  for (const { start, produced } of calls) {
    const prompt = render(blocks.slice(0, start), p, blocks[start - 1].turn)
    const len = sum(prompt.map((b) => b.n))
    rows.push({ prompt: len, hit: Math.max(...cached.map((c) => lcp(prompt, c))), blocks: prompt })
    cached.push([...prompt, ...produced])                                      // 解码出的 token 也进缓存
    out += sum(produced.map((b) => b.n)); th += produced[0].n
    ctx = len + sum(produced.map((b) => b.n))
  }
  const P = sum(rows.map((r) => r.prompt)), Hh = sum(rows.map((r) => r.hit))
  return { rows, miss: P - Hh, hitRate: Hh / P, ctx, thinkFrac: out ? th / out : 0 }
}
const sim = computed(() => hits(policy.value, think.value, result.value, steps.value))
const base = computed(() => hits('keep_all', think.value, result.value, steps.value))
const bw = computed(() => W / sim.value.rows.length)
const ky = computed(() => H / Math.max(...sim.value.rows.map((r) => r.prompt)))
const pickI = computed(() => Math.min(pick.value, sim.value.rows.length - 1))
const sel = computed(() => sim.value.rows[pickI.value])
const kx = computed(() => W / Math.max(1, sel.value.prompt))
const strip = computed(() => { let x0 = 0; return sel.value.blocks.map((b) => { const r = { ...b, x0 }; x0 += b.n; return r }) })
</script>

<style scoped>
svg { min-width: 520px; }
.bars { outline: none; cursor: pointer; }
.bars:focus-visible .pick { stroke: var(--accent); stroke-width: 2.5; }
.bg { fill: transparent; }
.hit { fill: var(--left); opacity: 0.85; }
.miss { fill: var(--danger); opacity: 0.6; }
.pick { fill: none; stroke: var(--text); stroke-width: 1.5; pointer-events: none; }
.turn { stroke: var(--border-strong); stroke-dasharray: 3 3; }
.cut { stroke: var(--danger); stroke-width: 2; }
.ax { font-size: 11px; fill: var(--text-muted); }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
</style>
