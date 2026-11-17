<!--
  投机解码实验台 (对应 llm_infer/m07_speculative_decoding/speculative.py)。
  一件事: 每次 target 调用的期望产出 (1−α^(K+1))/(1−α) —— K 再大也被 1/(1−α) 封顶, draft 不是免费的。
-->
<template>
  <LabFrame
    title="投机解码 — 小模型连猜 K 个, 大模型一次验完, 看每次调用能多出几个 token"
    sub="- 模拟什么: decode 每步只出 1 个 token, 却要把整份权重读一遍, GPU 算力大量闲着。投机解码让一个便宜的小模型 (draft) 先连猜 $K$ 个 token, 原来的大模型 (target) 一次 forward 把这 $K$ 个一起检查。输出和 target 自己一个一个生成完全一样, 只是每次 target 调用能多出几个 token。$\alpha$ 是每个 draft token 被 target 接受的概率 (接受率), 由 draft 和 target 有多像决定。检查的规则:
        - 从头开始连续猜对的都收下。
        - 第一个猜错的位置换成 target 自己算出的 token, 后面的作废。
        - $K$ 个全对再白送 1 个 (target 那次 forward 顺手算出了第 $K+1$ 个)。
      - 图上是什么: 上图横轴是 draft 长度 $K$, 曲线上的大圆点是当前 $K$; 下面还有一排回放格子:
        - 紫色实线: 每次 target 调用的期望产出 $E = (1 - \alpha^{K+1}) / (1 - \alpha)$ 个 token。
        - 绿色虚线: 加速比 $E / (1 + K \cdot c)$, 其中 $c$ 是 draft 一步的耗时相对 target 一步的比例, 实心点是加速比最高的 $K$。
        - 黄色虚线: $E$ 的上限 $1 / (1 - \alpha)$, $K$ 再大也到不了它。
        - 底部的点线: 1, 即普通 decode 每次调用 1 个 token。
        - 下面一排格子: 逐轮回放。d1 到 dK 是 draft 猜的 token; 验证后打 ✓ 的被接受, 「✗→t」是第一个被拒、换成 target 纠错 token 的位置, 「—」是作废的; 最后一格是第 $K+1$ 个槽位, 全部接受时显示 bonus。
      - 可以做什么: 三个滑杆, 然后是回放:
        - 拖「draft 长度 K」或直接拖曲线上的圆点。
        - 拖「每 token 接受率 α」看曲线和上限怎么变。
        - 拖「draft 成本 c」看加速比的最优 $K$ 怎么移。
        - 「换一组随机数」重抽回放里每轮接受几个。
        - 「下一步」和「时间轴」逐轮看 draft 连猜、target 验证、接受或纠错三个阶段。
      - 应该看到: $\alpha$ 低 (0.3 以下) 时曲线很快就平了, 拖大 $K$ 没用; $c$ 不为 0 时, $K$ 过大加速比反而下降。右侧「期望产出: 公式 / 模拟」是公式值和按 $\alpha$ 独立抛硬币模拟 20000 轮的均值, 两者接近 (差 0.05 以内标绿) 说明公式对。其余读数:
        - 「加速比」小于 1 标红, 意思是 draft 的开销超过了省下的 target 调用。
        - 「省掉的 target 调用」是 $1 - 1/E$。
        - 「回放至今」是回放到当前轮为止累计产出的 token 数和 target 调用数。"
    module="llm_infer/m07"
    run="python -m llm_infer.m07_speculative_decoding.demo"
    :challenge="{
      ask: 'α = 0.6 时把 K 从 4 拖到 10, draft 长了一倍多。每轮的期望产出涨了多少?',
      answer: '涨得很少: 2.31 变成 2.49, 只多了 0.18 个。\n- $\\alpha = 0.6$ 时 $E = (1 - 0.6^{K+1}) / (1 - 0.6)$: $K = 4$ 是 2.31, $K = 10$ 是 2.49, 上限 $1/(1-\\alpha) = 2.5$。接受是链式的: 第一个猜错, 后面全废。第 $i$ 个 draft token 能被用上的概率是 $\\alpha^i$, $0.6^5 \\approx 0.08$, 所以第 5 个以后的槽位几乎从不被用到, 长 draft 的尾巴基本是白猜。\n- draft 不是免费的: 把「draft 成本 c」拖到 0.2, 一轮耗时是 1 次 target 加 $K$ 次 draft, 即 $1 + 0.2K$ 倍 target 时间。加速比 $E / (1 + K \\cdot c)$ 在 $K = 2$ 就见顶 (1.40×), 再长反而更慢。\n反过来, draft 再烂 ($\\alpha \\approx 0.1$) 也不会改变输出, 只会更慢:\n- greedy 模式: 只接受和 target argmax 相同的 token, 输出和 target 自己 greedy 完全一样。\n- 采样模式: 以 $\\min(1, p/q)$ 的概率接受 ($p$ 是 target 给这个 token 的概率, $q$ 是 draft 给的), 拒绝时从归一化的残差分布 $\\max(0, p - q)$ 重采样。两步合起来, 每个位置的输出分布恰好等于 target 的分布 $p$。\n所以挑 draft 只看速度账: 接受率够高、单步够便宜。',
    }"
  >
    <template #controls>
      <LabSlider v-model="K" label="draft 长度 K" :min="1" :max="10" />
      <LabSlider v-model="alpha" label="每 token 接受率 α" :min="0" :max="0.95" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="cost" label="draft 成本 c (÷target)" :min="0" :max="0.5" :step="0.01" :format="(t) => t.toFixed(2)" />
      <div class="row">
        <button type="button" @click="seed++">换一组随机数</button>
        <StepPlayer :stepper="stepper" :label="`第 ${cur.round + 1} 轮 · ${PHASES[cur.phase]}`" />
      </div>
    </template>

    <svg ref="svg" viewBox="0 0 520 210" role="group" aria-label="每次 target 调用的期望产出与加速比随 draft 长度 K 的变化">
      <line x1="40" x2="510" y1="180" y2="180" class="axis" /><line x1="40" x2="40" y1="10" y2="180" class="axis" />
      <text v-for="k in 10" :key="k" :x="sx(k)" y="196" class="tick">{{ k }}</text>
      <text v-for="v in yTicks" :key="`y${v}`" x="34" :y="sy(v) + 3" class="tick end">{{ v }}</text>
      <text x="505" y="206" class="tick end">draft 长度 K</text>
      <line x1="40" x2="510" :y1="sy(1)" :y2="sy(1)" class="base" />
      <line v-if="cap <= yMax" x1="40" x2="510" :y1="sy(cap)" :y2="sy(cap)" class="cap" />
      <text x="508" :y="(cap <= yMax ? sy(cap) : 10) + 11" class="tick end warn">上限 1/(1−α) = {{ cap.toFixed(2) }}{{ cap > yMax ? ' (图外)' : '' }}</text>
      <polyline :points="pts(E)" class="curve e" />
      <polyline :points="pts(speed)" class="curve s" />
      <circle v-for="k in 10" :key="`d${k}`" :cx="sx(k)" :cy="sy(speed(k))" r="2.5" :class="['dot', { best: k === bestK }]" />
      <circle
        :cx="sx(K)" :cy="sy(E(K))" r="8" class="handle draggable" tabindex="0" role="slider" aria-label="当前 K 的位置: 左右拖动改变 draft 长度 K"
        :aria-valuenow="K" aria-valuemin="1" aria-valuemax="10"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.left.prevent="K = clamp(K - 1, 1, 10)" @keydown.right.prevent="K = clamp(K + 1, 1, 10)"
      />
    </svg>
    <div class="legend">
      <span><i class="e" />每次 target 调用的期望产出 (token 数)</span><span><i class="s" />加速比 <Tex text="$E / (1 + K \cdot c)$" /> (算上 draft 的开销), 实心点 = 加速比最高的 <Tex text="$K$" /></span>
    </div>

    <div class="round" aria-live="polite">
      <span class="lbl mono">draft</span>
      <span v-for="(c, i) in slots" :key="i" class="cell" :class="c.cls" :title="c.tip">{{ c.text }}</span>
      <span class="msg">{{ msg }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>每次 target 调用的期望产出: 公式 / 模拟 {{ ROUNDS }} 轮的均值</span><b :class="Math.abs(sim.mean - E(K)) < 0.05 ? 'good' : 'bad'">{{ E(K).toFixed(2) }} / {{ sim.mean.toFixed(2) }}</b></div>
      <div class="kv"><span>加速比 <Tex text="$E/(1+K \cdot c)$" /> (相对普通 decode, 算上 draft 开销)</span><b :class="speed(K) > 1 ? 'good' : 'bad'">{{ speed(K).toFixed(2) }}×</b></div>
      <div class="kv"><span>省掉的 target 调用 (<Tex text="$1 - 1/E$" />)</span><b>{{ ((1 - 1 / E(K)) * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>回放至今: 产出的 token / target 调用次数</span><b>{{ soFar.tokens }} / {{ soFar.calls }}</b></div>
      <div class="lab-note">
        <p>每轮产出 = 接受数 <Tex text="$n + 1$" />。那个「+1」永远有: 被拒时是 target 自己给的纠错 token, 全接受时是 bonus。所以最差也和普通 decode 一样每次调用出 1 个 token, 只是白付了 <Tex text="$K$" /> 次 draft 的钱 (加速比 &lt; 1)。</p>
        <p>公式的来历: 第 <Tex text="$i$" /> 个 draft token 被接受, 要求前 <Tex text="$i$" /> 个全对, 概率 <Tex text="$\alpha^i$" />; 把 <Tex text="$i = 0$" /> 到 <Tex text="$K$" /> 的 <Tex text="$\alpha^i$" /> 加起来就是 <Tex text="$E$" />。当前参数下加速比最高的 <Tex text="$K$" /> = {{ bestK }}。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { useDrag } from '@/composables/useDrag.js'
import { argmax, clamp, mulberry32, range } from '@/utils/labmath.js'

const ROUNDS = 20000, SHOWN = 12, PHASES = ['draft 连猜', 'target 一次验证', '接受 / 纠错']
const K = ref(4), alpha = ref(0.7), cost = ref(0.1), seed = ref(1)
const svg = ref(null)
const { start } = useDrag()

// ★ 第一个拒绝之前的都接受: P(n ≥ i) = α^i → E[n+1] = Σ_{i=0..K} α^i = (1−α^(K+1))/(1−α)
const E = (k) => (alpha.value === 0 ? 1 : (1 - alpha.value ** (k + 1)) / (1 - alpha.value))
const speed = (k) => E(k) / (1 + k * cost.value)     // 一轮耗时 = 1 次 target + K 次 draft
const cap = computed(() => 1 / (1 - alpha.value))
const bestK = computed(() => argmax(range(10).map((i) => speed(i + 1))) + 1)

// 模拟: 每个槽位独立以 α 接受, 遇到第一个拒绝就停 (与 accept_greedy / accept_sampling 的循环同构)
const sim = computed(() => {
  const rand = mulberry32(seed.value * 7919 + 13)
  const accepts = []
  let total = 0
  for (let r = 0; r < ROUNDS; r++) {
    let n = 0
    while (n < K.value && rand() < alpha.value) n++
    total += n + 1
    if (r < SHOWN) accepts.push(n)
  }
  return { mean: total / ROUNDS, accepts }
})

const yMax = computed(() => Math.max(2, E(10) * 1.2))
const yTicks = computed(() => { const s = yMax.value > 6 ? 2 : 1; return range(Math.floor(yMax.value / s) + 1).map((i) => i * s) })
const sx = (k) => 40 + (k - 1) * (470 / 9)
const sy = (v) => 180 - (clamp(v, 0, yMax.value) / yMax.value) * 170
const pts = (f) => range(10).map((i) => `${sx(i + 1)},${sy(f(i + 1))}`).join(' ')
const onMove = ({ x }) => { K.value = clamp(Math.round((x - 40) / (470 / 9)) + 1, 1, 10) }

const stepper = useStepper(() => SHOWN * 3, { interval: 700 })
const cur = computed(() => ({ round: Math.floor(stepper.step.value / 3), phase: stepper.step.value % 3 }))
watch([K, alpha, seed], () => stepper.reset())

const slots = computed(() => {
  const n = sim.value.accepts[cur.value.round], ph = cur.value.phase, all = n === K.value
  const cells = range(K.value).map((i) => {
    if (ph === 0) return { text: `d${i + 1}`, cls: 'on', tip: 'draft 猜的 token' }
    if (ph === 1) return { text: `d${i + 1}`, cls: 'hot', tip: 'target 同一次 forward 里验证' }
    if (i < n) return { text: '✓', cls: 'ok', tip: '与 target 一致, 接受' }
    if (i === n) return { text: '✗→t', cls: 'bad', tip: '第一个拒绝: 换成 target 的纠错 token' }
    return { text: '—', cls: 'dim', tip: '前面已拒绝, 作废 (KV 被 truncate 回滚)' }
  })
  cells.push(ph === 0 ? { text: '', cls: 'dim', tip: '第 K+1 个槽位' }
    : ph === 1 ? { text: '+1', cls: 'hot', tip: '第 K+1 个槽位也一起算' }
      : { text: all ? 'bonus' : '—', cls: all ? 'ok' : 'dim', tip: all ? '全部接受: 再白送 1 个' : '用不上' })
  return cells
})
const msg = computed(() => {
  const n = sim.value.accepts[cur.value.round]
  if (cur.value.phase === 0) return `draft 自回归 ${K.value} 步 (成本 ${K.value}×c)`
  if (cur.value.phase === 1) return `target 1 次 forward 验 ${K.value + 1} 个槽位`
  return `接受 ${n} 个 + ${n === K.value ? 'bonus' : '纠错'} 1 个 = 本轮 ${n + 1} 个 token`
})
const soFar = computed(() => {
  const done = cur.value.round + (cur.value.phase === 2 ? 1 : 0)
  return { calls: done, tokens: sim.value.accepts.slice(0, done).reduce((a, n) => a + n + 1, 0) }
})
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 460px; touch-action: pan-x pan-y; }
.axis { stroke: var(--border-strong); stroke-width: 1; }
.base { stroke: var(--text-dim); stroke-dasharray: 2 4; }
.cap { stroke: var(--warn); stroke-dasharray: 5 4; }
.tick { font-size: 10px; fill: var(--text-dim); text-anchor: middle; }
.tick.end { text-anchor: end; }
.tick.warn { fill: var(--warn); }
.curve { fill: none; stroke-width: 2; }
.curve.e { stroke: var(--accent); }
.curve.s { stroke: var(--left); stroke-dasharray: 4 3; }
.dot { fill: var(--bg-elev); stroke: var(--left); }
.dot.best { fill: var(--left); }
.handle { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--text-muted); margin: 6px 0 12px; }
.legend i { display: inline-block; width: 14px; height: 3px; margin-right: 5px; vertical-align: middle; }
.legend i.e { background: var(--accent); } .legend i.s { background: var(--left); }
.round { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.round .cell { min-width: 40px; height: 26px; }
.round .lbl { font-size: 11px; color: var(--text-dim); margin-right: 4px; }
.round .msg { font-size: 12px; color: var(--text-muted); margin-left: 8px; }
</style>
