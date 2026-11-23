<!--
  投机解码实验台 (对应 llm_infer/m07_speculative_decoding/speculative.py)。
  一件事: 每次 target 调用的期望产出 (1−α^(K+1))/(1−α) —— K 再大也被 1/(1−α) 封顶, draft 不是免费的。
-->
<template>
  <LabFrame
    title="投机解码 — 小模型连猜 K 个, 大模型一次验完, 看每次调用能多出几个 token"
    sub="- 模拟什么: 标准自回归 decode 每步仅产出 1 个 token 却必须遍历读取整份大模型权重, 导致 GPU 算力因访存带宽受限而大量闲置。投机解码 (speculative decoding) 引入一个轻量级草稿模型 (draft) 先连续自回归猜测 $K$ 个候选 token, 再交由原目标大模型 (target) 用单次并行 forward 一次性验证全部 $K$ 个位置。该机制严格保证最终输出分布与 target 独立逐步生成完全一致, 同时令单次 target 前向调用平均产出多个 token。设单步候选被 target 接受的概率为 $\alpha$ (取决于 draft 与 target 分布的接近程度), 单轮并行验证规则如下:
  - 从起始位置起连续通过验证的草稿 token 全部保留。
  - 遇到首个被拒绝的位置时, 直接替换为 target 在该位置自身算出的修正 token, 其后所有剩余草稿候选全部丢弃。
  - 若全部 $K$ 个草稿候选均被接受, 则额外获得第 $K+1$ 个 bonus token (由 target 同一次并行 forward 在最后一个草稿位置直接输出)。
- 图上是什么: 上方为随投机步长 $K$ 变化的理论收益曲线 (大圆点标示当前选定的 $K$), 下方为单轮验证过程的槽位回放:
  - 紫色实线: 单次 target 调用的期望产出 token 数 $E = (1 - \alpha^{K+1}) / (1 - \alpha)$。
  - 绿色虚线: 端到端墙钟加速比 $E / (1 + K \cdot c)$, 其中 $c$ 为单步 draft 耗时相对于单步 target 耗时的比例, 绿色实心圆点标示加速比最高的最优步长 $K$。
  - 黄色虚线: 期望产出渐近上限 $1 / (1 - \alpha)$, 无论 $K$ 取多大均无法超越该水平线。
  - 底部基准点线: 纵坐标 1, 代表标准自回归 decode 每次调用固定产出 1 个 token。
  - 下方回放槽位: d1 至 dK 代表草稿模型依次猜测的候选; 验证后标 ✓ 的槽位代表接受, 标「✗→t」的槽位代表首个被拒并由 target 直接替换纠错的位置, 标「—」的槽位代表连带作废的后续猜测; 末尾第 $K+1$ 格在前面全部接受时亮起并产出 bonus token。
- 可以做什么: 调节三个核心参数滑杆并逐步回放验证状态机:
  - 拖动「draft 长度 K」或直接拖动曲线上的圆点修改单轮猜测步数。
  - 拖动「每 token 接受率 α」观察期望产出曲线与渐近上限的变化。
  - 拖动「draft 成本 c」观察考虑草稿开销后最优步长 $K$ 的左移规律。
  - 点击「换一组随机数」重新采样回放序列中各轮的接受结果。
  - 使用「下一步」与「时间轴」分步观察草稿连猜、target 并行验证、截断接受或纠错三个阶段。
- 应该看到: 当接受率 $\alpha$ 较低 (0.3 以下) 时期望产出曲线迅速饱和走平, 盲目拉长 $K$ 几乎无额外收益; 当草稿成本 $c$ 不为 0 时, $K$ 过大会导致加速比越过峰值后显著回落。右侧「期望产出: 公式 / 模拟」并列对比解析公式值与按接受率 $\alpha$ 独立随机模拟 20000 轮统计出的样本均值, 两者差值在 0.05 以内时标绿验证公式吻合。其余读数含义如下:
  - 「加速比」低于 1 时标红, 表示运行草稿模型的累计耗时已超出节省下来的 target 调用收益。
  - 「省掉的 target 调用」按 $1 - 1/E$ 核算相对标准自回归减少的大模型前向次数占比。
  - 「回放至今」统计演示回放推进至当前轮次时的累计产出 token 总数与累计 target 调用次数。"
    module="llm_infer/m07"
    run="python -m llm_infer.m07_speculative_decoding.demo"
    :challenge="{
      ask: '当单步接受率 $\\alpha = 0.6$ 时, 将草稿长度 $K$ 从 4 拖至 10 (猜测步数增加一倍以上), 单轮期望产出 token 数增加了多少?',
      answer: '增幅极小: 单轮期望产出仅从 2.31 升至 2.49, 净增仅 0.18 个 token。\n- 链式截断的指数衰减效应: 当 $\\alpha = 0.6$ 时, 代入 $E = (1 - 0.6^{K+1}) / (1 - 0.6)$ 可得 $K = 4$ 时为 2.31、$K = 10$ 时为 2.49, 已逼近无穷步长下的理论渐近上限 $1/(1-\\alpha) = 2.5$。由于链式验证一旦遇到首个错误位置后续猜测即全部作废, 第 $i$ 个草稿位置能够存活生效的概率呈几何级数衰减为 $\\alpha^i$; 其中 $0.6^5 \\approx 0.08$, 意味着第 5 步之后的草稿槽位绝大部分时间都会被截断丢弃, 徒增无效猜测。\n- 草稿模型自身的串行耗时: 将「draft 成本 c」拖至 0.2 时, 每轮总耗时等于 1 次 target 前向加上 $K$ 次串行 draft 前向, 合 $1 + 0.2K$ 倍单步 target 时间。此时综合加速比 $E / (1 + K \\cdot c)$ 在 $K = 2$ 处即达到峰值 (1.40×), 继续拉长 $K$ 反而会使端到端速度变慢。\n反之, 即便草稿模型质量极差 ($\\alpha \\approx 0.1$), 也只会造成推理减速而绝不会损及输出质量, 因为投机采样在数学上是严格无损的:\n- 贪心解码 (greedy) 模式: 仅接受与 target 自身 argmax 完全一致的候选 token, 产出序列与 target 独立贪心解码严格逐位相同。\n- 随机采样模式: 按接受率 $\\min(1, p/q)$ 判定是否保留草稿候选 ($p$ 与 $q$ 分别为 target 与 draft 在该位置赋予该 token 的概率); 一旦拒绝则从归一化残差分布 $\\max(0, p - q)$ 中重新采样修正 token。可以严格证明两步合成后的边缘分布恰好恒等于目标分布 $p$。\n因此工程上选型草稿模型只需权衡速度收益: 兼顾较高的接受率 $\\alpha$ 与极低的单步相对开销 $c$。',
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
