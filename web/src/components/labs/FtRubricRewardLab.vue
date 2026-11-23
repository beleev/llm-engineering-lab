<!--
  标量 RM vs rubric 奖励实验台。只讲一件事: 偏好标签里的长度偏差被标量 RM 学走之后,
  在基座分布内挑回复 (best-of-N) 时它只吃掉一点选择力; 用 RL 把策略推出分布时, 策略就沿长度方向钻空子。
  与 llm_finetune/run_finetune/rubric_reward/demo.py 是同一套设定; 前端直接用标注公式的系数当 RM。
-->
<template>
  <LabFrame
    title="标量 RM vs rubric — 长度偏差什么时候会被钻空子"
    sub="- 模拟什么: 后训练里给回复打分的两种办法。标量 RM (reward model, 用 Bradley-Terry 损失从偏好对学出的打分器) 只输出一个数; rubric 奖励先为每道题写一份评分清单, judge 逐条判「满足 / 不满足」再求满足比例。这里的回复用 5 条清单条目加长度表示: 前 4 条是内容条目, 第 5 条是「简洁」, 长度超过 300 token 就违反。偏好标签带长度偏差 β: 标注者把多写 100 token 当成 β 个 logit 的好处。
- 图上是什么: 上半张图的纵轴是真实质量 (5 条里满足的比例), 下半张图的纵轴是被选中回复的平均长度, 虚线是 300 token 的简洁上限。横轴在 best-of-N 模式下是候选数 N, 在 RL 模式下是训练步数。灰色虚线是按真实质量挑 (oracle), 黄线是标量 RM, 绿线是 rubric。
- 可以做什么: 拖「标注的长度偏差 β」改变 RM 学到的长度系数; 拖「judge 判错率 ε」改变 rubric 每条判反的概率; 两个模式按钮切换优化压力: best-of-N 只在基座分布里挑, RL 让策略自己移动长度和内容; 点图上任意一列 (或聚焦后按回车) 选中这个 N 或这一步, 右侧读数随之变化; 「换一组」换随机种子。
- 应该看到: best-of-N 下 RM 挑出的回复随 N 慢慢变长, 但质量仍接近 oracle; 切到 RL, RM 一侧的长度冲到 600 上限, 真实质量停在 0.8 左右, rubric 一侧的质量接近 1、长度留在 300 以内。右侧读数: 两种奖励在选中位置的真实质量, 高的一方变绿; 平均长度超过 300 变红。把 β 拖到 0, RM 就不再往长处跑。"
    module="llm_finetune/run_finetune/rubric_reward/demo.py"
    run="python -m llm_finetune.run_finetune.rubric_reward.demo"
    :challenge="{
      ask: '保持 β = 0.8、ε = 0.1。\n1. best-of-N 模式下, N = 64 时标量 RM 和 rubric 谁挑出的回复真实质量更高?\n2. 切到 RL 模式, 300 步后呢?\n3. 同一个 RM, 为什么两种优化压力下结论相反?',
      answer: '1. best-of-N 下标量 RM 更高 (前端约 0.97 对 0.90), Python 实测 0.974 对 0.916。\n2. RL 下反过来: RM 一侧真实质量停在 0.79 左右, 长度到 600 上限; rubric 一侧到 0.99, 长度在 300 以内。\n3. 原因在于 RM 只在训练分布内被校准过:\n- best-of-N 的候选全部来自基座分布, 长度集中在 200 ± 60, RM 的长度系数只能在这个范围里起作用, 选择力主要还是花在内容条目上。这里的 RM 直接看得见每条条目, 比真实 RM 读文本容易得多。\n- RL 的策略可以把长度推到训练数据里从没出现过的位置。RM 的长度项是线性的, 越长分越高, 多写 100 token 的收益 β = 0.8 与违反「简洁」条目的损失 (约 0.85) 相当, 300 token 之后再写长仍在加分, 于是策略一路写长。RM 分一直在涨, 真实质量却因为丢了「简洁」条目而停在 0.8 附近。这就是 reward hacking: 代理奖励和真实目标在分布内相关, 被优化推出分布后分道扬镳。\nrubric 的分数里没有长度这一项, 只能靠真的满足条目来涨分; judge 的随机判错只让学习变慢, 不改变最优点。代价是每道题都要写清单, judge 的错误如果有系统偏向 (例如看见长回答就判「讲全了」), 同样会被钻空子。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" :aria-pressed="mode === m.id" @click="mode = m.id; sel = m.xs.length - 1">{{ m.name }}</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="beta" label="标注的长度偏差 β (每 100 token)" :min="0" :max="1.5" :step="0.1" :format="(v) => v.toFixed(1)" />
      <LabSlider v-model="eps" label="judge 判错率 ε" :min="0" :max="0.4" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg viewBox="0 0 640 330" role="group" aria-label="不同奖励下被选中回复的真实质量与长度">
      <text x="8" y="16" class="t">真实质量</text>
      <text x="8" y="196" class="t">平均长度</text>
      <line v-for="g in [0.6, 0.8, 1.0]" :key="'g' + g" :x1="X0" :x2="X1" :y1="yq(g)" :y2="yq(g)" class="grid" />
      <text v-for="g in [0.6, 0.8, 1.0]" :key="'gl' + g" :x="X0 - 6" :y="yq(g) + 4" text-anchor="end" class="t">{{ g.toFixed(1) }}</text>
      <line :x1="X0" :x2="X1" :y1="yl(300)" :y2="yl(300)" stroke="var(--danger)" stroke-dasharray="4 4" />
      <text :x="X1" :y="yl(300) - 4" text-anchor="end" class="t">简洁上限 300</text>
      <text v-for="g in [0, 300, 600]" :key="'ll' + g" :x="X0 - 6" :y="yl(g) + 4" text-anchor="end" class="t">{{ g }}</text>
      <g v-for="a in ARMS" :key="a.id">
        <polyline :points="line(a.id, 'q')" fill="none" :stroke="`var(--${a.c})`" stroke-width="2" :stroke-dasharray="a.dash" />
        <polyline :points="line(a.id, 'L')" fill="none" :stroke="`var(--${a.c})`" stroke-width="2" :stroke-dasharray="a.dash" />
      </g>
      <text v-for="(x, i) in xs" :key="'x' + i" :x="xp(i)" y="322" text-anchor="middle" class="t">{{ x }}</text>
      <!-- ★ 点一列 = 把手放在这个 N / 这一步上 -->
      <rect
        v-for="(x, i) in xs" :key="'c' + i" :x="xp(i) - colW / 2" y="24" :width="colW" height="286"
        :class="['col', { on: sel === i }]" tabindex="0" role="button" :aria-label="`选中 ${xLabel} ${x}`" :aria-pressed="sel === i"
        @click="sel = i" @keydown.enter="sel = i"
      />
      <g v-for="a in ARMS" :key="'legend' + a.id">
        <line :x1="a.lx" :x2="a.lx + 18" y1="12" y2="12" :stroke="`var(--${a.c})`" stroke-width="2" :stroke-dasharray="a.dash" />
        <text :x="a.lx + 22" y="16" class="t">{{ a.name }}</text>
      </g>
    </svg>

    <template #stats>
      <p class="hint">{{ xLabel }} = {{ xs[sel] }}</p>
      <div class="kv"><span>标量 RM 的真实质量</span><b :class="pt.rm.q >= pt.rubric.q ? 'good' : 'bad'">{{ pt.rm.q.toFixed(3) }}</b></div>
      <div class="kv"><span>rubric 的真实质量</span><b :class="pt.rubric.q >= pt.rm.q ? 'good' : 'bad'">{{ pt.rubric.q.toFixed(3) }}</b></div>
      <div class="kv"><span>标量 RM 的平均长度</span><b :class="pt.rm.L <= 300 ? 'good' : 'bad'">{{ pt.rm.L.toFixed(0) }}</b></div>
      <div class="kv"><span>rubric 的平均长度</span><b :class="pt.rubric.L <= 300 ? 'good' : 'bad'">{{ pt.rubric.L.toFixed(0) }}</b></div>
      <div class="lab-note">
        <p>前端直接用标注公式的系数当 RM (每条条目 0.8, 长度 β); Python 里 Bradley-Terry 训练学到的是条目 0.74–0.90、长度 0.81。</p>
        <p>Python 实测 (β=0.8, ε=0.1, 不随左侧变化): best-of-N 在 N=64 时 RM 0.974 / 256 token, rubric 0.916 / 195 token; RL 300 步后 RM 0.792 / 600 token, rubric 0.994 / 143 token。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, clamp } from '@/utils/labmath.js'

const X0 = 48, X1 = 620
const MODES = [
  { id: 'bon', name: 'best-of-N (分布内挑)', xs: [1, 2, 4, 8, 16, 32, 64] },
  { id: 'rl', name: 'RL (策略可离开分布)', xs: [0, 25, 50, 75, 100, 150, 200, 250, 300] },
]
const ARMS = [
  { id: 'oracle', name: 'oracle', c: 'text-dim', dash: '4 3', lx: 120 },
  { id: 'rm', name: '标量 RM', c: 'warn', dash: '', lx: 220 },
  { id: 'rubric', name: 'rubric', c: 'left', dash: '', lx: 330 },
]
const beta = ref(0.8), eps = ref(0.1), mode = ref('bon'), seed = ref(1), sel = ref(6)
const xs = computed(() => MODES.find((m) => m.id === mode.value).xs)
const xLabel = computed(() => (mode.value === 'bon' ? 'N' : '步数'))
const colW = computed(() => (X1 - X0) / xs.value.length)
const xp = (i) => X0 + colW.value * (i + 0.5)
const yq = (q) => 30 + (1 - q) / 0.5 * 140              // 质量 0.5–1.0 → 上半张图
const yl = (L) => 300 - L / 600 * 100                   // 长度 0–600 → 下半张图

// 一条回复: 4 个内容条目 + 长度; 第 5 条「简洁」由长度决定。判定的随机数总是抽, 改 ε 不会换候选
function reply(rand, p, mu) {
  const content = p.map((pi) => (rand() < pi ? 1 : 0))
  const raw = mu + 60 * randn(rand), L = Math.round(clamp(raw, 20, 600))
  const items = [...content, L <= 300 ? 1 : 0]
  const judged = items.map((s) => (rand() < eps.value ? 1 - s : s))
  const q = items.reduce((a, b) => a + b, 0) / 5
  const score = { oracle: q + 1e-6 * rand(), rm: 0.8 * q * 5 + beta.value * L / 100, rubric: judged.reduce((a, b) => a + b, 0) / 5 }   // ★ RM 带长度项, rubric 没有
  return { content, raw, L, q, score }
}

function bestOfN(rand) {
  return xs.value.map((n) => {
    const pick = { oracle: null, rm: null, rubric: null }, acc = { oracle: [0, 0], rm: [0, 0], rubric: [0, 0] }
    for (let t = 0; t < 200; t++) {          // 200 道题: 拖滑杆时重算一次约 70 ms
      const cands = Array.from({ length: n }, () => reply(rand, [0.5, 0.5, 0.5, 0.5], 200))
      for (const a of Object.keys(pick)) {
        pick[a] = cands.reduce((b, c) => (c.score[a] > b.score[a] ? c : b))
        acc[a][0] += pick[a].q / 200; acc[a][1] += pick[a].L / 200
      }
    }
    return Object.fromEntries(Object.entries(acc).map(([a, [q, L]]) => [a, { q, L }]))
  })
}

// REINFORCE + Adam, 参数 = 4 个内容 logit + 长度均值 (200 + 100·μ), 批均值当 baseline; 每步 128 条 (Python 256), 其余与 Python 同设定
function rlArm(rand, arm) {
  const th = [0, 0, 0, 0, 0], m = th.map(() => 0), v = th.map(() => 0), out = {}
  for (let step = 0; step <= 300; step++) {
    const p = th.slice(0, 4).map((x) => 1 / (1 + Math.exp(-x))), mu = 200 + 100 * th[4]
    const batch = Array.from({ length: 128 }, () => reply(rand, p, mu))
    if (xs.value.includes(step)) out[step] = { q: batch.reduce((s, b) => s + b.q, 0) / 128, L: batch.reduce((s, b) => s + b.L, 0) / 128 }
    const base = batch.reduce((s, b) => s + b.score[arm], 0) / 128
    const g = th.map(() => 0)
    for (const b of batch) {
      const A = (b.score[arm] - base) / 128
      b.content.forEach((c, i) => { g[i] += A * (c - p[i]) })                // ∂log π/∂logit = c − p
      g[4] += A * (b.raw - mu) * 100 / 3600                                   // ∂log N/∂μ, μ 以 100 token 为单位
    }
    g.forEach((gi, i) => {                                                     // Adam 上升一步, lr 0.05
      m[i] = 0.9 * m[i] + 0.1 * gi; v[i] = 0.999 * v[i] + 0.001 * gi * gi
      th[i] += 0.05 * (m[i] / (1 - 0.9 ** (step + 1))) / (Math.sqrt(v[i] / (1 - 0.999 ** (step + 1))) + 1e-8)
    })
  }
  return out
}

const curves = computed(() => {
  if (mode.value === 'bon') return bestOfN(mulberry32(seed.value))
  const runs = Object.fromEntries(ARMS.map((a) => [a.id, rlArm(mulberry32(seed.value), a.id)]))
  return xs.value.map((s) => Object.fromEntries(ARMS.map((a) => [a.id, runs[a.id][s]])))
})
const line = (arm, key) => curves.value.map((c, i) => `${xp(i)},${key === 'q' ? yq(c[arm].q) : yl(c[arm].L)}`).join(' ')
const pt = computed(() => curves.value[Math.min(sel.value, curves.value.length - 1)])
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.grid { stroke: var(--border); stroke-dasharray: 2 4; }
.col { fill: transparent; cursor: pointer; }
.col.on, .col:focus-visible { fill: var(--accent); fill-opacity: 0.1; outline: none; }
</style>
