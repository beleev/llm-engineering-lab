<!-- 扩散语言模型并行解码实验台 (对应 llm_infer/m37_diffusion_decoding): make_task + decode 的 JS 移植, Oracle 给精确条件分布。 -->
<template>
  <LabFrame
    title="扩散语言模型的并行解码 — 一步揭开几个位置, 相关的位置会不会拼错"
    sub="- 模拟什么: 掩码扩散语言模型从 32 个 [MASK] 开始, 每步给每个遮住的位置一个分布, 挑一部分位置揭开。这里的模型是合成的 Oracle, 序列由互相独立的组拼成: 独立位置把握很大; 相关短语有三条合法写法, 同一步揭开的位置各取各的最大值, 可能拼出表里没有的组合。Oracle 给的是精确条件分布, 所以拼错全部来自解码策略。
- 图上是什么: 32 个格子是回答的 32 个位置, 问号表示还遮着:
  - 格子底边的色条: 无色是独立位置, 紫色是共用 token 的短语, 蓝色是「三角」短语 (三个位置两两出现过、三个一起不在表里)。
  - 黄色: 这一步揭开的位置。
  - 绿色 / 红色: 播放到最后一步时, 该短语拼对 / 拼错。
- 可以做什么: 切换「固定 k 个」和「置信度阈值」; 拖动 k 或 τ; 拖动「相关短语比例」; 用播放条逐步看揭开顺序; 点击任一格子看它所在短语的写法和这一步的置信度; 点「换一组」换一条序列。
- 应该看到: 固定 k 越大步数越少, 红色越多; 阈值 0.7 以上时独立位置一步揭开一大片, 相关短语一次只揭开一个, 不出现红色。右侧是总步数、平均每步揭开数, 以及这一条和 200 条序列里短语拼错的比例 (有拼错标红)。"
    module="llm_infer/m37"
    run="python -m llm_infer.m37_diffusion_decoding.demo"
    :challenge="{
      ask: '切到「置信度阈值」, 把 τ 从 0.7 降到 0.5。三角短语 (蓝色) 会怎样? 为什么 τ=0.7 时它们不会拼错?',
      answer: 'τ=0.5 时三角短语经常拼错。三角短语的三条写法是 (a,b,x)、(a,y,c)、(z,b,c), 各约 1/3, 所以三个位置的最大概率都约 2/3: 三个都过 0.5, 同一步揭开得到 (a,b,c), 而它不在表里。\nτ=0.7 时: 三个位置的最大概率分别是 $w_0+w_1$、$w_0+w_2$、$w_1+w_2$, 加起来等于 2, 不可能三个都超过 0.7。最多两个同时过阈值, 两个位置都过半时它们的组合一定在表里; 其余情况只揭开置信度最高的一个, 剩下的位置在新条件下重新归一, 不会再拼错。\nFast-dLLM 的定理 1 给的是一般情况: 同一步揭开 n 个位置, 置信度都要高于 $1-\\epsilon$ 且 $(n+1)\\epsilon \\le 1$, n=3 时就是置信度 ≥ 0.75。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'fixed' }" @click="mode = 'fixed'">固定每步 k 个</button>
        <button type="button" :class="{ active: mode === 'threshold' }" @click="mode = 'threshold'">置信度阈值 τ</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-if="mode === 'fixed'" v-model="k" label="每步揭开 k" :min="1" :max="32" />
      <LabSlider v-else v-model="tau" label="阈值 τ" :min="0.3" :max="0.99" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="pGroup" label="相关短语比例" :min="0" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <div class="cells seq" role="group" aria-label="32 个位置">
      <button
        v-for="(t, i) in frame.x" :key="i" type="button" class="cell"
        :class="cellClass(i)" :style="{ boxShadow: BAND[run.kind[i]] }" :aria-pressed="sel === i"
        :aria-label="`位置 ${i}, ${t < 0 ? '遮住' : 'token ' + t}`" @click="sel = i"
      >{{ t < 0 ? '?' : t }}</button>
    </div>
    <StepPlayer :stepper="stepper" :label="isLast ? `全部揭开, 共 ${steps} 步` : `第 ${s + 1} / ${steps} 步, 这一步揭开 ${frame.chosen.length} 个`" />
    <div class="lab-note detail">
      <p>位置 {{ sel }} 所在的组 ({{ KIND[run.kind[sel]] }}) 的合法写法:</p>
      <ul class="pts">
        <li v-for="(p, j) in selGroup.phrases" :key="j"><b>{{ p.join(' ') }}:</b> 概率 {{ selGroup.probs[j].toFixed(2) }}</li>
      </ul>
      <p>这一步开始时它的置信度 (最大条件概率) = {{ frame.x[sel] >= 0 ? '已揭开' : frame.conf[sel].toFixed(2) }}。</p>
    </div>

    <template #stats>
      <div class="kv"><span>总步数 (固定 k=1 要 32 步)</span><b>{{ steps }}</b></div>
      <div class="kv"><span>平均每步揭开</span><b>{{ (L / steps).toFixed(2) }}</b></div>
      <div class="kv"><span>这一条: 拼错的短语</span><b :class="run.bad > 0 ? 'bad' : 'good'">{{ run.bad }} / {{ run.total }}</b></div>
      <div class="kv"><span>200 条平均拼错率</span><b :class="batch.rate > 0 ? 'bad' : 'good'">{{ (batch.rate * 100).toFixed(1) }}% · {{ batch.steps.toFixed(1) }} 步</b></div>
      <div class="lab-note">
        <p>Python demo 的混合负载 (实测, 不随左侧变化): 固定 k=2 要 16 步、拼错 1.2%; τ=0.7 平均 10.2 步、拼错 0。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const L = 32, V = 64
const KIND = ['独立位置', '共用 token 的短语', '三角短语']
const BAND = ['none', 'inset 0 -3px 0 var(--accent)', 'inset 0 -3px 0 var(--eye)']
const mode = ref('threshold'), k = ref(4), tau = ref(0.7), pGroup = ref(0.15), seed = ref(0), sel = ref(0)

const gamma = (r, a) => -sum(range(a).map(() => Math.log(r() + 1e-12)))        // 整数 α 的 Gamma 抽样
const dirichlet = (r, as) => { const g = as.map((a) => gamma(r, a)); const s = sum(g); return g.map((x) => x / s) }
const pickDistinct = (r, n) => { const a = range(V); for (let i = 0; i < n; i++) { const j = i + Math.floor(r() * (V - i)); [a[i], a[j]] = [a[j], a[i]] } return a.slice(0, n) }

function makeTask(r, p) {                                                       // 同 make_task: 独立位置 / 共用 token 短语 / 三角短语
  const groups = []
  let pos = 0
  while (pos < L) {
    if (r() < p && L - pos >= 3 && r() < 0.5) {
      const [a, b, c, x, y, z] = pickDistinct(r, 6)
      groups.push({ start: pos, kind: 2, phrases: [[a, b, x], [a, y, c], [z, b, c]], probs: dirichlet(r, [20, 20, 20]) })
    } else if (r() < p && L - pos >= 2) {
      const n = Math.min(2 + Math.floor(r() * 3), L - pos), t = pickDistinct(r, 2 * n), d = Math.floor(r() * n)
      const p0 = t.slice(0, n), shared = t.slice(n), p2 = [...shared]
      p2[d] = (shared[d] + 1 + Math.floor(r() * (V - 1))) % V
      while (p2[d] === p0[d]) p2[d] = (p2[d] + 1) % V
      groups.push({ start: pos, kind: 1, phrases: [p0, shared, p2], probs: dirichlet(r, [5, 4, 4]) })
    } else {
      const top = 0.92 + r() * 0.075, t = pickDistinct(r, 4)
      groups.push({ start: pos, kind: 0, phrases: t.map((v) => [v]), probs: [top, ...Array(3).fill((1 - top) / 3)] })
    }
    pos += groups.at(-1).phrases[0].length
  }
  return groups
}

function marginals(groups, x) {                                                 // 在和已揭开位置一致的写法上重新归一
  const P = range(L).map(() => new Map())
  for (const g of groups) {
    const n = g.phrases[0].length
    let w = g.probs.map((p, j) => (g.phrases[j].every((t, i) => x[g.start + i] < 0 || x[g.start + i] === t) ? p : 0))
    const s = sum(w)
    w = s > 0 ? w.map((v) => v / s) : g.probs                                   // 已经拼错: 剩下的位置按无条件分布填
    for (let i = 0; i < n; i++) g.phrases.forEach((ph, j) => P[g.start + i].set(ph[i], (P[g.start + i].get(ph[i]) || 0) + w[j]))
  }
  return P.map((m) => [...m.entries()].reduce((a, b) => (b[1] > a[1] ? b : a)))   // 每个位置的 [argmax token, 置信度]
}

function decode(groups, m, kk, t) {
  const x = Array(L).fill(-1), frames = []
  while (x.some((v) => v < 0)) {
    const best = marginals(groups, x)
    const conf = best.map(([, c], i) => (x[i] < 0 ? c : -1))
    const order = range(L).sort((a, b) => conf[b] - conf[a] || a - b)
    const nMasked = x.filter((v) => v < 0).length
    const n = m === 'fixed' ? Math.min(kk, nMasked) : Math.max(1, conf.filter((c) => c > t).length)   // ★ 都不过阈值就只揭开 1 个
    const chosen = order.slice(0, n)
    frames.push({ x: [...x], chosen, conf: best.map(([, c]) => c) })
    chosen.forEach((i) => { x[i] = best[i][0] })                                // 同一步的位置各取各的最大值
  }
  frames.push({ x: [...x], chosen: [], conf: Array(L).fill(1) })
  const multi = groups.filter((g) => g.phrases[0].length > 1)
  const isBad = (g) => !g.phrases.some((ph) => ph.every((v, i) => x[g.start + i] === v))
  return { frames, x, bad: multi.filter(isBad).length, total: multi.length, badGroups: multi.filter(isBad) }
}

const run = computed(() => {
  const groups = makeTask(mulberry32(seed.value * 7919 + 1), pGroup.value)
  const res = decode(groups, mode.value, k.value, tau.value)
  const kind = [], gid = []
  groups.forEach((g, gi) => g.phrases[0].forEach(() => { kind.push(g.kind); gid.push(gi) }))
  const badPos = new Set(res.badGroups.flatMap((g) => range(g.phrases[0].length).map((i) => g.start + i)))
  return { ...res, groups, kind, gid, badPos }
})
const batch = computed(() => {                                                  // 200 条序列的平均, 和 Python 表同一口径
  let bad = 0, total = 0, steps = 0
  for (let i = 0; i < 200; i++) {
    const r = decode(makeTask(mulberry32(i * 104729 + 17), pGroup.value), mode.value, k.value, tau.value)
    bad += r.bad; total += r.total; steps += r.frames.length - 1
  }
  return { rate: total ? bad / total : 0, steps: steps / 200 }
})
const stepper = useStepper(() => run.value.frames.length)
watch(run, () => stepper.reset())
const s = computed(() => Math.min(stepper.step.value, run.value.frames.length - 1))
const frame = computed(() => run.value.frames[s.value])
const selGroup = computed(() => run.value.groups[run.value.gid[sel.value]])
const steps = computed(() => run.value.frames.length - 1)                     // 最后一帧是「全部揭开」, 不算一步
const isLast = computed(() => s.value === steps.value)
const cellClass = (i) => ({
  hot: frame.value.chosen.includes(i), on: sel.value === i,
  bad: isLast.value && run.value.badPos.has(i), ok: isLast.value && run.value.kind[i] > 0 && !run.value.badPos.has(i),
})
</script>

<style scoped>
.seq { grid-template-columns: repeat(16, minmax(22px, 1fr)); margin-bottom: 8px; }
.seq .cell { height: 30px; cursor: pointer; }
.detail { margin-top: 10px; }
</style>
