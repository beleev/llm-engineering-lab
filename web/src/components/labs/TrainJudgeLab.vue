<!--
  LLM-as-judge 的位置偏差 (对应 llm_train/m21_llm_eval:rule_judge / judge_swap)。
  只讲一件事: 裁判给第一个位置加分, 单次判就偏向 A; 交换顺序各判一次、两次一致才算, 偏差抵消, 代价是平局。
  在前端按 rule_judge 的同一公式现模拟 2000 对 (质量 ~ N(0,1), 裁判 = 质量差 + bias + noise·N(0,1))。
  随机数与 numpy 不同, 数字会和 demo 差一点; demo 第 [4] 段 (bias 0.8, noise 0.5) 的实测写在读数区。
-->
<template>
  <LabFrame
    title="LLM 裁判 — 先出现的回答占便宜吗?"
    sub="每个点是一对回答, 横轴是真实质量差 $q_A - q_B$ (右边 A 更好)。绿点 = 判对, 红点 = 判错, 灰点 = 交换判的平局。
      虚线是单次判的分界: 裁判给第一个位置 (A) 加了 bias 分, 分界左移到 $-\text{bias}$。拖虚线改 bias。"
    module="llm_train/m21"
    run="python -m llm_train.m21_llm_eval.demo"
    :challenge="{
      ask: '单次判时, 红点集中在哪一侧? 切到交换判后, 红点去了哪里?',
      answer: '- 单次判: 红点都在虚线和 0 之间, 即 B 实际更好、但差距小于 bias 的那些对, 被判给了先出现的 A。所以 「好的在 A」 准确率接近 1, 「好的在 B」 掉到 0.6 左右。\n- 交换判: A、B 各当一次第一个, 先手加分对两边一样。只有两次都选同一个才算数, 那一片变成了灰色的平局。\n- 代价: 2 倍调用; bias 0.8 时约 40% 的对判不出 (demo 0.407)。真实做法把平局记 0.5 分, 或加几个裁判投票。\n交换只消位置偏差。长度偏差、自我偏好与位置无关, 消不掉。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'single' }" @click="mode = 'single'">单次判</button>
        <button type="button" :class="{ active: mode === 'swap' }" @click="mode = 'swap'">交换判 (两次一致才算)</button>
        <button type="button" @click="seed++">换一组数据</button>
      </div>
      <LabSlider v-model="bias" label="先手加分 bias" :min="0" :max="1.5" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="noise" label="裁判噪声" :min="0.1" :max="1.5" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="每对回答的真实质量差与裁判结果">
      <line :x1="px(0)" :x2="px(0)" :y1="8" :y2="H - 20" class="zero" />
      <g v-for="t in [-3, -2, -1, 0, 1, 2, 3]" :key="t">
        <text :x="px(t)" :y="H - 6" class="tick" text-anchor="middle">{{ t }}</text>
      </g>
      <circle v-for="(d, i) in shown" :key="i" :cx="px(d.diff)" :cy="d.y" r="2.6" :class="d.cls" />
      <g class="draggable" tabindex="0" role="slider" aria-label="先手加分 bias" :aria-valuenow="bias"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (bias = clamp(Math.round(-pxInv(x) * 20) / 20, 0, 1.5)) })"
        @keydown.left.prevent.stop="bias = clamp(+(bias + 0.05).toFixed(2), 0, 1.5)" @keydown.right.prevent.stop="bias = clamp(+(bias - 0.05).toFixed(2), 0, 1.5)">
        <line :x1="px(-bias)" :x2="px(-bias)" :y1="8" :y2="H - 20" class="bound" />
        <rect :x="px(-bias) - 8" y="4" width="16" height="16" rx="4" class="handle" />
        <text :x="px(-bias) - 12" y="16" class="hl" text-anchor="end">−bias</text>
      </g>
    </svg>

    <template #stats>
      <template v-if="mode === 'single'">
        <div class="kv"><span>选第一个 (A) 的比例</span><b :class="r.pickA > 0.6 ? 'bad' : 'good'">{{ r.pickA.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的在 A</span><b>{{ r.accA1.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的在 B</span><b :class="r.accA1 - r.accB1 > 0.1 ? 'bad' : 'good'">{{ r.accB1.toFixed(3) }}</b></div>
      </template>
      <template v-else>
        <div class="kv"><span>平局比例</span><b :class="r.tie > 0.3 ? 'bad' : ''">{{ r.tie.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的在 A</span><b>{{ r.accA2.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的在 B</span><b :class="Math.abs(r.accA2 - r.accB2) < 0.05 ? 'good' : 'bad'">{{ r.accB2.toFixed(3) }}</b></div>
      </template>
      <div class="kv"><span>真实 A 更好的比例</span><b>{{ r.aBetter.toFixed(3) }}</b></div>
      <p class="lab-note">★ m21 实测 (bias 0.8, noise 0.5): 单次判选 A 0.700, 准确率 0.994 / 0.593; 交换判 0.993 / 0.991, 平局 0.407。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const N = 2000, SHOW = 400
const mode = ref('single'), bias = ref(0.8), noise = ref(0.5), seed = ref(1), svg = ref(null)
const { start } = useDrag()

// 质量和噪声只随 seed 变; 拖 bias / noise 时点不乱跳
const base = computed(() => {
  const rand = mulberry32(seed.value * 7919 + 21)
  return range(N).map(() => ({ qa: randn(rand), qb: randn(rand), e1: randn(rand), e2: randn(rand), jit: rand() }))
})
// rule_judge: 质量差 + bias + 噪声 > 0 → 选第一个
const judge = (qFirst, qSecond, e) => qFirst - qSecond + bias.value + noise.value * e > 0

const sim = computed(() => base.value.map((d) => {
  const pickA = judge(d.qa, d.qb, d.e1)       // A 在前
  const pickB = judge(d.qb, d.qa, d.e2)       // ★ 交换: B 在前
  const v = pickA && !pickB ? 1 : !pickA && pickB ? -1 : 0
  return { ...d, aBetter: d.qa > d.qb, pickA, v }
}))
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0)
const r = computed(() => {
  const s = sim.value, A = s.filter((d) => d.aBetter), B = s.filter((d) => !d.aBetter)
  const dec = (xs) => xs.filter((d) => d.v !== 0)
  return {
    aBetter: A.length / N,
    pickA: mean(s.map((d) => +d.pickA)),
    accA1: mean(A.map((d) => +d.pickA)), accB1: mean(B.map((d) => +!d.pickA)),
    tie: mean(s.map((d) => +(d.v === 0))),
    accA2: mean(dec(A).map((d) => +(d.v === 1))), accB2: mean(dec(B).map((d) => +(d.v === -1))),
  }
})

const W = 560, H = 200, X0 = 14
const px = (v) => X0 + ((clamp(v, -3.5, 3.5) + 3.5) / 7) * (W - 2 * X0)
const pxInv = (x) => ((x - X0) / (W - 2 * X0)) * 7 - 3.5
const shown = computed(() => sim.value.slice(0, SHOW).map((d) => {
  const correct = mode.value === 'single' ? d.pickA === d.aBetter : d.v === (d.aBetter ? 1 : -1)
  const cls = mode.value === 'swap' && d.v === 0 ? 'tie' : correct ? 'ok' : 'bad'
  return { diff: d.qa - d.qb, y: 24 + d.jit * (H - 52), cls }
}))
</script>

<style scoped>
.zero { stroke: var(--border-strong); }
.bound { stroke: var(--warn); stroke-width: 2; stroke-dasharray: 5 4; }
.handle { fill: var(--warn); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.hl { font-size: 10px; fill: var(--warn); font-family: "SF Mono", Menlo, monospace; }
.ok { fill: var(--left); }
.bad { fill: var(--danger); }
.tie { fill: var(--text-dim); opacity: 0.6; }
</style>
