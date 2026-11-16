<!--
  LLM-as-judge 的位置偏差 (对应 llm_train/m21_llm_eval:rule_judge / judge_swap)。
  只讲一件事: 裁判给第一个位置加分, 单次判就偏向 A; 交换顺序各判一次、两次一致才算, 偏差抵消, 代价是平局。
  在前端按 rule_judge 的同一公式现模拟 2000 对 (质量 ~ N(0,1), 裁判 = 质量差 + bias + noise·N(0,1))。
  随机数与 numpy 不同, 数字会和 demo 差一点; demo 第 [4] 段 (bias 0.8, noise 0.5) 的实测写在读数区。
-->
<template>
  <LabFrame
    title="LLM 裁判的位置偏差 — 先出现的回答占便宜吗? 交换顺序能治吗?"
    sub="- 模拟什么: 写作、对话这类没有标准答案的题, 常请一个大模型当裁判 (LLM-as-judge), 把两个回答 A、B 一起给它, 问哪个更好。真实的 LLM 裁判有位置偏差: 偏爱先出现的那个回答。这个台子用一条规则模拟裁判: 打分 = 两个回答的真实质量差 + 先手加分 bias + 随机噪声, 大于 0 就选排在前面的那个。交换判是消位置偏差的标准做法: A 在前判一次, B 在前再判一次, 两次都选同一个才算它赢, 否则记平局。前端按同一公式现模拟 2000 对回答, 图上画出其中 400 对。
      - 图上是什么: 每个点是一对回答, 横轴是真实质量差 $q_A - q_B$: 右边是 A 真的更好, 左边是 B 更好; 纵向位置是随机的, 只为把点铺开。绿点是判对, 红点是判错, 灰点是交换判里的平局。中间的实线是 0。黄色虚线是单次判的分界: 裁判给第一个位置 (A) 加了 bias 分, 所以分界不在 0, 而是左移到 −bias; 虚线和 0 之间那一片点 B 其实更好, 却会被判给 A。
      - 可以做什么: 点「单次判」或「交换判 (两次一致才算)」切换判法; 拖「先手加分 bias」滑杆, 或直接左右拖黄色虚线, 改裁判偏心的程度; 拖「裁判噪声」改裁判随机出错的幅度; 「换一组数据」换一批回答。
      - 应该看到: 单次判时红点集中在虚线和 0 之间。右侧「选第一个 (A) 的比例」在 bias 大时远超一半, 而最后一行「真实 A 更好的比例」只有一半左右; 「准确率: 好的在 A / 好的在 B」分开算两种情况各自的判对率, 好的在 B 时明显更低, 两者差距超过 0.1 变红。切到交换判, 那一片红点变成灰色平局, 两个准确率几乎一样 (差小于 0.05 变绿); 代价是「平局比例」, bias 0.8 时约四成的对判不出胜负, 超过 0.3 变红。"
    module="llm_train/m21"
    run="python -m llm_train.m21_llm_eval.demo"
    :challenge="{
      ask: '单次判时, 红点集中在哪一侧? 切到交换判后, 红点去了哪里?',
      answer: '单次判的红点集中在虚线和 0 之间 (0 的左侧); 交换判后它们变成了灰色的平局。\n- 单次判: 裁判打分 = 质量差 + bias + 噪声, 所以 A 只要不比 B 差超过 bias 就会被选中。横轴在 −bias 和 0 之间的点, B 其实更好但差距小于 bias, 全被判给了先出现的 A, 这就是那一片红点。把 bias 拖大, 这片红区跟着变宽。结果是「好的在 A」准确率接近 1, 「好的在 B」掉到 0.6 左右 (m21 实测 0.994 对 0.593), 「选第一个的比例」0.700, 而真实 A 更好的只有 0.498。噪声大的话 0 附近两侧还会有零星红点, 那是随机出错, 和位置无关。\n- 交换判: A、B 各当一次第一个, 先手加分两边都拿到一次, 互相抵消。只有两次都选同一个才算它赢: 质量差明显的对两次结果一致, 判对; 原来那片红区里的对, A 在前时选 A、B 在前时选 B, 两次不一致, 记平局, 所以红点变成灰点。两种情况的准确率回到 0.99 左右 (实测 0.993 / 0.991)。\n- 代价: 裁判调用翻倍; bias 0.8 时约 40% 的对判不出胜负 (实测平局 0.407), 都是质量接近、偏差压过真实差距的那些。真实做法把平局记 0.5 分, 或再加几个裁判投票。\n交换只消位置偏差。真实 LLM 裁判还有长度偏差 (偏爱长回答) 和自我偏好 (偏爱自家模型的回答), 这两样与位置无关, 交换消不掉。',
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

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="每对回答的真实质量差与裁判结果, 分界虚线可拖">
      <line :x1="px(0)" :x2="px(0)" :y1="8" :y2="H - 20" class="zero" />
      <g v-for="t in [-3, -2, -1, 0, 1, 2, 3]" :key="t">
        <text :x="px(t)" :y="H - 6" class="tick" text-anchor="middle">{{ t }}</text>
      </g>
      <circle v-for="(d, i) in shown" :key="i" :cx="px(d.diff)" :cy="d.y" r="2.6" :class="d.cls" />
      <g class="draggable focus-inset" tabindex="0" role="slider" aria-label="先手加分 bias" :aria-valuenow="bias"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (bias = clamp(Math.round(-pxInv(x) * 20) / 20, 0, 1.5)) })"
        @keydown.left.prevent.stop="bias = clamp(+(bias + 0.05).toFixed(2), 0, 1.5)" @keydown.right.prevent.stop="bias = clamp(+(bias - 0.05).toFixed(2), 0, 1.5)">
        <line :x1="px(-bias)" :x2="px(-bias)" :y1="8" :y2="H - 20" class="bound" />
        <rect :x="px(-bias) - 8" y="4" width="16" height="16" rx="4" class="handle" />
        <text :x="px(-bias) - 12" y="16" class="hl" text-anchor="end">−bias</text>
      </g>
    </svg>

    <template #stats>
      <template v-if="mode === 'single'">
        <div class="kv"><span>选第一个 (A) 的比例 (没有偏差时应接近下面「真实 A 更好」的比例)</span><b :class="r.pickA > 0.6 ? 'bad' : 'good'">{{ r.pickA.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 A 位时判对的比例</span><b>{{ r.accA1.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 B 位时判对的比例</span><b :class="r.accA1 - r.accB1 > 0.1 ? 'bad' : 'good'">{{ r.accB1.toFixed(3) }}</b></div>
      </template>
      <template v-else>
        <div class="kv"><span>平局比例 (两次判不一致, 给不出胜负)</span><b :class="r.tie > 0.3 ? 'bad' : ''">{{ r.tie.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 A 位时判对的比例 (只算分出胜负的)</span><b>{{ r.accA2.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 B 位时判对的比例 (只算分出胜负的)</span><b :class="Math.abs(r.accA2 - r.accB2) < 0.05 ? 'good' : 'bad'">{{ r.accB2.toFixed(3) }}</b></div>
      </template>
      <div class="kv"><span>真实 A 更好的比例 (数据本身, 和裁判无关)</span><b>{{ r.aBetter.toFixed(3) }}</b></div>
      <p class="lab-note">★ 这里的随机数和 numpy 不同, 数字会和 demo 差一点。m21 实测 (bias 0.8, noise 0.5, 不随滑杆变化): 单次判选 A 的比例 0.700, 准确率 好的在 A 0.994 / 在 B 0.593; 交换判准确率 0.993 / 0.991, 平局 0.407。</p>
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
