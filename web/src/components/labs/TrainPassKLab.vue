<!--
  pass@k 的两个估计 (对应 llm_train/m21_llm_eval:pass_at_k)。
  只讲一件事: 采 n 个、c 个对时, 直接代 1-(1-c/n)^k 会系统性偏低; 无偏估计 1-C(n-c,k)/C(n,k) 的期望恰好等于真值。
  期望在前端按二项分布精确求和 (不是抽样)。数据集级的蒙特卡洛结果取自 m21 demo 第 [3] 段输出。
-->
<template>
  <LabFrame
    title="pass@k — 朴素公式为什么偏低?"
    sub="一道题的真实通过率是 $p$, 我们采 $n$ 个样本、数出 $c$ 个对, 再估 pass@k。
      实线是真值 $1-(1-p)^k$, 虚线是朴素公式 $1-(1-c/n)^k$ 在 $c \sim \mathrm{Binomial}(n, p)$ 下的期望。拖动圆点换 $p$。"
    module="llm_train/m21"
    run="python -m llm_train.m21_llm_eval.demo"
    :challenge="{
      ask: '把 $k$ 拉到 1, 两条线会怎样? 再把 $n$ 拉到和 $k$ 一样大呢?',
      answer: '- $k=1$: 两个公式都是 $c/n$, 是 $p$ 的无偏估计, 偏差为 0 (demo: 0.1883 / 0.1884 / 0.1884)。\n- $k \\gt 1$: $1-(1-x)^k$ 对 $x$ 是凹的。由 Jensen 不等式, 把带噪声的 $c/n$ 代进去, 期望一定低于真值。$k$ 越大越凹, 偏得越多。\n- $n = k$: 无偏估计退化成 「采 $k$ 个里有没有对的」, 仍然无偏, 但方差大。标准做法是采 $n \\ge k$ 个, 用无偏公式。\n无偏公式 $1-\\binom{n-c}{k}/\\binom{n}{k}$ 是 「从 $n$ 个里无放回挑 $k$ 个, 至少一个对」 的概率; 朴素公式等于有放回地挑。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="每题采样数 n" :min="1" :max="50" />
      <LabSlider v-model="k" label="k" :min="1" :max="n" />
      <div class="row">
        <button v-for="kk in [1, 5, 10]" :key="kk" type="button" :class="{ active: k === kk && n === 20 }" @click="(n = 20), (k = kk)">n=20, k={{ kk }}</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="pass@k 真值与朴素估计的期望随通过率变化, 曲线上的圆点可拖">
      <line :x1="X0" :x2="W - 10" :y1="py(0)" :y2="py(0)" class="axis" />
      <line :x1="X0" :x2="X0" :y1="py(0)" :y2="py(1)" class="axis" />
      <g v-for="t in [0, 0.25, 0.5, 0.75, 1]" :key="t">
        <text :x="px(t)" :y="py(0) + 14" class="tick" text-anchor="middle">{{ t }}</text>
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <text :x="W - 10" :y="py(0) - 6" class="tick" text-anchor="end">真实通过率 p →</text>
      <polygon :points="gap" class="gap" />
      <polyline :points="line(truth)" class="truth" />
      <polyline :points="line(naiveMean)" class="naive" />
      <g class="draggable" tabindex="0" role="slider" aria-label="真实通过率 p" :aria-valuenow="p"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (p = clamp(Math.round(pxInv(x) * 100) / 100, 0, 1)) })"
        @keydown.right.prevent.stop="p = clamp(+(p + 0.01).toFixed(2), 0, 1)" @keydown.left.prevent.stop="p = clamp(+(p - 0.01).toFixed(2), 0, 1)">
        <line :x1="px(p)" :x2="px(p)" :y1="py(0)" :y2="py(1)" class="guide" />
        <circle :cx="px(p)" :cy="py(truth(p))" r="7" class="dot" />
        <circle :cx="px(p)" :cy="py(naiveMean(p))" r="5" class="dot2" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>p = {{ p.toFixed(2) }}: 真值</span><b>{{ truth(p).toFixed(4) }}</b></div>
      <div class="kv"><span>无偏估计的期望</span><b :class="Math.abs(unbMean(p) - truth(p)) < 0.005 ? 'good' : 'bad'">{{ unbMean(p).toFixed(4) }}</b></div>
      <div class="kv"><span>朴素公式的期望</span><b :class="bias < -0.005 ? 'bad' : ''">{{ naiveMean(p).toFixed(4) }}</b></div>
      <div class="kv"><span>朴素偏差</span><b :class="bias < -0.005 ? 'bad' : 'good'">{{ bias >= 0 ? '+' : '' }}{{ bias.toFixed(4) }}</b></div>
      <div class="lab-note">
        <p>★ 无偏公式与 Python 的 pass_at_k 相同: 1 − ∏(1 − k/i), i 从 n−c+1 到 n。</p>
        <p>m21 数据集级实测, 不随滑杆变化 (300 题, p ~ Beta(0.5, 2), n=20): k=10 真值 0.5864, 无偏 0.5861, 朴素 0.5470 (−0.0394)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const n = ref(20), k = ref(10), p = ref(0.15), svg = ref(null)
const { start } = useDrag()
watch(n, (v) => { if (k.value > v) k.value = v })

// 与 llm_train/m21_llm_eval/demo.py:pass_at_k 相同
const passAtK = (nn, c, kk) => (nn - c < kk ? 1 : 1 - range(c).reduce((acc, j) => acc * (1 - kk / (nn - c + 1 + j)), 1))
// 二项分布概率 (对数域, n ≤ 50 足够稳)
const LOGFACT = range(52).reduce((acc, i) => (acc.push(i ? acc[i - 1] + Math.log(i) : 0), acc), []) // log(i!)
const lgamma = (x) => LOGFACT[x - 1] // log((x-1)!), x 为正整数
const binom = (nn, c, q) => {
  if (q <= 0) return c === 0 ? 1 : 0
  if (q >= 1) return c === nn ? 1 : 0
  return Math.exp(lgamma(nn + 1) - lgamma(c + 1) - lgamma(nn - c + 1) + c * Math.log(q) + (nn - c) * Math.log(1 - q))
}
const expect = (f, q) => range(n.value + 1).reduce((s, c) => s + binom(n.value, c, q) * f(c), 0)
const truth = (q) => 1 - (1 - q) ** k.value
const unbMean = (q) => expect((c) => passAtK(n.value, c, k.value), q)
const naiveMean = (q) => expect((c) => 1 - (1 - c / n.value) ** k.value, q) // ★ 凹函数 → 期望偏低
const bias = computed(() => naiveMean(p.value) - truth(p.value))

const W = 560, H = 240, X0 = 34
const px = (v) => X0 + v * (W - X0 - 14)
const pxInv = (x) => (x - X0) / (W - X0 - 14)
const py = (v) => 12 + (1 - v) * (H - 38)
const PS = range(101).map((i) => i / 100)
const line = (f) => PS.map((q) => `${px(q)},${py(f(q))}`).join(' ')
const gap = computed(() => [...PS.map((q) => `${px(q)},${py(truth(q))}`), ...[...PS].reverse().map((q) => `${px(q)},${py(naiveMean(q))}`)].join(' '))
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.truth { fill: none; stroke: var(--left); stroke-width: 2.5; }
.naive { fill: none; stroke: var(--danger); stroke-width: 2; stroke-dasharray: 6 4; }
.gap { fill: var(--danger); opacity: 0.18; }
.guide { stroke: var(--text-dim); stroke-dasharray: 2 3; }
.dot { fill: var(--left); stroke: var(--bg); stroke-width: 2; }
.dot2 { fill: var(--danger); stroke: var(--bg); stroke-width: 2; }
</style>
