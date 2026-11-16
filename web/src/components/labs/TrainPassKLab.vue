<!--
  pass@k 的两个估计 (对应 llm_train/m21_llm_eval:pass_at_k)。
  只讲一件事: 采 n 个、c 个对时, 直接代 1-(1-c/n)^k 会系统性偏低; 无偏估计 1-C(n-c,k)/C(n,k) 的期望恰好等于真值。
  期望在前端按二项分布精确求和 (不是抽样)。数据集级的蒙特卡洛结果取自 m21 demo 第 [3] 段输出。
-->
<template>
  <LabFrame
    title="pass@k 的两种算法 — 把 c/n 直接代进公式, 为什么平均总是算低?"
    sub="- 模拟什么: 代码题常用 pass@k 打分, 意思是让模型对一道题生成 k 个答案, 其中至少有一个通过单元测试的概率。实际评测时每题先采 n 个样本 (n 不小于 k), 跑测试数出其中 c 个是对的, 再用 n 和 c 估 pass@k。有两种算法。朴素公式把 c/n 当成这道题的通过率 p, 直接代进 $1-(1-p)^k$。无偏估计是 $1-\binom{n-c}{k}/\binom{n}{k}$, 其中 $\binom{n}{k}$ 是组合数, 读作「n 选 k」, 表示从 n 个东西里挑 k 个一共有多少种挑法; 整个式子算的是从这 n 个样本里不放回地挑 k 个、至少挑到一个对的概率。「无偏」的意思是: 同一道题反复重采 n 个样本, 每次算出的估计值有高有低, 但平均值正好等于真值。这个台子假设题目的真实通过率 p 已知, 这时 c 服从二项分布 (采 n 个, 每个独立地以概率 p 答对, c 取 0 到 n 各有一个概率); 前端把每个 c 的概率乘上用它算出的估计值再求和, 精确算出两种公式的平均值 (期望), 不靠随机抽样。
      - 图上是什么: 横轴是这道题的真实通过率 p, 从 0 到 1; 纵轴是 pass@k 的值。绿色实线是真值 $1-(1-p)^k$, 也就是知道 p 时试 k 次至少对一次的真实概率。红色虚线是朴素公式的期望: 在这个 p 下反复采 n 个样本、每次把 c/n 代进公式, 算出来的平均值。两条线之间的淡红色区域, 就是朴素公式平均算低了多少。无偏估计的期望和绿线完全重合, 图上不另画, 只在右侧读数里给出。竖直的点线标出当前选中的 p, 大绿点在真值线上, 小红点在朴素线上。
      - 可以做什么: 拖「每题采样数 n」改每题采几个样本 (1 到 50)。拖「k (pass@k 里试几次)」改 k, 它最多等于 n, 把 n 调到比 k 小时 k 会自动跟着降。三个按钮「n=20, k=1 / 5 / 10」一键切到 m21 demo 用的设置。在图上左右拖竖线上的圆点, 或者用 Tab 选中后按左右方向键 (每次 0.01), 改真实通过率 p。
      - 应该看到: 只要 k 大于 1, 红色虚线就整条落在绿线下面, 只在 p=0 和 p=1 两端重合; 缝隙在 p 较小时最宽, p 越接近 1 越窄。默认 n=20、k=10、p=0.15 时: 真值 0.8031, 无偏估计的期望 0.8031, 朴素公式的期望 0.7247, 朴素偏差 −0.0784。右侧读数逐项看: 「真值」是用当前 p 直接算的 pass@k; 「无偏估计的期望」和真值相差不到 0.005 时显示绿色, 不管怎么拖都应该是绿的; 「朴素公式的期望」是红色虚线在当前 p 的高度; 「朴素偏差」是朴素期望减去真值, 负数表示算低了, 低于 −0.005 显示红色。k 越大, 缝隙越宽; n 越大, c/n 越准, 缝隙越窄 (n=50、k=10、p=0.15 时偏差缩到 −0.0314)。k=1 时两条线重合, 偏差为 0。"
    module="llm_train/m21"
    run="python -m llm_train.m21_llm_eval.demo"
    :challenge="{
      ask: '把 $k$ 拉到 1, 两条线会怎样? 再把 $n$ 拉到和 $k$ 一样大呢?',
      answer: '$k=1$ 时两条线完全重合, 偏差为 0。$n$ 拉到和 $k$ 一样大时, 缝隙反而更宽, 但右侧「无偏估计的期望」仍然等于真值。\n- $k=1$: pass@1 就是试 1 次就对的概率, 真值是 $p$。朴素公式 $1-(1-c/n)^1$ 化简就是 $c/n$; 无偏公式 $1-\\binom{n-c}{1}/\\binom{n}{1} = 1-(n-c)/n$, 也是 $c/n$。$c/n$ 是 n 个样本里对的比例, 它的平均正好是 $p$, 所以两种算法都没有偏差。m21 实测 $k=1$: 真值 0.1883, 无偏 0.1884, 朴素 0.1884, 差的 0.0001 是随机抽样误差。\n- $k \\gt 1$ 时朴素为什么偏低: $1-(1-x)^k$ 是一条凹曲线, x 小时涨得快, 越往后涨得越慢、越平。c/n 围绕 p 上下波动: 往高处偏时, 曲线只多涨一点; 往低处偏时, 曲线掉得多。一高一低平均下来, 就比 $1-(1-p)^k$ 低。这就是 Jensen 不等式: 对凹函数, 先代值再平均, 不会超过先平均再代值。k 越大曲线越弯, 偏得越多: $n=20$、$p=0.15$ 时, $k=5$ 偏差 −0.0381, $k=10$ 偏差 −0.0784。\n- $n = k$: 只采 k 个样本, c/n 波动更大, 朴素公式偏得更多。$n=k=10$、$p=0.15$ 时朴素期望 0.6484, 偏差 −0.1547, 约是 $n=20$ 时的两倍。无偏公式这时退化成「这 k 个样本里有没有对的」: 有记 1, 没有记 0。它的平均正好是试 k 次至少对一次的概率, 所以期望仍是真值 0.8031; 但每次只能给出 0 或 1, 波动 (方差) 很大。\n所以标准做法是每题采 n 个 (n 比 k 大, 比如 $n=20$ 估 pass@10), 再用无偏公式。两个公式的区别可以这样记: 无偏公式是从这 n 个样本里不放回地挑 k 个、至少一个对的概率; 朴素公式等于有放回地挑 k 次, 相当于假装 c/n 就是真实通过率, 没考虑 c/n 本身带着随机误差。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="每题采样数 n" :min="1" :max="50" />
      <LabSlider v-model="k" label="k (pass@k 里试几次)" :min="1" :max="n" />
      <div class="row">
        <button v-for="kk in [1, 5, 10]" :key="kk" type="button" :class="{ active: k === kk && n === 20 }" @click="(n = 20), (k = kk)">n=20, k={{ kk }}</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="横轴真实通过率 p, 纵轴 pass@k: 绿色实线是真值, 红色虚线是朴素公式的期望; 竖线上的圆点可左右拖动改 p">
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
      <div class="kv"><span>p = {{ p.toFixed(2) }} 时的真值 (知道 p 直接算的 pass@k)</span><b>{{ truth(p).toFixed(4) }}</b></div>
      <div class="kv"><span>无偏估计的期望 (反复重采 n 个样本的平均, 应等于真值)</span><b :class="Math.abs(unbMean(p) - truth(p)) < 0.005 ? 'good' : 'bad'">{{ unbMean(p).toFixed(4) }}</b></div>
      <div class="kv"><span>朴素公式的期望 (把 c/n 当 p 代进去, 再取平均)</span><b :class="bias < -0.005 ? 'bad' : ''">{{ naiveMean(p).toFixed(4) }}</b></div>
      <div class="kv"><span>朴素偏差 (朴素期望减真值, 负数就是算低了)</span><b :class="bias < -0.005 ? 'bad' : 'good'">{{ bias >= 0 ? '+' : '' }}{{ bias.toFixed(4) }}</b></div>
      <div class="lab-note">
        <p>★ 无偏公式在代码里不直接算组合数, 而是把 C(n−c, k) / C(n, k) 约分成 c 项连乘: 1 − ∏(1 − k/i), i 从 n−c+1 取到 n。结果一样, n 很大也不会溢出; 和 m21 demo.py 里的 pass_at_k 是同一种写法。错的样本不到 k 个 (n−c 小于 k) 时, 挑 k 个必然挑到对的, 直接记 1。</p>
        <p>m21 demo 第 [3] 段的数据集级实测, 不随滑杆变化: 300 道题, 每题的真实通过率 p 从 Beta(0.5, 2) 分布里抽 (平均 0.2, 多数题很难), 每题采 n=20 个样本, 整个实验重复 400 次取平均。k=10 时真值 0.5864, 无偏 0.5861, 朴素 0.5470, 偏低 0.0394; k=5 时朴素偏低 0.0203; k=1 时两者相同。</p>
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
