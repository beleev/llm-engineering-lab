<!--
  gradcheck ε 实验台 (对应 llm_basic/gradcheck.py:numeric_grad)。
  只讲一件事: 数值梯度的误差是 U 形的 —— ε 太大截断误差, ε 太小舍入误差。
  曲线是浏览器里真算出来的 (JS number = float64; float32 用 Math.fround 逐步舍入模拟)。
-->
<template>
  <LabFrame
    title="gradcheck 的 ε — 检查梯度时, 为什么步长不是越小越好"
    sub="- 模拟什么: 手写的反向传播写错了不会报错, 只会悄悄把模型训坏, 所以要对答案。gradcheck 的办法: 把一个权重 w 挪一点点 (这一点点就是 $\varepsilon$), 看 loss 变了多少, 用差商 $(f(w+\varepsilon) - f(w-\varepsilon)) / 2\varepsilon$ 估出一个「不可能写错」的数值梯度, 再和手写 backward 算出来的解析梯度比。这里被检查的是一个 3 类 softmax + 交叉熵小网络的一个权重 W[0][0], 做法和 llm_basic/gradcheck.py 一样, 曲线是浏览器里真算出来的。
      - 图上是什么: 横轴是步长 $\varepsilon$ (对数刻度, 左边小右边大), 纵轴是两种梯度的相对误差 |数值 − 解析| ÷ max(|数值|, |解析|), 也是对数刻度, 越低越好。横着的虚线是 gradcheck.py 的通过线 1e-6, 误差落在线下才算通过。竖线是当前选的 $\varepsilon$, 圆点是它对应的误差, 过线与否用颜色区分。底部标出了 gradcheck.py 默认用的 1e-5。
      - 可以做什么: 切「中心差分 / 前向差分」换差商公式: 中心差分往左右各挪一次, 误差按 $\varepsilon^2$ 缩小; 前向差分只往右挪一次, 误差只按 $\varepsilon$ 缩小。切「float64 / float32」换浮点精度, float32 每一步运算都要多舍入一次。按「注入 bug」把解析梯度故意写错 (backward 忘了减 onehot), 看真错误长什么样。拖图上的竖线或下面的滑杆选 $\varepsilon$。
      - 应该看到: 曲线是 U 形。$\varepsilon$ 往右太大, 差商不再逼近真正的斜率, 这叫截断误差; $\varepsilon$ 往左太小, $f(w+\varepsilon)$ 和 $f(w-\varepsilon)$ 几乎相等, 相减时有效数字被浮点舍入吃掉, 这叫舍入误差。默认 float64 + 中心差分时, 谷底在 $\varepsilon \approx$ 1e-5 附近, 误差约 1e-11。右侧「解析梯度」是 backward 算的, 「数值梯度」是差商算的, 「相对误差」是两者的差距, 「最佳 $\varepsilon$ / 最低误差」是这条曲线谷底的位置。注入 bug 后曲线变成一条和 $\varepsilon$ 无关的高台: 真 bug 不会随 $\varepsilon$ 改善, 而 $\varepsilon$ 选错造成的「假失败」会。"
    module="llm_basic/gradcheck.py"
    run="python llm_basic/gradcheck.py"
    :challenge="{
      ask: '先猜: 切到 float32 后, U 形谷底往哪边移、最低误差变成多少量级? 这对 gradcheck 的写法意味着什么?',
      answer: '谷底往右移 ($\\varepsilon$ 要取大一些才最准), 最低误差只有 1e-6 到 1e-5 这个量级, 几乎够不着通过线。\n- 中心差分的总误差大约是 $\\varepsilon^2$ (截断误差, $\\varepsilon$ 越大它越大) 加上 $\\delta/\\varepsilon$ (舍入误差, $\\varepsilon$ 越小它越大)。$\\delta$ 是浮点数的相对精度, 也就是两个相邻的可表示数之间的相对间隔。\n- float64: $\\delta \\approx$ 1e-16, 两项相等的谷底在 $\\varepsilon \\approx \\delta^{1/3} \\approx$ 1e-5, 最低误差约 1e-11, 比通过线低 5 个数量级, 非常从容。\n- float32: $\\delta \\approx$ 6e-8, 谷底右移到 $\\varepsilon \\approx$ 1e-3, 最低误差只有 1e-6 到 1e-5。这时已经分不清「实现正确」和「有个小 bug」。\n所以 gradcheck.py 全程用 float64、$\\varepsilon$ = 1e-5、中心差分, 而不是训练时用的低精度。\n注入 bug 后, 曲线是一条和 $\\varepsilon$ 无关的高台 (误差约 1.6, 就是错的解析梯度和正确的数值梯度之间差的那一截)。真错误的指纹是「换 $\\varepsilon$ 也不会变好」。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: central }" @click="central = true">中心差分 (f(w+ε)−f(w−ε))/2ε</button>
        <button type="button" :class="{ active: !central }" @click="central = false">前向差分 (f(w+ε)−f(w))/ε</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: !f32 }" @click="f32 = false">float64</button>
        <button type="button" :class="{ active: f32 }" @click="f32 = true">float32</button>
        <button type="button" :class="{ active: bug }" :aria-pressed="bug" @click="bug = !bug">{{ bug ? '✓ 已注入 bug: backward 忘了减 onehot' : '注入 bug: backward 忘了减 onehot' }}</button>
      </div>
      <LabSlider v-model="k" label="步长 ε = 10^k" :min="-12" :max="-1" :step="0.25" :format="(t) => '1e' + t.toFixed(2)" />
    </template>

    <svg ref="svg" viewBox="0 0 560 300" role="group" aria-label="相对误差随 ε 变化的曲线, 竖线可拖动选 ε">
      <g class="grid">
        <template v-for="e in [-16, -12, -8, -4, 0]" :key="'y' + e">
          <line :x1="X0" :x2="X1" :y1="py(e)" :y2="py(e)" /><text :x="X0 - 6" :y="py(e) + 4" class="yl">1e{{ e }}</text>
        </template>
        <template v-for="e in [-12, -10, -8, -6, -4, -2]" :key="'x' + e">
          <line :x1="px(e)" :x2="px(e)" :y1="Y0" :y2="Y1" /><text :x="px(e)" :y="Y1 + 14" class="xl">1e{{ e }}</text>
        </template>
      </g>
      <line :x1="X0" :x2="X1" :y1="py(-6)" :y2="py(-6)" class="pass" />
      <text :x="X1 - 4" :y="py(-6) - 4" class="passl">通过线 1e-6</text>
      <polyline :points="pts" class="curve" :class="{ bad: bug }" />
      <text :x="px(-11.6)" :y="Y0 + 12" class="ann">← 舍入误差 ∝ δ/ε</text>
      <text :x="px(-1.2)" :y="Y0 + 12" class="ann end">截断误差 ∝ ε{{ central ? '²' : '' }} →</text>
      <line :x1="px(-5)" :x2="px(-5)" :y1="Y1 - 6" :y2="Y1" class="tick" /><text :x="px(-5)" :y="Y1 + 28" class="xl tickl">gradcheck.py 默认 1e-5</text>
      <g class="draggable" tabindex="0" role="slider" aria-label="拖动选择 ε" :aria-valuenow="k" aria-valuemin="-12" aria-valuemax="-1" :aria-valuetext="`ε = 1e${k.toFixed(2)}`"
        @pointerdown="start($event, { svg, onMove })" @keydown.left="k = clamp(k - 0.25, -12, -1)" @keydown.right="k = clamp(k + 0.25, -12, -1)">
        <line :x1="px(k)" :x2="px(k)" :y1="Y0" :y2="Y1" class="marker" />
        <rect :x="px(k) - 12" :y="Y0" width="24" :height="Y1 - Y0" fill="transparent" />
        <circle :cx="px(k)" :cy="py(Math.log10(cur.err))" r="7" class="dot" :class="cur.err < 1e-6 ? 'ok' : 'no'" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>解析梯度 (手写 backward 算的) {{ bug ? '(带 bug)' : '' }}</span><b>{{ ana.toFixed(8) }}</b></div>
      <div class="kv"><span><Tex text="数值梯度 (差商算的) @ $\varepsilon$" />={{ eps.toExponential(1) }}</span><b>{{ cur.num.toFixed(8) }}</b></div>
      <div class="kv"><span>相对误差 (通过线 1e-6)</span><b :class="cur.err < 1e-6 ? 'good' : 'bad'">{{ cur.err.toExponential(1) }}</b></div>
      <div class="kv"><span><Tex text="这条曲线的最佳 $\varepsilon$ / 谷底的最低误差" /></span><b>{{ best.eps.toExponential(0) }} / {{ best.err.toExponential(0) }}</b></div>
      <p class="lab-note">
        <Tex :text="bug ? '有 bug 时误差是一条约等于常数的高台: 无论 $\\varepsilon$ 取多少都过不了线, 因为解析梯度本身就是错的, 和用什么步长去量它没关系。$\\varepsilon$ 选得不好造成的「假失败」会随 $\\varepsilon$ 改善, 真 bug 不会。'
          : cur.err < 1e-6 ? '通过: 手写 backward 和差商对上了。注意谷底左右两侧斜率不同: 左边 $\\varepsilon$ 每缩小 10 倍, 误差涨 10 倍 (舍入误差 $\\delta/\\varepsilon$); 右边按 $\\varepsilon$ 的幂次下降 (截断误差)。'
          : '没过线, 但解析梯度是对的, 只是 $\\varepsilon$ 选错了: 要么太大 (截断误差), 要么太小 (舍入误差)。这就是 gradcheck 要固定 float64 + 1e-5 的原因, 否则分不清是步长选错还是代码写错。'" />
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const central = ref(true), f32 = ref(false), bug = ref(false), k = ref(-5)
const svg = ref(null)
const { start } = useDrag()

// 被检查的函数: 3 类 softmax + CE, 只扰动 W[0][0]。r = 每步运算后的舍入 (float32 用 fround)
const Wm = [[0.6, -0.4], [-0.3, 0.8], [0.2, 0.1]], xin = [1.3, -0.7], Y = 0
const loss = (w00, r) => {
  const z = Wm.map((row, i) => r(r((i === 0 ? w00 : row[0]) * xin[0]) + r(row[1] * xin[1])))
  const m = Math.max(...z)
  const lse = r(m + r(Math.log(r(z.reduce((s, t) => r(s + r(Math.exp(r(t - m)))), 0)))))
  return r(lse - z[Y])
}
const p0 = (() => { const z = Wm.map((r) => r[0] * xin[0] + r[1] * xin[1]); const e = z.map(Math.exp); return e[0] / (e[0] + e[1] + e[2]) })()
// ★ 正确: (p0 − 1)·x0; bug 版忘了减 onehot
const ana = computed(() => (bug.value ? p0 : p0 - 1) * xin[0])

const measure = (kk) => {
  const r = f32.value ? Math.fround : (t) => t
  const e = r(10 ** kk), w = r(Wm[0][0])
  const num = central.value ? (loss(r(w + e), r) - loss(r(w - e), r)) / (2 * e) : (loss(r(w + e), r) - loss(w, r)) / e
  // ★ 分母与 gradcheck.py:check_op 相同: max(|g_a|, |g_n|, 1e-12)
  const err = Math.abs(num - ana.value) / Math.max(Math.abs(num), Math.abs(ana.value), 1e-12)
  return { num, err: Math.max(err, 1e-16) }   // 恰好为 0 时压到图的地板上
}
const ks = range(45).map((i) => -12 + i * 0.25)
const curve = computed(() => ks.map((kk) => ({ k: kk, ...measure(kk) })))
const cur = computed(() => measure(k.value))
const eps = computed(() => 10 ** k.value)
const best = computed(() => { const b = curve.value.reduce((a, c) => (c.err < a.err ? c : a)); return { eps: 10 ** b.k, err: b.err } })

const X0 = 52, X1 = 548, Y0 = 14, Y1 = 262
const px = (kk) => X0 + ((kk + 12) / 11) * (X1 - X0)
const py = (le) => Y1 - ((clamp(le, -16, 1) + 16) / 17) * (Y1 - Y0)
const pts = computed(() => curve.value.map((c) => `${px(c.k).toFixed(1)},${py(Math.log10(c.err)).toFixed(1)}`).join(' '))
const onMove = ({ x }) => { k.value = clamp(Math.round((((x - X0) / (X1 - X0)) * 11 - 12) * 4) / 4, -12, -1) }
</script>

<style scoped>
.grid line { stroke: var(--border); stroke-width: 1; }
.yl { text-anchor: end; font-size: 10px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 10px; fill: var(--text-dim); }
.tick { stroke: var(--eye); stroke-width: 2; }
.tickl { fill: var(--eye); }
.pass { stroke: var(--left); stroke-dasharray: 5 4; }
.passl { text-anchor: end; font-size: 10px; fill: var(--left); }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.curve.bad { stroke: var(--danger); }
.ann { font-size: 10px; fill: var(--text-muted); }
.ann.end { text-anchor: end; }
.marker { stroke: var(--text-muted); stroke-dasharray: 3 3; }
.dot { stroke: var(--bg); stroke-width: 2; }
.dot.ok { fill: var(--left); }
.dot.no { fill: var(--danger); }
</style>
