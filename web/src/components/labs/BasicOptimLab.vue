<!--
  优化器轨迹实验台 (对应 llm_basic/optim.py:adam_step)。
  病态二次损失 f = ½(x² + κ·y²): y 方向陡, x 方向平。同一个 lr 下看 SGD / 动量 / Adam 怎么走。
-->
<template>
  <LabFrame
    title="优化器轨迹 — 同一个学习率, SGD / 动量 / Adam 三种走法"
    sub="- 模拟什么: 训练就是在 loss 的地形上往低处走: 每一步算出梯度 (当前位置最陡的上坡方向), 朝反方向挪一点, 挪多远由学习率 lr 决定。这里的地形是一个只有两个参数的玩具 loss $f = \tfrac{1}{2}(x^2 + \kappa y^2)$: $y$ 方向比 $x$ 方向陡 $\kappa$ 倍, 这种「一个方向陡、一个方向平」的地形叫病态。真实 Transformer 里各个参数的梯度量级相差好几个数量级, 就是它的放大版。三个优化器对应 llm_basic/optim.py 里的 adam_step 和它的两个简化版。
      - 图上是什么: 一圈圈椭圆是等高线 (loss 相同的点连成的线), 越靠中心 loss 越小, 中心 (0, 0) 是最低点; 椭圆又扁又长就是因为 $y$ 方向陡。大圆点是起点, 可以拖。三条折线是三个优化器从同一个起点出发各走 80 步的轨迹, 颜色对照图下方的图例:
        - SGD: 每步只用当前梯度。
        - 动量: 把历史梯度累加起来当速度, 同一个方向走得越久越快。
        - Adam: 给每个坐标单独按它自己梯度的大小归一化步长。
      - 可以做什么: 三根滑杆、起点和播放条:
        - 拖「学习率 lr」改每步迈多远。
        - 拖「动量 β」改动量和 Adam 对历史梯度的记忆程度 (0 就是不记)。
        - 拖「条件数 κ」改两个方向的陡峭差距。
        - 拖图上的起点换出发位置。
        - 下面的播放条一步一步放轨迹 (「下一步」或拖「时间轴」), 看每一步各走到哪。
      - 应该看到: 右侧三个「当前 loss」是播放到这一步时各优化器所在位置的 loss, 显示「发散」说明它已经飞出去了。「SGD 稳定上限 $2/\kappa$」和「动量稳定上限」是 lr 超过多少就会发散的理论值, 当前 lr 超过它时这一行会被标出来。默认 lr = 0.04、$\kappa$ = 20 时:
        - SGD: 在陡的 $y$ 方向来回跳、在平的 $x$ 方向慢慢蹭。
        - 动量: 更快但会冲过头再折回来。
        - Adam: 头几步两个坐标迈的步子几乎一样大, 与坡度无关, 直奔谷底。"
    module="llm_basic/optim.py"
    run="python llm_basic/optim.py"
    :challenge="{
      ask: '把 κ 拖到 40, lr 拖到 0.06。先猜: 三个优化器谁先发散? 再把 lr 调回 0.04, 谁在平缓的 x 方向走得最慢?',
      answer: 'SGD 先发散; lr 调回 0.04 后, 也是 SGD 在 $x$ 方向最慢。\n- SGD: 在 $y$ 方向每步把坐标乘 $(1 - \\mathrm{lr}\\cdot\\kappa)$。$\\mathrm{lr} > 2/\\kappa = 0.05$ 时这个因子的绝对值大于 1 (lr = 0.06 时是 1 − 2.4 = −1.4), 每步不但没靠近谷底, 反而跳到更远的另一侧, 越震越大, 这叫发散。\n- 动量: 稳定上限是 $2(1+\\beta)/\\kappa$, $\\beta$ = 0.8 时是 0.09, 比 SGD 的 0.05 宽得多, lr = 0.06 还稳得住。\n- Adam: 每步位移约等于 $\\mathrm{lr}\\cdot\\hat{m}/\\sqrt{\\hat{v}}$, $\\hat{m}$ 是梯度的滑动平均, $\\sqrt{\\hat{v}}$ 是梯度大小的滑动平均, 两者一除大小约为 1, 所以每步大约走 lr 这么远, 和坡有多陡无关。它不会发散, 只会在谷底附近以 lr 为幅度来回抖。\nlr 调回 0.04 后, SGD 在 $x$ 方向每步只把坐标乘 $(1 - \\mathrm{lr})$ = 0.96, 80 步都走不到谷底。在病态地形上, lr 被最陡的方向卡死 (再大就发散), 最平的方向就学不动。\nAdam 用 $\\sqrt{\\hat{v}}$ 给每个坐标单独归一化, 等于每个参数有自己的学习率: 陡的方向自动迈小步, 平的方向自动迈大步。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lr" label="学习率 lr (每步迈多远)" :min="0.01" :max="0.4" :step="0.01" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="beta" label="动量 β (对历史梯度的记忆, 也是 Adam 的 β₁)" :min="0" :max="0.99" :step="0.01" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="kappa" label="条件数 κ (y 方向比 x 方向陡几倍)" :min="1" :max="50" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value} 步`" />
    </template>

    <svg ref="svg" viewBox="-3.2 -2.1 6.4 4.2" role="group" aria-label="损失等高线与优化轨迹">
      <ellipse v-for="c in LEVELS" :key="c" cx="0" cy="0" :rx="Math.sqrt(2 * c)" :ry="Math.sqrt(2 * c / kappa)" class="contour" />
      <line x1="-3.2" x2="3.2" y1="0" y2="0" class="axis" /><line y1="-2.1" y2="2.1" x1="0" x2="0" class="axis" />
      <g v-for="o in OPTS" :key="o.id">
        <polyline :points="poly(o.id)" class="path" :style="{ stroke: o.color }" />
        <circle :cx="at(o.id)[0]" :cy="-at(o.id)[1]" r="0.07" :style="{ fill: o.color }" />
      </g>
      <circle
        :cx="x0" :cy="-y0" r="0.13" class="start draggable" tabindex="0" role="slider" aria-label="拖动起点"
        :aria-valuetext="`${x0.toFixed(1)}, ${y0.toFixed(1)}`"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.left="x0 = clamp(x0 - 0.1, -3, 3)" @keydown.right="x0 = clamp(x0 + 0.1, -3, 3)"
        @keydown.up.prevent="y0 = clamp(y0 + 0.1, -2, 2)" @keydown.down.prevent="y0 = clamp(y0 - 0.1, -2, 2)"
      />
    </svg>
    <div class="legend">
      <span v-for="o in OPTS" :key="o.id"><i :style="{ background: o.color }" />{{ o.label }}</span>
      <span><i class="s" />起点 (可拖)</span>
    </div>

    <template #stats>
      <div v-for="o in OPTS" :key="o.id" class="kv">
        <span>{{ o.label }} 走到这一步时的 loss</span>
        <b :class="lossAt(o.id) < 1e-3 ? 'good' : lossAt(o.id) > f0 ? 'bad' : ''">{{ fmt(lossAt(o.id)) }}</b>
      </div>
      <div class="kv"><span><Tex text="SGD 不发散的 lr 上限 $2/\kappa$" /></span><b :class="lr < 2 / kappa ? 'good' : 'bad'">{{ (2 / kappa).toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="动量不发散的 lr 上限 $2(1+\beta)/\kappa$" /></span><b :class="lr < 2 * (1 + beta) / kappa ? 'good' : 'bad'">{{ (2 * (1 + beta) / kappa).toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>
          SGD 每一步把坐标:
        </p>
        <ul class="pts">
          <li>在 <Tex text="$y$ 方向乘 $(1 - \mathrm{lr}\cdot\kappa)$" /> = {{ (1 - lr * kappa).toFixed(2) }}。绝对值超过 1 就发散 (每步跳到更远的另一侧)。</li>
          <li>在 <Tex text="$x$ 方向乘 $(1 - \mathrm{lr})$" /> = {{ (1 - lr).toFixed(2) }}。接近 1 就几乎不动。</li>
        </ul>
        <p>
          一个 lr 没法同时伺候两个方向。
        </p>
        <p>
          Adam 把每个坐标的步长除以这个坐标自己梯度大小的滑动平均, 陡的方向和平的方向都变成「每步大约走 lr」, 所以同一个 lr 两个方向都能走。
        </p>
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
import { clamp } from '@/utils/labmath.js'

const N = 80, LEVELS = [0.05, 0.25, 0.75, 1.5, 3, 5]
const OPTS = [
  { id: 'sgd', label: 'SGD', color: 'var(--danger)' },
  { id: 'mom', label: '动量', color: 'var(--eye)' },
  { id: 'adam', label: 'Adam', color: 'var(--left)' },
]
const lr = ref(0.04), beta = ref(0.8), kappa = ref(20), x0 = ref(-2.6), y0 = ref(1.4)
const svg = ref(null)
const { start } = useDrag()
const onMove = ({ x, y }) => { x0.value = clamp(x, -3, 3); y0.value = clamp(-y, -2, 2) }

const f = ([x, y]) => 0.5 * (x * x + kappa.value * y * y)
const grad = ([x, y]) => [x, kappa.value * y]

const paths = computed(() => {
  const a = lr.value, b1 = beta.value, b2 = 0.999, eps = 1e-8
  const out = { sgd: [], mom: [], adam: [] }
  let w = [x0.value, y0.value]
  for (let t = 0; t <= N; t++) { out.sgd.push(w); const g = grad(w); w = [w[0] - a * g[0], w[1] - a * g[1]] }
  w = [x0.value, y0.value]; let v = [0, 0]
  for (let t = 0; t <= N; t++) { out.mom.push(w); const g = grad(w); v = v.map((vi, i) => b1 * vi + g[i]); w = w.map((wi, i) => wi - a * v[i]) }
  w = [x0.value, y0.value]; let m = [0, 0], s = [0, 0]
  for (let t = 1; t <= N + 1; t++) {
    out.adam.push(w); const g = grad(w)
    m = m.map((mi, i) => b1 * mi + (1 - b1) * g[i]); s = s.map((si, i) => b2 * si + (1 - b2) * g[i] * g[i])
    // ★ 偏差修正后按坐标归一化: 步长 ≈ lr, 与该方向的坡度无关
    w = w.map((wi, i) => wi - a * (m[i] / (1 - b1 ** t)) / (Math.sqrt(s[i] / (1 - b2 ** t)) + eps))
  }
  return out
})

const stepper = useStepper(() => N + 1, { interval: 120 })
watch([lr, beta, kappa, x0, y0], () => { stepper.pause(); stepper.step.value = N })
stepper.step.value = N
const at = (id) => paths.value[id][stepper.step.value] || paths.value[id][N]
// 发散时坐标会到 1e30, 画之前夹住, 否则 SVG 渲染异常
const poly = (id) => paths.value[id].slice(0, stepper.step.value + 1).map(([x, y]) => `${clamp(x, -50, 50).toFixed(3)},${clamp(-y, -50, 50).toFixed(3)}`).join(' ')
const lossAt = (id) => f(at(id))
const f0 = computed(() => f([x0.value, y0.value]))
const fmt = (t) => (t > 1e4 ? '发散' : t < 1e-3 ? t.toExponential(1) : t.toFixed(3))
</script>

<style scoped>
svg { background: var(--bg-elev); border-radius: var(--radius-sm); width: 100%; }
.contour { fill: none; stroke: var(--border-strong); stroke-width: 0.015; }
.axis { stroke: var(--border); stroke-width: 0.01; }
.path { fill: none; stroke-width: 0.03; stroke-linejoin: round; }
.start { fill: var(--accent); stroke: var(--text); stroke-width: 0.03; }
.start:focus-visible { outline: none; stroke-width: 0.07; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; font-size: 12px; color: var(--text-muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
.legend i.s { background: var(--accent); }
</style>
