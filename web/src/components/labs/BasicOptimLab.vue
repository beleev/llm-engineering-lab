<!--
  优化器轨迹实验台 (对应 llm_basic/optim.py:adam_step)。
  病态二次损失 f = ½(x² + κ·y²): y 方向陡, x 方向平。同一个 lr 下看 SGD / 动量 / Adam 怎么走。
-->
<template>
  <LabFrame
    title="优化器轨迹 — 同一个学习率, SGD / 动量 / Adam 三种走法"
    sub="- 模拟什么: 模型训练是在损失曲面上沿负梯度方向迭代下降的过程, 单步更新幅度由学习率 lr 控制。本实验台采用二元二次目标函数 $f = \tfrac{1}{2}(x^2 + \kappa y^2)$ 模拟病态曲率地形: $y$ 方向的曲率是 $x$ 方向的 $\kappa$ 倍, 对应真实 Transformer 中不同参数维度梯度量级相差数个数量级的现象。三种优化器分别对应 llm_basic/optim.py 中的 adam_step 及其两个简化变体。
- 图上是什么: 同心椭圆表示损失等高线, 越靠近中心损失越小, 全局极小值位于中心 (0, 0); 椭圆呈扁长形态源于 $y$ 方向曲率更大。可拖拽的大圆点表示初始参数位置, 三条折线分别对应三种优化器从同一起点迭代 80 步的参数轨迹, 颜色与图下方图例一一对应:
  - SGD: 每步仅沿当前瞬时梯度方向更新。
  - 动量: 对历史梯度做指数累加作为更新速度, 方向一致的分量持续加速。
  - Adam: 按各坐标自身历史梯度二阶矩对步长做逐坐标自适应归一化。
- 可以做什么: 可调节三组滑杆、初始位置与轨迹播放条:
  - 拖动「学习率 lr」调节单步更新步长。
  - 拖动「动量 β」调节动量法与 Adam 对历史梯度的衰减记忆系数 (取 0 时不保留历史记忆)。
  - 拖动「条件数 κ」改变两个坐标轴之间的曲率比值。
  - 拖动图上的起点改变迭代初始坐标。
  - 使用底部播放条逐步回放优化轨迹 (点击「下一步」或拖动「时间轴」), 观察各步迭代落点。
- 应该看到: 右侧三项「当前 loss」实时显示当前回放步各优化器所在位置的损失值, 若显示「发散」则代表迭代已越出稳定区域而数值爆炸。「SGD 稳定上限 $2/\kappa$」与「动量稳定上限」给出触发发散的理论学习率阈值, 当前 lr 超出阈值时对应行会高亮提示。默认学习率 lr 为 0.04、$\kappa$ 为 20 时:
  - SGD: 在高曲率的 $y$ 方向剧烈震荡, 同时在低曲率的 $x$ 方向推进迟缓。
  - 动量: 收敛更快, 但越过谷底后会产生超调并折返。
  - Adam: 前几步在两个坐标轴上的更新步幅基本一致, 不受局部曲率差异制约, 直接朝极小值点收敛。"
    module="llm_basic/optim.py"
    run="python llm_basic/optim.py"
    :challenge="{
      ask: '将条件数 κ 拖至 40, 学习率 lr 拖至 0.06。请先判断: 三个优化器中哪一个最先发散? 随后将 lr 调回 0.04, 哪一个在平缓的 x 方向收敛最慢?',
      answer: 'SGD 最先发散; 将 lr 调回 0.04 后, 在 $x$ 方向收敛最慢的同样是 SGD。\n- SGD: 在 $y$ 方向每步将坐标乘以 $(1 - \\mathrm{lr}\\cdot\\kappa)$。当 $\\mathrm{lr} &gt; 2/\\kappa = 0.05$ 时, 该缩放因子的绝对值大于 1 ($\\mathrm{lr} = 0.06$ 时为 $1 - 2.4 = -1.4$), 每步迭代不仅无法靠近极小值, 反而跨过谷底跳向更远的另一侧, 振幅逐轮放大并导致发散。\n- 动量: 理论稳定上限为 $2(1+\\beta)/\\kappa$, 当 $\\beta = 0.8$ 时上限为 0.09, 显著高于 SGD 的 0.05, 因此在 $\\mathrm{lr} = 0.06$ 下仍能保持稳定。\n- Adam: 单步更新量约为 $\\mathrm{lr}\\cdot\\hat{m}/\\sqrt{\\hat{v}}$, 其中 $\\hat{m}$ 为梯度一阶矩的指数滑动平均, $\\sqrt{\\hat{v}}$ 为梯度二阶矩平方根的滑动平均; 两者相除后量级约为 1, 因此单步位移幅度约为 lr, 与局部曲率大小无关。它不会因曲率过大而发散, 仅会在极小值附近以约 lr 的幅度轻微震荡。\n将 lr 调回 0.04 后, SGD 在 $x$ 方向每步仅将坐标乘以 $1 - \\mathrm{lr} = 0.96$, 迭代 80 步仍无法到达谷底。在病态曲率地形上, 全局学习率上限受最陡方向制约 (继续增大即触发发散), 导致最平缓方向的收敛极其缓慢。\nAdam 利用 $\\sqrt{\\hat{v}}$ 对各坐标独立做步长归一化, 等效于为每个参数分配自适应有效学习率: 高曲率方向自动缩小步长, 低曲率方向自动保持较大步长。',
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
          <li>在 <Tex text="$y$ 方向每步乘以 $(1 - \mathrm{lr}\cdot\kappa)$" /> = {{ (1 - lr * kappa).toFixed(2) }}。绝对值超过 1 就发散 (每步跳到更远的另一侧)。</li>
          <li>在 <Tex text="$x$ 方向每步乘以 $(1 - \mathrm{lr})$" /> = {{ (1 - lr).toFixed(2) }}。接近 1 就几乎不动。</li>
        </ul>
        <p>单一全局学习率 lr 无法同时兼顾曲率悬殊的两个方向。</p>
        <p>Adam 将各坐标的更新步长除以该坐标历史梯度幅值的滑动平均, 使高曲率与低曲率方向的单步有效位移均稳定在 lr 量级, 从而在单一全局 lr 下同步推进两个方向的收敛。</p>
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
