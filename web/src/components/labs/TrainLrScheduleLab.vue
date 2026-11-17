<!--
  学习率调度 (对应 llm_train/m10_training_stability:warmup_cosine_lr / wsd_lr)。
  只讲一件事: cosine 的每一步 LR 都依赖总步数, 想加训就得从头来; WSD 的稳定段与总步数无关, 任何稳定段的 checkpoint 都能接着训。
-->
<template>
  <LabFrame
    title="学习率调度 Cosine vs WSD — 训到一半想加训, 要回退多少步重训?"
    sub="- 模拟什么: 学习率 (lr) 是每一步参数更新的步长, 训练中它不是常数, 而是按一条调度曲线变: 开头从小爬到峰值 (叫 warmup, 因为 Adam 的二阶矩估计在最初几百步很不准, 一上来就迈大步容易走飞), 结尾再降下来让模型落进谷底。这个台子模拟「原计划训 $T_0$ 步, 训完想加到 $T_1$ 步」这件事: 已经训过的步里, 有多少步用的 lr 和「一开始就按 $T_1$ 规划」的一样, 这些步算数; 不一样的那些要回退重训。两种常见曲线:
        - cosine: 把 warmup 之后每一步的 lr 都写成「当前步数 ÷ 总步数」的余弦函数, 所以总步数一改整条曲线都变。
        - WSD (warmup–stable–decay): 中间一长段保持峰值不动, 只在最后约 10% 做退火 (decay), 稳定段的公式里根本没有总步数。
      - 图上是什么: 横轴是训练步数 (0 到 2000), 纵轴是 lr 相对峰值的比例。
        - 实线是按原计划 $T_0$ 走的 lr 曲线。
        - 虚线是假如一开始就知道要训 $T_1$ 步、本该走的曲线。
        - 红色阴影是已训过的步里实线和虚线不一致的部分, 阴影越宽浪费越多。
        - 横轴下方三个彩色方块是可拖的手柄: warmup 结束点、原计划 $T_0$、新计划 $T_1$。
      - 可以做什么: 控件从上到下:
        - 点「Warmup + Cosine」或「WSD」切换调度。
        - 左右拖三个手柄 (或聚焦后用方向键, 一次 10 步) 改 warmup 长度、原计划和新计划的总步数。
        - WSD 下拖「退火占比」改最后退火段占总步数的比例。
        - 拖「最终 LR / 峰值 LR」改退火的终点。
      - 应该看到: cosine 下 warmup 一结束实线和虚线就分开, 阴影覆盖几乎整个已训区间; WSD 下两条线在稳定段完全重合, 只有最后退火段有阴影。右侧读数:
        - 「仍算数的」: 已训 $T_0$ 步里 lr 和新计划一致的步数 (从头数, 第一次不一致就停)。
        - 「需要回退重训」: $T_0$ 减去它, 绿色表示一半以上能保留。
        - 最后一行: 对比第 $T_0$ 步实际用的 lr 和按新计划本该用的 lr。"
    module="llm_train/m10"
    run="python -m llm_train.m10_training_stability.demo"
    :challenge="{
      ask: '原计划 1000 步, 训完发现 loss 还在降, 想加到 1600 步。cosine 和 WSD 各要回退多少步重训?',
      answer: '答案:\n- cosine 要回退 900 步, 几乎从头来。\n- WSD 只回退 100 步。\n为什么:\n- cosine: warmup 之后每一步的 lr 都是 $f((\\text{step} - \\text{warmup}) / (\\text{总步数} - \\text{warmup}))$, 总步数从 1000 改成 1600, 分母变了, 同一个 step 算出的 lr 全都不一样 (m10 实测: 总步数 100 改 200, 前 90 步的 lr 最大差 5.96e-4)。已经训过的 1000 步里只有前 100 步的 warmup 还「算数」, 读数显示回退 900 步。不想重训的话只剩一条路: 从第 1000 步的 checkpoint 硬把 lr 重新拉高 (re-warmup), 但 lr 突然变大 loss 会先反弹一截。\n- WSD: 稳定段的 lr 就是峰值, 公式里没有总步数, 所以不管计划训多久, 前 900 步 (退火开始之前) 的 lr 一模一样。浪费的只有原来那 100 步退火, 读数显示回退 100 步。加训的做法:\n  1. 回到退火刚开始时的 checkpoint (第 900 步)。\n  2. 继续保持峰值 lr 训到 1440 步。\n  3. 再做最后 10% (160 步) 的退火就行。\n这也是 MiniCPM / DeepSeek-V3 / Kimi 用 WSD 的原因: 一条主干一直用峰值 lr 往前训, 随时可以从任意 checkpoint 分叉出一段短退火来拿「成品」模型, 数据追加、规模实验都不用重训。退火起点前的那个 checkpoint 是所有分支的公共祖先。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'cosine' }" @click="mode = 'cosine'">Warmup + Cosine</button>
        <button type="button" :class="{ active: mode === 'wsd' }" @click="mode = 'wsd'">WSD (Warmup-Stable-Decay)</button>
      </div>
      <LabSlider v-if="mode === 'wsd'" v-model="decayFrac" label="WSD 退火占比" :min="0.05" :max="0.4" :step="0.05" :format="(v) => Math.round(v * 100) + '%'" />
      <LabSlider v-model="minRatio" label="最终 LR / 峰值 LR" :min="0" :max="0.3" :step="0.05" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="学习率随训练步数的变化, 三个手柄可拖动">
      <line :x1="X0" :x2="W - 10" :y1="py(0)" :y2="py(0)" class="axis" />
      <g v-for="t in [0, 500, 1000, 1500, 2000]" :key="t">
        <line :x1="px(t)" :x2="px(t)" :y1="py(0)" :y2="py(0) + 4" class="axis" />
        <text :x="px(t)" :y="py(0) + 15" class="tick" text-anchor="middle">{{ t }}</text>
      </g>
      <text :x="X0 - 6" :y="py(1) + 3" class="tick" text-anchor="end">峰值</text>
      <line :x1="X0" :x2="W - 10" :y1="py(1)" :y2="py(1)" class="gridl" />
      <polygon :points="gapPoly" class="gap" />
      <polyline :points="poly(t1)" class="ideal" />
      <polyline :points="poly(t0)" class="plan" />
      <g v-for="h in handles" :key="h.key" class="draggable" tabindex="0" role="slider" :aria-label="h.label" :aria-valuenow="h.v"
        @pointerdown="start($event, { svg, onMove: ({ x }) => h.set(Math.round(pxInv(x) / 10) * 10) })"
        @keydown.right.prevent="h.set(h.v + 10)" @keydown.left.prevent="h.set(h.v - 10)">
        <line :x1="px(h.v)" :x2="px(h.v)" :y1="py(1) - 8" :y2="py(0)" :stroke="h.color" stroke-dasharray="2 3" />
        <rect :x="px(h.v) - 9" :y="py(0) + 20" width="18" height="18" rx="4" :fill="h.color" />
        <text :x="px(h.v)" :y="py(0) + 52" class="hl" text-anchor="middle" :fill="h.color">{{ h.short }} {{ h.v }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>已训 {{ t0 }} 步里仍「算数」的 (lr 和新计划一致的步数)</span><b :class="keepFrac > 0.5 ? 'good' : 'bad'">{{ keep }} 步</b></div>
      <div class="kv"><span>加训到 {{ t1 }} 步需要回退重训的</span><b :class="keepFrac > 0.5 ? 'good' : 'bad'">{{ t0 - keep }} 步</b></div>
      <div class="kv"><span>第 {{ t0 }} 步的 lr (相对峰值): 实际用的 vs 按新计划本该用的</span><b>{{ lr(t0 - 1, t0).toFixed(2) }} vs {{ lr(t0 - 1, t1).toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>「仍算数」怎么数: 从第 0 步起逐步比较实线和虚线的 lr, 第一次不一致就停, 因为从那一步起训练轨迹就和新计划分道了, 后面的步再怎么对也接不上。</p>
        <p>cosine 在 warmup 之后每一步都把 total_steps 放在分母里, 总步数一改整条曲线都变。★ WSD 的稳定段 <code class="inline">return base_lr</code> 里没有 total_steps, 只有退火段依赖它。</p>
        <p>warmup 的意义两者相同: Adam 的二阶矩估计在最初几百步很不准, 先用小 lr 走过最危险的一段。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

// 默认 cosine: 先看到「加训要回退 900 步」, 再切 WSD 看它怎么治
const mode = ref('cosine'), warm = ref(100), t0 = ref(1000), t1 = ref(1600), decayFrac = ref(0.1), minRatio = ref(0.1), svg = ref(null)
const { start } = useDrag()
const XMAX = 2000

// 与 Python 逐行相同 (base_lr = 1)
const lr = (step, total) => {
  const w = warm.value, mr = minRatio.value
  if (step < w) return (step + 1) / w
  if (mode.value === 'cosine') {
    const p = (step - w) / Math.max(1, total - w)
    return mr + (1 - mr) * 0.5 * (1 + Math.cos(Math.PI * p))
  }
  const decayStart = total - Math.floor(total * decayFrac.value)
  if (step < decayStart) return 1 // ★ 稳定段: 与 total 无关
  return 1 - (1 - mr) * ((step - decayStart) / Math.max(1, total - decayStart))
}

// 已训过的 [0, T0) 里, 有多少步的 LR 与 "按 T1 规划" 的一致 (从头数, 第一次不一致就停)
const keep = computed(() => {
  let s = 0
  while (s < t0.value && Math.abs(lr(s, t0.value) - lr(s, t1.value)) < 1e-9) s++
  return s
})
const keepFrac = computed(() => keep.value / t0.value)

const W = 620, H = 250, X0 = 40
const px = (s) => X0 + (s / XMAX) * (W - X0 - 14)
const pxInv = (x) => ((x - X0) / (W - X0 - 14)) * XMAX
const py = (v) => 22 + (1 - v) * (H - 90)
const pts = (total) => range(Math.floor(total / 5) + 1).map((i) => Math.min(i * 5, total - 1)).map((s) => [px(s), py(lr(s, total))])
const poly = (total) => pts(total).map((p) => p.join(',')).join(' ')
const gapPoly = computed(() => {
  const ss = range(Math.floor(t0.value / 5) + 1).map((i) => Math.min(i * 5, t0.value - 1))
  const top = ss.map((s) => `${px(s)},${py(lr(s, t1.value))}`)
  const bot = ss.map((s) => `${px(s)},${py(lr(s, t0.value))}`).reverse()
  return [...top, ...bot].join(' ')
})

const handles = computed(() => [
  { key: 'w', short: 'warmup', label: 'warmup 结束步', v: warm.value, color: 'var(--eye)', set: (v) => (warm.value = clamp(v, 10, t0.value - 100)) },
  { key: 't0', short: 'T₀', label: '原计划总步数', v: t0.value, color: 'var(--accent)', set: (v) => (t0.value = clamp(v, warm.value + 100, t1.value)) },
  { key: 't1', short: 'T₁', label: '新计划总步数', v: t1.value, color: 'var(--left)', set: (v) => (t1.value = clamp(v, t0.value, XMAX)) },
])
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.gridl { stroke: var(--border); stroke-dasharray: 2 4; }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.plan { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.ideal { fill: none; stroke: var(--left); stroke-width: 2; stroke-dasharray: 6 4; }
.gap { fill: var(--danger); opacity: 0.25; }
.hl { font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
</style>
