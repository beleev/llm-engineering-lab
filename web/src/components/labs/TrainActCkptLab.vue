<!--
  激活重算 (对应 llm_train/m07_activation_checkpointing:run)。
  只讲一件事: 峰值 ≈ L/k (常驻的段边界) + k (当前段重算出来的瞬时激活), k≈√L 时最小。
-->
<template>
  <LabFrame
    title="激活重算 — 只存段边界、反向时重跑, 段长 k 怎么选?"
    sub="- 模拟什么: 反向传播算第 i 层的梯度时, 需要第 i 层的输入激活 (前向时算出来的中间结果)。最省事的做法是前向时把 L 层的输入全部存下来, 显存要 L+1 份 (本例把每层的激活算一份)。激活重算 (activation checkpointing) 只在每 k 层的段边界存一份, 反向走到某一段时再从这段的入口重跑一遍前向, 把段内的激活临时算出来, 用完就丢。峰值显存约等于 L/k (常驻的段边界) 加 k (当前段临时算出来的), k 取 √L 附近最小; 代价是多跑约一次前向。梯度和不重算时逐位相同, 换的只是「存」和「算」。
      - 图上是什么: 上面一张曲线图, 下面一排格子:
        - 上图横轴是段长 k, 纵轴是峰值激活份数 (整个过程中同一时刻显存里最多有几份激活)。
        - 曲线是每个 k 对应的峰值, 绿色竖虚线标在 √L 的位置, 黄色圆点是当前的 k。
        - 下面一排格子是 a0 到 aL 共 L+1 个激活 (a_i 是第 i 层的输入, aL 是最后一层的输出)。
        - 格子颜色对照图例: 深色是前向时存下的段边界, 亮色是反向时临时重算出来的, 灰色是此刻不在显存里; 带描边的是当前正在处理的位置。
      - 可以做什么: 控件从上到下:
        - 拖「层数 L」和「段长 k」改配置 (k=1 是每层都存, k=L 是只存输入); 也可以在上图直接左右拖黄色圆点改 k。
        - 用播放器一帧一帧看一次前向加反向里每个激活的生死: 前向时段边界依次点亮, 反向时每段先重算 (亮色格子一个个冒出来) 再逐层释放。
        - 换参数后播放器会自动停在最挤的那一帧。
      - 应该看到: k=1 和 k=L 的峰值一样高, 都是 L+1 份: 全存当然高, 只存输入时反向要把整段 L 层一次性重算出来, 也一样高。右侧读数:
        - 「此刻活着的激活」: 当前帧显存里的份数。
        - 「峰值」: 整个过程中的最大值, 等于最优时变绿、接近全存时变红。
        - 「最优 k」: 曲线的最低点。
        - 最后一行: 前向层调用次数 (不重算是 L 次, 重算后多出 L − L/k 次) 和由此多出的计算量百分比 (按前向:反向 = 1:2 折算, 上限 +33%)。"
    module="llm_train/m07"
    run="python -m llm_train.m07_activation_checkpointing.demo"
    :challenge="{
      ask: 'L = 36 层。k=1 (全存)、k=6、k=36 (只存输入) 三种, 峰值激活各是多少份? 哪个最省?',
      answer: 'k=1 和 k=36 都是 37 份, k=6 只要 12 份, 最省。\n- k=1 (全存): 前向把 a0 到 a36 全部存下来, 37 份, 反向不用重算。这是基线。\n- k=36 (只存输入): 前向只留 a0 和 a36 两份, 看起来最省。但反向第一步就要算第 35 层的梯度, 它需要 a35, 而 a35 只能从 a0 一路重跑 35 层前向得到, 中间的 a1 到 a35 全部临时出现在显存里, 峰值又回到 37 份。白白多算一遍前向, 一点没省。\n- $k=6=\\sqrt{L}$: 前向存 a0、a6、a12、a18、a24、a30、a36 共 7 个边界; 反向走到最后一段时从 a30 重算 a31 到 a35 共 5 份临时激活, 峰值 7 + 5 = 12 份。把播放器拖到最挤的那一帧就能数出来。\n公式是峰值 ≈ L/k + k, 两项一个随 k 减小一个随 k 增大, 在 k = √L 处最小。重算的计算代价几乎与 k 无关: 不管怎么分段, 都是把每段除出口以外的层多跑一次, 合计约多一次前向, 按前向:反向 = 1:2 折算在 +33% 以内。所以 k 只需要对着显存选, 不用考虑算力。',
    }"
  >
    <template #controls>
      <LabSlider v-model="L" label="层数 L" :min="4" :max="64" />
      <LabSlider v-model="k" label="段长 k (每 k 层存一次)" :min="1" :max="L" />
      <StepPlayer :stepper="stepper" :label="frame.label" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="峰值激活随段长变化曲线, 曲线上的圆点可拖">
      <line :x1="X0" :x2="W - 10" :y1="H - 24" :y2="H - 24" class="axis" />
      <line :x1="X0" :x2="X0" :y1="10" :y2="H - 24" class="axis" />
      <text :x="X0 - 6" :y="py(L + 1) + 4" class="tick" text-anchor="end">{{ L + 1 }}</text>
      <text :x="X0 - 6" :y="py(best.peak) + 4" class="tick" text-anchor="end">{{ best.peak }}</text>
      <text :x="W - 10" :y="H - 8" class="tick" text-anchor="end">段长 k →</text>
      <text :x="X0 + 4" :y="14" class="tick">峰值激活份数</text>
      <line :x1="px(Math.sqrt(L))" :x2="px(Math.sqrt(L))" :y1="14" :y2="H - 24" class="sqrt" />
      <text :x="px(Math.sqrt(L)) + 4" :y="26" class="sqrt-t">√L = {{ Math.sqrt(L).toFixed(1) }}</text>
      <polyline :points="curve.map((c) => `${px(c.k)},${py(c.peak)}`).join(' ')" class="curve" />
      <circle
        :cx="px(k)" :cy="py(cur.peak)" r="8" class="handle draggable" tabindex="0" role="slider"
        aria-label="段长 k" :aria-valuenow="k"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (k = Math.round(clamp(pxInv(x), 1, L))) })"
        @keydown.right.prevent="k = Math.min(L, k + 1)" @keydown.left.prevent="k = Math.max(1, k - 1)"
      />
    </svg>

    <div class="cells acts" :style="{ gridTemplateColumns: `repeat(${Math.min(L + 1, 22)}, minmax(18px, 1fr))` }">
      <span
        v-for="i in L + 1" :key="i" class="cell"
        :class="[frame.live[i - 1] === 'ckpt' ? 'on' : frame.live[i - 1] === 'tmp' ? 'hot' : 'dim', { now: frame.at === i - 1 }]"
        :title="`a${i - 1}: 第 ${i - 1} 层的输入` + (frame.live[i - 1] ? '' : ' (此刻不在显存里)')"
      >{{ i - 1 }}</span>
    </div>
    <p class="legend">
      <span class="cell on" /> 前向存下的段边界 <span class="cell hot" /> 反向时重算出来的 <span class="cell dim" /> 此刻不在显存里
    </p>

    <template #stats>
      <div class="kv"><span>此刻活着的激活 (当前帧显存里的份数)</span><b>{{ frame.count }}</b></div>
      <div class="kv"><span>峰值 (k = {{ k }} 时整个过程的最大值)</span><b :class="cur.peak === best.peak ? 'good' : cur.peak > L * 0.8 ? 'bad' : ''">{{ cur.peak }} 份</b></div>
      <div class="kv"><span>最优 k = {{ best.k }} 时的峰值</span><b>{{ best.peak }} 份</b></div>
      <div class="kv"><span>前向层调用 {{ cur.fwd }} 次 (不重算是 {{ L }} 次), 多出的计算量 (按前向:反向 = 1:2 折算)</span><b :class="{ bad: cur.extra > 0.3 }">+{{ (cur.extra * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">
        ★ 峰值出现在反向刚重算完最后一段的时候: 所有段边界都还在显存里, 再加上这一段内部刚算出来的激活, 约 L/k + k 份。
        重算出来的激活和原来逐位相同, 所以梯度也与不重算逐位相同 (m07 实测差 0.0)。换的只是「存」和「算」。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const L = ref(36), k = ref(6), svg = ref(null)
const { start } = useDrag()
watch(L, (v) => { if (k.value > v) k.value = v })

// 和 Python 的 run() 逐行对应: live 记录每个激活是 ckpt(前向存的) / tmp(重算出的) / 无
const simulate = (Ln, seg, wantFrames) => {
  const live = Array(Ln + 1).fill(null)
  let peak = 0, fwd = 0
  const frames = []
  const snap = (label, at) => {
    const count = live.filter(Boolean).length
    peak = Math.max(peak, count)
    if (wantFrames) frames.push({ label, at, live: [...live], count })
  }
  live[0] = 'ckpt'; snap('输入', 0)
  for (let i = 0; i < Ln; i++) {
    fwd++
    if ((i + 1) % seg === 0 || i + 1 === Ln) live[i + 1] = 'ckpt' // 只在段边界保存
    snap(`前向 层${i}`, i + 1)
  }
  for (let s0 = Math.floor((Ln - 1) / seg) * seg; s0 >= 0; s0 -= seg) {
    const end = Math.min(s0 + seg, Ln)
    for (let i = s0; i < end - 1; i++) { fwd++; live[i + 1] = 'tmp'; snap(`重算 层${i}`, i + 1) } // ★ 瞬时激活也占显存
    for (let i = end - 1; i >= s0; i--) { live[i + 1] = null; snap(`反向 层${i}`, i) }
  }
  return { peak, fwd, frames, extra: (fwd - Ln) / (3 * Ln) }
}

const curve = computed(() => range(L.value).map((i) => ({ k: i + 1, ...simulate(L.value, i + 1, false) })))
const best = computed(() => curve.value.reduce((a, b) => (b.peak < a.peak ? b : a)))
const cur = computed(() => simulate(L.value, k.value, true))
const stepper = useStepper(() => cur.value.frames.length, { interval: 160 })
const frame = computed(() => cur.value.frames[Math.min(stepper.step.value, cur.value.frames.length - 1)])
// 换参数后停在峰值那一帧: 先看到最挤的时刻
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.frames.findIndex((f) => f.count === c.peak) }, { immediate: true })

const W = 460, H = 190, X0 = 40
const px = (kk) => X0 + ((kk - 1) / Math.max(1, L.value - 1)) * (W - X0 - 20)
const pxInv = (x) => 1 + ((x - X0) / (W - X0 - 20)) * (L.value - 1)
const py = (p) => 18 + (1 - p / (L.value + 1)) * (H - 46)
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.sqrt { stroke: var(--left); stroke-dasharray: 4 3; }
.sqrt-t { font-size: 10px; fill: var(--left); }
.handle { fill: var(--warn); stroke: var(--bg-card); stroke-width: 2; }
.acts { margin-top: 12px; }
.acts .cell { min-width: 18px; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 8px; font-size: 11px; color: var(--text-dim); }
.legend .cell { min-width: 14px; width: 14px; height: 14px; }
</style>
