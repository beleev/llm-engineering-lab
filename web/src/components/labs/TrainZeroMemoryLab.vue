<!--
  ZeRO 显存账 (对应 llm_train/m05_zero_fsdp)。
  只讲一件事: 混合精度 Adam 每参数 16 字节 = 2 + 2 + 12, ZeRO-1/2/3 依次把 12 / 2 / 2 这三块除以 N。
-->
<template>
  <LabFrame
    title="ZeRO 显存账 — 2 + 2 + 12 字节, 先切哪一块?"
    sub="四根柱子是同一个模型在 DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 下每张卡的常驻显存。
      上下拖动横着的虚线设定单卡显存上限, 看哪一级开始放得下。点柱子选中一级, 点图例或悬停色块看它是什么。纵轴跟着最高的柱子伸缩。"
    module="llm_train/m05"
    run="python -m llm_train.m05_zero_fsdp.demo"
    :challenge="{
      ask: '7B 模型、80 GB 的卡。卡数从 8 加到 64, ZeRO-1 的单卡显存能降到多少? 为什么再加卡也没用?',
      answer: 'ZeRO-1 只切 $12\\Psi$ 的优化器状态: $4\\Psi + 12\\Psi/N$。7B 时 $4\\Psi$ = 28 GB 是切不掉的地板:\n- $N=8$: 38.5 GB\n- $N=64$: 29.3 GB\n- $N \\to \\infty$: 也只到 28 GB\n想继续降, 只能把梯度 (ZeRO-2) 和参数 (ZeRO-3/FSDP) 也切掉。代价是 ZeRO-3 每层前向、反向各多一次 all-gather, 通信量约 1.5×。',
    }"
  >
    <template #controls>
      <LabSlider v-model="sizeIdx" label="模型参数量 Ψ" :min="0" :max="sizes.length - 1" :format="(i) => sizes[i] + 'B'" />
      <LabSlider v-model="logN" label="数据并行卡数 N" :min="0" :max="10" :format="(k) => 2 ** k" />
      <div class="row">
        <button v-for="(s, i) in stages" :key="s.name" type="button" :class="{ active: stage === i }" @click="stage = i">{{ s.name }}</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="各 ZeRO 级别的单卡显存堆叠柱状图, 柱子可选, 上限线可拖">
      <g v-for="t in ticks" :key="t">
        <line :x1="X0" :x2="W - 8" :y1="y(t)" :y2="y(t)" class="grid" />
        <text :x="X0 - 6" :y="y(t) + 4" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <text :x="4" :y="12" class="tick">GB / 卡</text>
      <g
        v-for="(b, i) in bars" :key="i" class="bar" :class="{ sel: stage === i }"
        tabindex="0" role="button" :aria-label="`选中 ${stages[i].name}`"
        @click="stage = i" @keydown.enter="stage = i"
      >
        <rect :x="bx(i) - 8" :y="8" :width="BW + 16" :height="H - 34" class="hit" />
        <rect
          v-for="seg in b.segs" :key="seg.key"
          :x="bx(i)" :y="y(seg.top)" :width="BW" :height="Math.max(0, y(seg.bot) - y(seg.top))"
          :fill="seg.color" :class="{ transient: seg.key === 'gather' }"
          @mouseenter="hover = { s: i, key: seg.key }" @mouseleave="hover = null" @click="pin = { s: i, key: seg.key }"
        />
        <text :x="bx(i) + BW / 2" :y="Math.max(20, y(b.total) - 6)" class="val" text-anchor="middle" :class="b.total <= limit ? 'fit' : 'oom'">
          {{ gb(b.total) }}
        </text>
        <text :x="bx(i) + BW / 2" :y="H - 8" class="lbl" text-anchor="middle">{{ stages[i].name }}</text>
      </g>
      <!-- ★ 可拖的显存上限线 -->
      <g class="draggable" tabindex="0" role="slider" aria-label="单卡显存上限" :aria-valuenow="limit"
        @pointerdown="start($event, { svg, onMove: ({ y: py }) => (limit = Math.round(clamp(yInv(py), 16, 192))) })"
        @keydown.up.prevent="limit = Math.min(192, limit + 4)" @keydown.down.prevent="limit = Math.max(16, limit - 4)">
        <rect :x="X0" :y="y(limit) - 9" :width="W - X0 - 8" height="18" fill="transparent" />
        <line :x1="X0" :x2="W - 8" :y1="y(limit)" :y2="y(limit)" class="limit" />
        <text :x="W - 10" :y="y(limit) - 5" class="limit-t" text-anchor="end">⇅ 单卡上限 {{ limit }} GB</text>
      </g>
    </svg>
    <!-- 图例就是按钮: 触屏和键盘靠它固定选中一块, 鼠标悬停色块是同一个说明 -->
    <div class="row legend">
      <button v-for="l in legend" :key="l.key" type="button" :class="{ active: shown && shown.key === l.key }" @click="pin = { s: l.key === 'gather' ? 3 : stage, key: l.key }">
        <i class="sw" :class="{ tr: l.key === 'gather' }" :style="{ background: l.color }" /><Tex :text="l.text" />
      </button>
    </div>
    <p class="lab-note" aria-live="polite">
      <template v-if="shown">{{ shown.stage }} · {{ shown.label }}: {{ gb(shown.top - shown.bot) }}。{{ shown.desc }}</template>
      <template v-else>点一个图例, 看这一块在选中的级别下占多少。</template>
    </p>

    <template #stats>
      <div class="kv"><span>{{ stages[stage].name }} 单卡常驻</span><b :class="cur.total <= limit ? 'good' : 'bad'">{{ gb(cur.total) }}</b></div>
      <div class="kv"><span>公式</span><b class="formula"><Tex :text="stages[stage].formula" /></b></div>
      <div class="kv"><span>放得进 {{ limit }} GB 的最低级别</span><b :class="firstFit < 0 ? 'bad' : 'good'">{{ firstFit < 0 ? '都放不下' : stages[firstFit].name }}</b></div>
      <div class="kv"><span>每步通信量 (相对 DDP)</span><b :class="stages[stage].commX > 1 ? 'bad' : ''">{{ stages[stage].comm }}</b></div>
      <div class="lab-note">
        <p>这里只算模型状态, 激活另算 (见下面的重算实验台)。</p>
        <ul class="pts">
          <li><b>显存地板:</b> <Tex text="ZeRO-1 是 $4\Psi$, ZeRO-2 是 $2\Psi$。只有 ZeRO-3 随 $N$ 无限下降: 用到哪层 gather 哪层, 算完立刻 free, 瞬时只多出一层的完整参数。" /></li>
          <li><b>通信:</b> ZeRO-2 不比 DDP 贵, 因为 all-reduce 本来就 = reduce-scatter + all-gather。ZeRO-1 若朴素地用 all-reduce + all-gather 实现会是 1.5×, DeepSpeed 改用 reduce-scatter 后持平。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

const sizes = [1, 3, 7, 13, 34, 70, 175, 405]
const LAYERS = 32
const stages = [
  { name: 'DDP', formula: '$16\\Psi$', comm: '1×', commX: 1 },
  { name: 'ZeRO-1', formula: '$4\\Psi + 12\\Psi/N$', comm: '1×', commX: 1 },
  { name: 'ZeRO-2', formula: '$2\\Psi + 14\\Psi/N$', comm: '1×', commX: 1 },
  { name: 'ZeRO-3', formula: '$16\\Psi/N$', comm: '≈1.5×', commX: 1.5 },
]
const sizeIdx = ref(2), logN = ref(3), stage = ref(1), limit = ref(80), svg = ref(null)
const hover = ref(null), pin = ref(null)                 // { s: 第几级, key: 哪一块 }; 悬停优先于点击固定的
const { start } = useDrag()

const W = 460, H = 300, X0 = 44, BW = 62
// ★ 纵轴上限跟着最高的柱子走 (最低 200 GB), 取一个整的数。写死的话大模型的柱子全顶到头, 比不出高低
const NICE = [1, 1.5, 2, 3, 4, 6, 8, 10]
const yMax = computed(() => {
  const top = Math.max(200, 1.02 * Math.max(...bars.value.map((b) => b.segs[b.segs.length - 1].top)))
  const unit = 10 ** Math.floor(Math.log10(top))
  return unit * NICE.find((n) => n * unit >= top)
})
const ticks = computed(() => [0, 1, 2, 3, 4].map((i) => (yMax.value * i) / 4))
const y = (g) => 20 + (1 - Math.min(g, yMax.value) / yMax.value) * (H - 50)
const yInv = (py) => (1 - (py - 20) / (H - 50)) * yMax.value
const bx = (i) => X0 + 22 + i * (BW + 38)
const gb = (g) => (g >= 100 ? g.toFixed(0) : g.toFixed(1)) + ' GB'

// ★ 全部显存账就这一张表: 每块多少字节/参数, 从第几级开始除以 N。1 GB = 1e9 字节, 所以 1B 参数 × 1 字节 = 1 GB
const parts = [
  { key: 'p', label: 'fp16 参数', bytes: 2, shardFrom: 3, color: 'var(--accent)', desc: '前向反向都要用, 只有 ZeRO-3/FSDP 才切它' },
  { key: 'g', label: 'fp16 梯度', bytes: 2, shardFrom: 2, color: 'var(--eye)', desc: 'ZeRO-2 起用 reduce-scatter 同步, 每卡只留自己那片' },
  { key: 'o', label: 'fp32 master + m + v', bytes: 12, shardFrom: 1, color: 'var(--right)', desc: 'Adam 逐元素更新, 每卡只更新自己那 1/N, 结果逐位相同' },
]
const bars = computed(() => {
  const psi = sizes[sizeIdx.value], N = 2 ** logN.value
  return stages.map((_, s) => {
    let acc = 0
    const segs = parts.map((p) => {
      const size = (p.bytes * psi) / (s >= p.shardFrom ? N : 1)
      const seg = { ...p, bot: acc, top: acc + size }
      acc += size
      return seg
    })
    const total = acc
    if (s === 3 && N > 1) {
      const extra = (2 * psi) / LAYERS * (1 - 1 / N) // 一层完整 fp16 参数里不属于自己的那部分
      segs.push({ key: 'gather', label: '瞬时 all-gather', color: 'var(--accent)', bot: acc, top: acc + extra, desc: '用到哪层 gather 哪层, 算完立刻 free; 不计入常驻' })
    }
    return { segs, total }
  })
})
const cur = computed(() => bars.value[stage.value])
const legend = [
  ...parts.map((p) => ({ key: p.key, color: p.color, text: `${p.label} $${p.bytes}\\Psi$` })),
  { key: 'gather', color: 'var(--accent)', text: `ZeRO-3 瞬时 all-gather 的一层 (按 ${LAYERS} 层估)` },
]
const shown = computed(() => {
  const h = hover.value || pin.value
  const seg = h && bars.value[h.s].segs.find((x) => x.key === h.key)
  return seg ? { ...seg, stage: stages[h.s].name } : null
})
const firstFit = computed(() => bars.value.findIndex((b) => b.total <= limit.value))
</script>

<style scoped>
.grid { stroke: var(--border); stroke-width: 1; }
.tick, .lbl { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.lbl { font-size: 11px; fill: var(--text-muted); }
.bar { cursor: pointer; outline: none; }
.bar .hit { fill: transparent; stroke: transparent; rx: 6; }
.bar.sel .hit, .bar:focus-visible .hit { fill: var(--accent-soft); stroke: var(--accent); }
.bar.sel .lbl { fill: var(--text); font-weight: 600; }
.transient { opacity: 0.35; stroke: var(--accent); stroke-dasharray: 3 2; }
.val { font-size: 11px; font-family: "SF Mono", Menlo, monospace; }
.val.fit { fill: var(--left); }
.val.oom { fill: var(--danger); }
.limit { stroke: var(--danger); stroke-width: 2; stroke-dasharray: 6 4; }
.limit-t { font-size: 11px; fill: var(--danger); }
.formula { font-size: 13px !important; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.legend button { font-size: 11px; min-height: 28px; padding: 2px 8px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.sw.tr { opacity: 0.35; outline: 1px dashed var(--accent); }
</style>
