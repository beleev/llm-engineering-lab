<!--
  ZeRO 显存账 (对应 llm_train/m05_zero_fsdp)。
  只讲一件事: 混合精度 Adam 每参数 16 字节 = 2 + 2 + 12, ZeRO-1/2/3 依次把 12 / 2 / 2 这三块除以 N。
-->
<template>
  <LabFrame
    title="ZeRO 显存账 — 每参数 2 + 2 + 12 字节, 分三档切给多张卡, 先切哪一块?"
    sub="- 模拟什么: 采用混合精度 Adam 优化器训练时, 模型每个参数在显存中需常驻 16 字节状态: 包括 2 字节的 fp16 参数副本 (用于前向与反向计算)、2 字节的 fp16 梯度以及 12 字节的 fp32 优化器状态 (master 主权重、一阶动量 m 与二阶矩 v 各占 4 字节)。标准 DDP 数据并行令 N 张卡各自保存一份完整冗余副本, 无论扩展多少张卡, 单卡常驻状态显存恒为 $16\Psi$ 字节 ($\Psi$ 表示模型参数总数)。ZeRO (零冗余优化器) 将上述模型状态沿数据并行组切分为 N 个互斥分片, 每张卡仅常驻保管本地负责的 1/N 分片, 计算需要完整张量时再通过集合通信按需临时拼齐。本实验台按解析公式核算单卡常驻模型状态显存 (按十进制 1 GB 等于 1e9 字节折算, 故 1B 参数乘 1 字节恰好对应 1 GB; 前向激活显存不计入本账本)。ZeRO 按切分彻底程度分为三个级别:
  - ZeRO-1: 仅切分 12 字节的 fp32 优化器状态。
  - ZeRO-2: 进一步将 2 字节的 fp16 梯度一并切分。
  - ZeRO-3 (亦称完全分片数据并行 FSDP): 将 2 字节的 fp16 模型参数也彻底切分, 前向与反向算到某一层时临时发起 all-gather 收集拼齐该层完整参数, 算完立即释放显存。
- 图上是什么: 四根堆叠柱状图对比同一模型规模在 DDP、ZeRO-1、ZeRO-2 与 ZeRO-3 四种策略下的单卡常驻显存占用。
  - 每根柱自下而上依次堆叠三项状态: fp16 参数、fp16 梯度与 fp32 master+m+v, 色块颜色与下方图例对应。
  - ZeRO-3 柱顶额外带有一小格虚线描边块, 代表前向或反向计算单层时临时 all-gather 进显存的当层完整参数峰值开销, 因算完即释放而不计入常驻总量。
  - 柱顶数字标明单卡常驻显存总 GB 数, 低于单卡显存上限时显示绿色, 超出上限无法装入时标红。
  - 横跨画布的水平红色虚线代表设定的单卡可用显存上限。
  - 纵轴量程随最高柱高自动自适应缩放。
- 可以做什么: 控件从上到下依次为:
  - 拖动「模型参数量 Ψ」(范围为 1B 至 405B) 与「数据并行卡数 N」(范围为 1 至 1024), 观察四根状态柱的高度变化。
  - 上下拖动水平红色虚线 (或聚焦后使用上下方向键) 设定目标硬件的单卡显存容量上限。
  - 点击任一柱体或下方的 DDP / ZeRO 级别按钮选中特定档位, 右侧面板随即切换为该级别的详细读数。
  - 点击图例按钮或悬停柱体内部色块, 图表下方会详细列出该状态分量在当前选中级别下占用的 GB 数及其组成说明。
- 应该看到: 增大数据并行卡数 N 时, DDP 柱高始终恒定不变; ZeRO-1 下降至 $4\Psi$ 下限后便停止缩减, ZeRO-2 下降至 $2\Psi$ 下限后停止缩减, 唯有 ZeRO-3 随卡数增加持续成反比下降。右侧读数含义如下:
  - 「单卡常驻」: 当前选中级别下单张卡需常驻承载的模型状态 GB 数, 可装入显存上限时显示绿色。
  - 「公式」: 当前选中级别对应的单卡显存解析表达式。
  - 「放得进 ... 的最低级别」: 四个档位中首个满足显存上限约束的最轻量切分级别, 级别越低所需额外通信开销越小。
  - 「每步通信量」: 相对标准 DDP 的每步跨卡通信量倍数, ZeRO-1/2 与 DDP 完全持平, ZeRO-3 增至约 1.5 倍。"
    module="llm_train/m05"
    run="python -m llm_train.m05_zero_fsdp.demo"
    :challenge="{
      ask: '针对 7B 参数模型与 80 GB 显存的计算卡, 将数据并行卡数从 8 增至 64 时, ZeRO-1 的单卡常驻显存可从多少降至多少? 为什么继续无限增加卡数也无法进一步压低显存?',
      answer: '单卡常驻显存从 38.5 GB 降至 29.3 GB, 继续增加卡数也无法突破 28 GB 的理论下限。\n- 显存公式核算: ZeRO-1 仅切分占 $12\\Psi$ 的 fp32 优化器状态, 单卡常驻显存满足 $4\\Psi + 12\\Psi/N$。首项 $4\\Psi$ 对应每张卡均需完整保留的 fp16 参数 (2 字节) 与 fp16 梯度 (2 字节), 在 7B 参数规模下固定占用 28 GB, 属于加卡无法分摊的常驻下限。\n- 当 $N=8$ 时: 等于 28 加上 84/8, 即 38.5 GB。\n- 当 $N=64$ 时: 等于 28 加上 84/64, 即 29.3 GB。\n- 当 $N \\to \\infty$ 时: 分片项趋于 0, 总量仅能无限逼近 28 GB。拖动「数据并行卡数 N」滑杆观察 ZeRO-1 柱体可见其降幅逐级放缓并最终贴合在 28 GB 水平线上。\n若需突破该下限, 必须进一步切分梯度 (升级至 ZeRO-2, 常驻下限减半至 $2\\Psi$ 即 14 GB) 乃至连同模型参数一并切分 (升级至 ZeRO-3 / FSDP, 常驻下限归 0)。其代价体现在跨卡集合通信上:\n- ZeRO-3: 每层在前向传播与反向传播前均需各执行一次参数 all-gather 以临时还原完整层权重, 使单步总通信量增至标准 DDP 的约 1.5 倍。\n- ZeRO-1 与 ZeRO-2: 单步总通信量与标准 DDP 完全持平, 因为标准 ring all-reduce 本身即由一次 reduce-scatter 加一次 all-gather 构成, ZeRO-1/2 只是将参数优化更新步骤插入两阶段之间, 让各卡仅在本地负责的分片上完成更新后再执行参数 all-gather。',
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
      <div class="kv"><span>{{ stages[stage].name }} 每卡常驻显存 (绿色 = 放得进上限)</span><b :class="cur.total <= limit ? 'good' : 'bad'">{{ gb(cur.total) }}</b></div>
      <div class="kv"><span>公式 (Ψ 是参数个数, N 是卡数)</span><b class="formula"><Tex :text="stages[stage].formula" /></b></div>
      <div class="kv"><span>放得进 {{ limit }} GB 的最低级别 (级别越低通信越省)</span><b :class="firstFit < 0 ? 'bad' : 'good'">{{ firstFit < 0 ? '都放不下' : stages[firstFit].name }}</b></div>
      <div class="kv"><span>每步通信量 (相对 DDP 的倍数)</span><b :class="stages[stage].commX > 1 ? 'bad' : ''">{{ stages[stage].comm }}</b></div>
      <div class="lab-note">
        <p>本实验台仅核算模型静态与优化状态显存 (参数、梯度及优化器状态), 前向中间激活显存需独立叠加计算 (参见下方激活重算实验台)。</p>
        <ul class="pts">
          <li><b>显存地板:</b> <Tex text="增加卡数也无法切分的常驻显存下限: ZeRO-1 为 $4\Psi$ (含 fp16 参数与 fp16 梯度), ZeRO-2 为 $2\Psi$ (仅含 fp16 参数)。唯有 ZeRO-3 可随并行卡数 $N$ 增大持续下降: 计算到某层时按需 all-gather 该层参数, 算完立即释放, 瞬时峰值仅额外多出一层的完整参数量 (对应柱顶虚线框)。" /></li>
          <li><b>通信:</b> ZeRO-2 不比 DDP 贵, 因为 DDP 的 all-reduce 本来就等于 reduce-scatter (求和后每卡只留自己那片) 加 all-gather (各片拼回整份), ZeRO-2 只是在两半中间插了一步「各自更新自己那片」。ZeRO-1 若朴素地用 all-reduce + all-gather 实现会是 1.5×, DeepSpeed 改用 reduce-scatter 后持平。ZeRO-3 每层前向、反向都要先 all-gather 参数, 约 1.5×。</li>
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
