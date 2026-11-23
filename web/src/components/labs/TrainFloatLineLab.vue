<!--
  浮点数轴 (对应 llm_train/core/numerics.py + m06_mixed_precision)。
  只讲一件事: 一个浮点格式 = 一段可表示范围 + 一个相对精度; 梯度分布落在范围外的部分会变 0 或 Inf, loss scale 就是把分布整体平移进范围。
-->
<template>
  <LabFrame
    title="浮点数轴 — 一批梯度落在哪个格式的范围里, loss scale 把它们挪进去"
    sub="- 模拟什么: 浮点格式在固定位宽下编码实数: 指数位决定可表示的最大与最小幅值 (动态范围), 尾数位决定区间内相邻可表示数值的疏密程度 (相对精度)。混合精度训练将前向与反向传播放在 16 位 (FP16 或 BF16) 乃至 8 位 (FP8) 格式下执行, 但梯度幅值通常极小: 落在格式最小非零值左侧的梯度直接下溢归零, 落在最大可表示值右侧的梯度则上溢为 Inf。loss scaling 在反向传播前先将 loss 乘以缩放系数 scale, 使整批梯度同步放大并平移进入目标格式的可表示区间, 参数更新前再除以同一系数还原。本实验台将五种浮点格式的可表示范围与一批典型梯度分布绘制在同一条对数数轴上。
- 图上是什么: 横轴为数值绝对值大小 (对数刻度), 从上到下依次为:
  - 顶部直方图: 一批典型梯度幅值 $|g|$ 乘以 loss scale 后的分布, 红色柱表示在当前选中格式下发生下溢归零或上溢的部分。
  - 下方五条色带: 每条色带对应一种浮点格式的可表示范围。深色段为 normal 数区间 (保持恒定相对精度), 浅色段为 subnormal 数区间 (低于最小 normal 数的过渡段, 有效精度逐步下降); 色带左端以左全部下溢为 0, 右端以右发生溢出。FP32 与 BF16 的完整范围远超画布边界, 真实边界标注于行首。
  - 带圆点的黄色竖线: 数值探针, 用于检查特定实数的量化舍入行为。
- 可以做什么: 控件从上到下依次为:
  - 点击格式按钮或直接点击对应色带切换目标浮点格式。
  - 拖动「loss scale」(范围为 2^0 至 2^24) 将整批梯度分布沿数轴向右平移。
  - 拖动「梯度典型量级」调整该批梯度分布的中心位置。
  - 左右拖动黄色探针竖线 (或使用方向键微调), 查看探针所在数值在选中格式下的舍入结果。
- 应该看到: 右侧前两行显示探针读数, 后两行统计整批梯度分布状态:
  - 「左 / 右邻居」: 选中格式中紧邻探针数值的两个可表示离散值, 间距越小则局部精度越高。
  - 「舍入后」: 探针数值实际存储的量化值及其相对误差; 若变为 0 则标注「下溢」, 若超出最大值则标注「溢出」(FP8 硬件通常截断至最大有限值, 标注为「饱和」而非 Inf)。
  - 「变 0」与「溢出」: 直方图中下溢与上溢红色部分各自所占的百分比。
  - FP16 模式下: 当 scale 为 1 时有相当比例的小梯度下溢为 0; 将 scale 调至 2^10 至 2^14 区间时下溢与溢出比例均趋近于 0; 继续调大则分布右尾开始上溢。
  - BF16 模式下: 色带完整覆盖整个梯度分布, 无需额外配置 loss scale。"
    module="llm_train/m06"
    run="python -m llm_train.m06_mixed_precision.demo"
    :challenge="{
      ask: '依次完成以下三步观察, 每步操作前先预测结果:\n1. 选择 FP16 格式, 当 loss scale 为 1 时有多少比例的梯度下溢为 0?\n2. 将 scale 调至什么区间最为合适?\n3. 切换至 BF16 格式后, 是否仍需使用 loss scale?',
      answer: '结论:\n- FP16 格式在 scale 为 1 时有大比例梯度下溢为 0。\n- 将 scale 调至 $2^{10}$ 至 $2^{14}$ 区间最为合适。\n- BF16 格式无需使用 loss scale。\n逐项解析:\n- 下溢原因: FP16 采用 5 位指数与 10 位尾数, 最小 normal 数为 $2^{-14}$ (约为 6e-5), 继续向下延伸的 subnormal 区间下限仅为 $2^{-24}$ (约为 6e-8)。默认梯度分布的中心位于 1e-6, 左半部分有大量梯度小于 6e-8, 因此右侧「梯度 × scale 后变 0」读数会显示显著的下溢比例, 直方图左侧呈红色。\n- scale 取值窗口: 乘以 $2^{10}$ 至 $2^{14}$ 相当于将梯度直方图整体向右平移 3 至 4 个数量级, 使整批梯度完整落入 FP16 色带内部, 此时「变 0」趋近于 0 且「溢出」保持为 0。若继续向右调大, 分布右尾将超出 FP16 的最大值 65504, 「溢出」读数变红预警, 而单个 Inf 梯度便会导致整步参数更新失效。\n- 动态 loss scaling 的作用正是自动追踪这一安全窗口: 一旦检测到梯度溢出便将 scale 减半并跳过当步更新; 若连续多步未溢出则将 scale 翻倍。在 m06 的 24 步实测中跳过了 5 步更新, scale 在 $2^{16}$ 至 $2^{20}$ 之间动态调整。\n- BF16 采用 8 位指数与 7 位尾数, 指数位宽与 FP32 完全一致, 色带两端远超画布边界, 梯度分布无论如何平移均处于可表示范围内, 因此无需 loss scale。其代价是尾数缩减至 7 位, 相邻可表示数的间距扩大为 FP16 的 8 倍, 将探针拖至同一数值处对比两种格式的「相对误差」即可验证。现代大模型预训练普遍采用 BF16 以省去 loss scaling 流程, FP16 配合 scaler 主要用于不支持原生 BF16 的早期显卡。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(f, key) in FMT" :key="key" type="button" :class="{ active: fmt === key }" @click="fmt = key">{{ f.name }}</button>
      </div>
      <LabSlider v-model="logScale" label="loss scale" :min="0" :max="24" :format="(k) => '2^' + k" />
      <LabSlider v-model="gradMu" label="梯度典型量级" :min="-9" :max="-2" :step="0.5" :format="(v) => '1e' + v" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="各浮点格式的可表示范围与梯度直方图; 色带可选, 竖线可拖">
      <g v-for="t in ticks" :key="t">
        <line :x1="px(t)" :x2="px(t)" :y1="HIST_H" :y2="H - 18" class="grid" />
        <text :x="px(t)" :y="H - 5" class="tick" text-anchor="middle">1e{{ t }}</text>
      </g>
      <rect
        v-for="(b, i) in hist" :key="i" :x="px(b.x0)" :width="Math.max(0.5, px(b.x1) - px(b.x0) - 0.5)"
        :y="HIST_H - b.h * (HIST_H - 8)" :height="b.h * (HIST_H - 8)"
        :class="['hbar', b.state]"
      />
      <g v-for="(f, key, i) in FMT" :key="key" class="band" :class="{ sel: fmt === key }" tabindex="0" role="button" :aria-label="`选中 ${f.name}`" @click="fmt = key" @keydown.enter="fmt = key">
        <rect :x="0" :y="rowY(i) - 2" :width="W" :height="ROW - 2" class="hit" />
        <rect :x="px(log10(f.minSub))" :width="px(log10(f.minNorm)) - px(log10(f.minSub))" :y="rowY(i) + 3" :height="ROW - 12" :fill="f.color" opacity="0.3" />
        <rect :x="px(log10(f.minNorm))" :width="px(log10(f.max)) - px(log10(f.minNorm))" :y="rowY(i) + 3" :height="ROW - 12" :fill="f.color" opacity="0.85" />
        <text :x="6" :y="rowY(i) + ROW / 2 - 2" class="fname">{{ f.name }}</text>
        <text :x="6" :y="rowY(i) + ROW / 2 + 9" class="frange">{{ f.minSub.toExponential(0) }} ~ {{ f.max.toExponential(1) }}</text>
      </g>
      <!-- ★ 可拖的数值 -->
      <g class="draggable focus-inset" tabindex="0" role="slider" aria-label="被考察的数值" :aria-valuenow="logV"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (logV = clamp(pxInv(x), LO, HI)) })"
        @keydown.right.prevent="logV = Math.min(HI, logV + 0.25)" @keydown.left.prevent="logV = Math.max(LO, logV - 0.25)">
        <rect :x="px(logV) - 10" :y="0" width="20" :height="H - 18" fill="transparent" />
        <line :x1="px(logV)" :x2="px(logV)" :y1="4" :y2="H - 18" class="probe" />
        <circle :cx="px(logV)" :cy="HIST_H" r="6" class="probe-dot" />
      </g>
    </svg>

    <template #stats>
      <!-- 前两行读竖线上的那个数, 后两行读整批梯度 -->
      <div class="kv"><span>探针 x = {{ sci(value) }} 在 {{ F.name }} 里的左 / 右邻居 (紧挨着它的两个可表示值)</span><b class="nb">{{ sci(nb.lo) }} / {{ sci(nb.hi) }}</b></div>
      <div class="kv"><span>x 实际会存成 (相对误差 {{ nb.err >= 1 ? '100' : (nb.err * 100).toPrecision(2) }}%)</span><b class="nb" :class="nb.q === 0 || !isFinite(nb.q) || nb.err > 0.05 ? 'bad' : 'good'">{{ nb.q === 0 ? '0 (下溢)' : isFinite(nb.q) ? sci(nb.q) : (F.saturate ? '饱和' : 'Inf (溢出)') }}</b></div>
      <div class="kv"><span>整批梯度 × scale 后变 0 的比例 (直方图左侧红色)</span><b :class="frac.under > 0.01 ? 'bad' : 'good'">{{ (frac.under * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>整批梯度 × scale 后溢出的比例 (直方图右侧红色)</span><b :class="frac.over > 0 ? 'bad' : 'good'">{{ (frac.over * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">指数位决定色带多长 (范围), 尾数位决定邻居多密 (精度)。当前格式 {{ F.name }}: {{ F.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

// 一个格式只由三个数决定 (同 core/numerics.py): 尾数位 m, 最大值, 最小 normal 指数 eMin
const mk = (name, m, max, eMin, color, note, saturate = false) =>
  ({ name, m, max, eMin, color, note, saturate, minNorm: 2 ** eMin, minSub: 2 ** (eMin - m) })
const FMT = {
  fp32: mk('FP32', 23, 3.4028e38, -126, 'var(--text-dim)', '8 位指数 + 23 位尾数, 基准。'),
  bf16: mk('BF16', 7, 3.3895e38, -126, 'var(--left)', '8 位指数 + 7 位尾数: 范围同 FP32, 精度约 0.4%。'),
  fp16: mk('FP16', 10, 65504, -14, 'var(--accent)', '5 位指数 + 10 位尾数: 精度 0.05%, 但小于 6e-8 就没了。'),
  e5m2: mk('FP8 E5M2', 2, 57344, -14, 'var(--eye)', '5 位指数 + 2 位尾数: 范围近似 FP16, 相对误差可到 ~12%; TE 配方用它装梯度。', true),
  e4m3: mk('FP8 E4M3', 3, 448, -6, 'var(--right)', '4 位指数 + 3 位尾数: 只有 ±448、最小 2e-3, 必须配 scale 才能用。', true),
}
const fmt = ref('fp16'), logScale = ref(0), gradMu = ref(-6), logV = ref(-5), svg = ref(null)
const { start } = useDrag()
const F = computed(() => FMT[fmt.value])
const log10 = Math.log10
const value = computed(() => 10 ** logV.value)

// ★ 找 x 在格式网格上的左右邻居: normal 区步长 = 2^(⌊log2 x⌋ − m), subnormal 区步长固定 2^(eMin − m)
const neighbours = (x, f) => {
  if (x > f.max) return { lo: f.max, hi: Infinity, q: f.saturate ? f.max : Infinity }
  const e = Math.max(Math.floor(Math.log2(x)), f.eMin)
  const stepSize = 2 ** (e - f.m)
  const lo = Math.floor(x / stepSize) * stepSize, hi = lo + stepSize
  return { lo, hi, q: x - lo < hi - x ? lo : hi }
}
const nb = computed(() => {
  const n = neighbours(value.value, F.value)
  const q = n.q === Infinity && F.value.saturate ? F.value.max : n.q
  return { ...n, err: isFinite(q) ? Math.abs(q - value.value) / value.value : 1 }
})

// 一批对数正态的梯度 (固定 seed, 拖滑杆不乱跳), 乘上 loss scale 后分桶
const samples = range(4000).map(((rand) => () => randn(rand))(mulberry32(7)))
const LO = -14, HI = 8, BINS = 110 // 只画梯度活动的区间; FP32/BF16 的色带两端伸出画面外 (真实范围写在行首)
const scaled = computed(() => samples.map((z) => gradMu.value + 1.3 * z + logScale.value * log10(2)))
const stateOf = (lg) => (lg > log10(F.value.max) ? 'over' : lg < log10(F.value.minSub / 2) ? 'under' : 'ok')
const hist = computed(() => {
  const counts = Array(BINS).fill(0)
  for (const lg of scaled.value) counts[clamp(Math.floor(((lg - LO) / (HI - LO)) * BINS), 0, BINS - 1)]++
  const top = Math.max(...counts)
  return counts.map((c, i) => {
    const x0 = LO + (i / BINS) * (HI - LO), x1 = LO + ((i + 1) / BINS) * (HI - LO)
    return { x0, x1, h: c / top, state: stateOf((x0 + x1) / 2) }
  }).filter((b) => b.h > 0)
})
const frac = computed(() => {
  const s = scaled.value.map(stateOf)
  return { under: s.filter((v) => v === 'under').length / s.length, over: s.filter((v) => v === 'over').length / s.length }
})

const W = 640, H = 250, HIST_H = 70, ROW = 30
const XL = 100
const px = (lg) => XL + ((clamp(lg, LO, HI) - LO) / (HI - LO)) * (W - XL - 10)
const pxInv = (x) => LO + ((x - XL) / (W - XL - 10)) * (HI - LO)
const rowY = (i) => HIST_H + 8 + i * ROW
const ticks = [-12, -8, -4, 0, 4, 8]
const sci = (v) => (v === 0 ? '0' : !isFinite(v) ? '∞' : v.toExponential(3))
</script>

<style scoped>
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.fname { font-size: 10px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.band { cursor: pointer; outline: none; }
.band .hit { fill: transparent; }
.band.sel .hit, .band:focus-visible .hit { fill: var(--accent-soft); }
.frange { font-size: 8px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.band.sel .fname { fill: var(--text); font-weight: 600; }
.hbar { fill: var(--text-dim); }
.hbar.under, .hbar.over { fill: var(--danger); }
.probe { stroke: var(--warn); stroke-width: 2; }
.probe-dot { fill: var(--warn); stroke: var(--bg-card); stroke-width: 2; }
.nb { font-size: 12px !important; }
</style>
