<!--
  浮点数轴 (对应 llm_train/core/numerics.py + m06_mixed_precision)。
  只讲一件事: 一个浮点格式 = 一段可表示范围 + 一个相对精度; 梯度分布落在范围外的部分会变 0 或 Inf, loss scale 就是把分布整体平移进范围。
-->
<template>
  <LabFrame
    title="浮点数轴 — 一批梯度落在哪个格式的范围里, loss scale 把它们挪进去"
    sub="- 模拟什么: 一个浮点格式用固定的位数存一个数: 指数位决定能表示多大、多小 (范围), 尾数位决定范围内相邻两个可表示的数挨得多近 (相对精度)。混合精度训练把前向反向放在 16 位 (FP16 或 BF16) 甚至 8 位 (FP8) 上算, 可梯度往往非常小: 落在格式最小值左边的直接变 0 (下溢), 落在最大值右边的变 Inf (溢出)。loss scaling 的办法是把 loss 先乘一个 scale 再反向, 整批梯度跟着一起乘 scale、平移进范围, 更新前再除回去。这个台子把五种格式的范围和一批典型梯度画在同一根对数数轴上。
      - 图上是什么: 横轴是数值大小, 对数刻度。从上到下:
        - 最上面的直方图: 一批典型梯度 $|g|$ 乘上 loss scale 之后的分布, 红色柱子是在当前选中的格式里会变 0 或溢出的部分。
        - 下面五条色带: 每条是一种格式能表示的范围。深色段是 normal 数 (相对精度正常), 浅色段是 subnormal 数 (比最小 normal 还小的一小段, 精度递减); 色带左端以左全部变 0, 右端以右溢出。FP32 和 BF16 的范围远超画面, 真实范围写在行首。
        - 带圆点的黄色竖线: 探针, 用来考察某一个具体的数。
      - 可以做什么: 控件从上到下:
        - 点格式按钮或直接点色带选一种格式。
        - 拖「loss scale」(2^0 到 2^24) 把整批梯度向右平移。
        - 拖「梯度典型量级」改这批梯度的中心位置。
        - 左右拖探针竖线 (或用方向键), 看这个数在选中格式里会被舍成什么。
      - 应该看到: 右侧前两行读探针, 后两行读整批梯度:
        - 「左 / 右邻居」: 选中格式里紧挨着这个数的两个可表示值, 两者挨得越近精度越高。
        - 「舍入后」: 它实际会存成什么, 后面是相对误差, 变 0 写「下溢」, 超过最大值写「溢出」(或「饱和」, FP8 的硬件做法是截到最大值而不是 Inf)。
        - 「变 0」和「溢出」: 分别是直方图里红色部分占的百分比。
        - FP16: scale = 1 时有一大截梯度变 0; 把 scale 调到 2^10 到 2^14 两项都接近 0; 再大右尾开始溢出。
        - BF16: 整条色带盖住整个分布, 不再需要 scale。"
    module="llm_train/m06"
    run="python -m llm_train.m06_mixed_precision.demo"
    :challenge="{
      ask: '依次做三件事, 每步先猜:\n1. 选 FP16, loss scale = 1 时有多少梯度变成 0?\n2. 把 scale 调到多大最合适?\n3. 再切到 BF16, 还需要 loss scale 吗?',
      answer: '答案:\n- FP16 下 scale = 1 时一大截梯度变 0。\n- scale 调到 $2^{10}$ 到 $2^{14}$ 最合适。\n- BF16 不需要 scale。\n逐条解释:\n- 为什么变 0: FP16 是 5 位指数加 10 位尾数, 最小的 normal 数是 $2^{-14}$ ≈ 6e-5, 再往下的 subnormal 段最小只能到 $2^{-24}$ ≈ 6e-8。默认这批梯度的中心在 1e-6, 左边有一大截比 6e-8 还小, 读数「梯度 × scale 后变 0」会显示一个明显的百分比, 直方图左半是红的。\n- 怎么选 scale: 乘 $2^{10}$ 到 $2^{14}$ 相当于把直方图向右平移 3 到 4 个数量级, 整个分布进了 FP16 的色带, 「变 0」接近 0, 「溢出」还是 0。再往右拖, 右尾撞上 FP16 的最大值 65504, 「溢出」开始变红: 一个 Inf 就会毁掉整步更新。\n- 动态 loss scaling 就是让框架自动找这个窗口: 发现溢出就把 scale 减半并跳过这一步, 连续多步正常就把 scale 翻倍。m06 的 24 步里跳了 5 步, scale 在 $2^{16}$ 到 $2^{20}$ 之间来回。\n- BF16 是 8 位指数加 7 位尾数, 指数位和 FP32 一样多, 色带两端都伸出画面, 整个梯度分布怎么挪都在范围内, 所以不需要 loss scale。代价是尾数只剩 7 位, 相邻可表示数的间距是 FP16 的 8 倍: 把探针拖到同一个数上对比两种格式的「相对误差」就能看到。现代预训练几乎都用 BF16 省掉 scaling, FP16 加 scaler 主要见于老显卡。',
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
