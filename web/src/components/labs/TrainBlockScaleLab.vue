<!--
  Block scaling (对应 llm_train/m13_fp8_training + m15_fp4_microscaling + core/numerics.py:quant_blockwise)。
  只讲一件事: scale 由 amax 决定, 一个 outlier 会拖垮和它共享 scale 的所有元素 —— block 越小, 受害者越少。
-->
<template>
  <LabFrame
    title="Block scaling — 低位宽量化时, 一个 outlier 会连累多少邻居?"
    sub="- 模拟什么: 低位宽浮点格式 (如 8 位的 FP8 与 4 位的 FP4) 可表示的动态范围很窄, 张量元素需先除以缩放因子 scale 映射至可表示区间内再舍入, 反量化时再乘回 scale。scale 由一组元素中的最大绝对值 (amax) 决定, 约为 amax 除以目标格式的最大可表示值。当整张量共用单一 scale 时, 单个幅值极大的 outlier (离群值) 会拉高 scale, 导致共享该 scale 的常规小元素在缩放后趋近于 0, 舍入后直接下溢归零。block scaling 将张量按固定 block 大小切分为若干小组, 每组独立计算 scale, 使 outlier 仅影响同组内的相邻元素。本实验台使用一行 64 个元素的张量演示该过程, 常规元素典型量级为 0.02, 其中包含一个可调节倍数的 outlier。
- 图上是什么: 上方为柱状图, 下方为两行误差色块:
  - 上方 64 根柱子表示各元素的绝对值 (纵轴为对数刻度), 柱顶带圆形拖拽手柄的柱子即为 outlier。
  - 交替变换的背景底色将柱子划分为一个个独立的 block。
  - 每个 block 内的红色虚线标示该组 scale 下最小非零可表示值的一半, 柱高低于该虚线的元素在量化舍入后会归零。
  - 下方两行色块展示各元素量化后的相对误差: 第一行「整张量」为 64 个元素共用单一 scale 的结果, 第二行「block」为按当前 block 大小分组量化的结果。
  - 色块颜色编码误差幅度: 红色代表下溢归零 (相对误差为 100%), 黄色越深代表相对误差越大; 悬停色块可查看具体数值与相对误差。
- 可以做什么: 控件从上到下依次为:
  - 点击「FP8 E4M3 / MXFP4 / NVFP4」切换量化格式。
  - FP8 E4M3: 含 4 位指数与 3 位尾数, 最大可表示值为 448。
  - MXFP4 与 NVFP4: 元素本体均为 4 位 E2M1 格式 (仅能表示 0 以及 ±0.5、1、1.5、2、3、4、6), 区别在于 scale 本身的编码格式: MXFP4 的 scale 仅能取 2 的整数次幂, 而 NVFP4 的 scale 采用带 3 位尾数的浮点格式以紧贴 amax。
  - 拖动「block 大小」(范围为 4 至 64, 取 64 时等价于整张量共用 scale)。
  - 拖动「outlier 倍数」滑杆或上下拖动柱顶手柄, 调整 outlier 相对典型值的放大倍数。
  - 点击其他柱子可将 outlier 移至对应位置。
  - 点击「换一组数」重新生成随机样本。
- 应该看到: outlier 倍数越大, 「整张量」行的红色色块越多, 而「block」行仅有 outlier 所在的单个 block 出现红格。FP8 需在 outlier 达到十万倍量级时才出现大面积归零; FP4 在 outlier 仅十几倍时即发生归零, 因此 FP4 的 block 大小必须缩小至 16 至 32。右侧读数含义如下:
  - 两行「归零元素」: 分别统计整张量缩放与 block 缩放下被量化为 0 的元素个数。
  - 「平均相对误差」: 两种方案下全部元素相对误差的均值, 当 block 方案显著低于整张量方案时显示为绿色。
  - 「每元素位宽」: 将 scale 本身的存储开销分摊至每个元素后的等效位宽, block 越小分摊开销越高 (MXFP4 在 block 32 下为 4.25 bit, NVFP4 在 block 16 下为 4.5 bit)。"
    module="llm_train/m13 · m15"
    run="python -m llm_train.m15_fp4_microscaling.demo"
    :challenge="{
      ask: '先选择 FP8 E4M3 格式: outlier 需要放大到多少倍, 整张量共用 scale 才会导致其余元素开始下溢归零? 再切换至 MXFP4 格式: 为什么 FP4 的 block 大小必须缩小至 16~32?',
      answer: '结论:\n- FP8 E4M3 需在 outlier 达到十万倍量级时才会使其他元素开始归零。\n- FP4 在 outlier 达到十几倍时即出现归零, 因此必须使用很小的 block。\n机制解析:\n- FP8 E4M3 的动态范围较宽: 最大可表示值为 448, 最小非零 subnormal 数为 $2^{-9}$, 两者相差 20 多万倍 (约为 $2^{17.8}$)。因此一旦由 amax 确定 scale, 比 amax 小十几万倍的元素仍可表示。block scaling 在 FP8 上主要是防范极端离群值的保护机制, 而非常规数值下的刚需, DeepSeek-V3 采用 128 的 block 大小即已足够。拖动 outlier 倍数观察「整张量」行可见:\n  - 1 万倍以内几乎不出现红格 (在 m13 的训练消融实验中, 全程采用 E4M3 时整张量 scale 与 block scale 的验证集 loss 分别为 4.64e-4 与 4.61e-4, 基本持平)。\n  - 10 万倍时约有 18% 的元素下溢归零。\n  - 100 万倍时几乎全部元素归零。\n- FP4 E2M1 仅有 $\\pm\\{0.5, 1, 1.5, 2, 3, 4, 6\\}$ 共 7 个非零幅值, 最大值与最小值仅相差 12 倍。由 amax 确定 scale 后, 比 amax 小 12 倍以上的元素均无法表示 (低于图中红色虚线) 并被舍入为 0。常规元素本身存在幅值波动, 只要 outlier 比邻近元素大十几倍, 共享同一 scale 的元素便会大面积归零。因此必须缩小 block 大小, 使组内元素幅值相近并独立确定 scale。其代价是每个 block 需额外存储一个 8 bit 的 scale: block 为 32 时摊至每元素增加 0.25 bit, block 为 16 时增加 0.5 bit, 可从右侧「每元素位宽」读数直接观察。\n- 两种 FP4 格式的差异在于 scale 自身的表示精度:\n  - NVFP4: scale 采用带 3 位尾数的 E4M3 格式, 可将 amax 精确映射至最大值 6。\n  - MXFP4: scale 仅能取 2 的整数次幂, amax/scale 会落在 [4, 8) 区间内; 一旦落入 (6, 8) 子区间, 连 amax 自身也会被截断饱和至 6 (m15 实测中有 42% 的 block 发生了此类饱和截断)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(f, key) in RECIPES" :key="key" type="button" :class="{ active: recipe === key }" @click="recipe = key">{{ f.name }}</button>
        <button type="button" @click="seed++">换一组数</button>
      </div>
      <LabSlider v-model="logBlock" label="block 大小" :min="2" :max="6" :format="(k) => (k === 6 ? '64 (= 整张量)' : 2 ** k)" />
      <LabSlider v-model="logOut" label="outlier 倍数" :min="0" :max="6" :step="0.05" :format="(v) => '×' + fmtNum(Math.round(10 ** v))" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="张量各元素的绝对值 (对数) 与 block scale; outlier 的把手可拖">
      <rect v-for="b in nBlocks" :key="'bg' + b" :x="bx((b - 1) * block)" :width="block * BW" y="6" :height="PLOT" :class="['blk', { alt: b % 2 }]" />
      <g v-for="t in [-4, -2, 0, 2, 4]" :key="t">
        <line :x1="X0" :x2="W" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 4" :y="py(t) + 3" class="tick" text-anchor="end">1e{{ t }}</text>
      </g>
      <rect
        v-for="(v, i) in x" :key="i" :x="bx(i) + 1" :width="BW - 2" :y="py(log10(Math.abs(v)))" :height="Math.max(1, PLOT + 6 - py(log10(Math.abs(v))))"
        :class="['bar', { out: i === outIdx }]" @click="outIdx = i"
      />
      <!-- 每个 block 的 "能表示的最小非零值" 线: 低于它的元素会变 0 -->
      <line v-for="(f, b) in blockRes.floors" :key="'fl' + b" :x1="bx(b * block)" :x2="bx((b + 1) * block)" :y1="py(log10(f))" :y2="py(log10(f))" class="floor" />
      <circle
        :cx="bx(outIdx) + BW / 2" :cy="py(log10(Math.abs(x[outIdx])))" r="7" class="handle draggable"
        tabindex="0" role="slider" aria-label="outlier: 上下方向键改大小, 左右方向键换位置" :aria-valuenow="logOut"
        :aria-valuetext="`第 ${outIdx} 个元素, 基准的 ${Math.round(10 ** logOut)} 倍`"
        @keydown.left.prevent="outIdx = Math.max(0, outIdx - 1)" @keydown.right.prevent="outIdx = Math.min(NEL - 1, outIdx + 1)"
        @pointerdown="start($event, { svg, onMove: ({ y }) => (logOut = clamp(Math.round((pyInv(y) - log10(BASE)) * 20) / 20, 0, 6)) })"
        @keydown.up.prevent="logOut = Math.min(6, logOut + 0.25)" @keydown.down.prevent="logOut = Math.max(0, logOut - 0.25)"
      />
      <g v-for="(row, ri) in [tensorRes, blockRes]" :key="ri">
        <text :x="X0 - 4" :y="PLOT + 30 + ri * 22" class="tick" text-anchor="end">{{ ri ? 'block' : '整张量' }}</text>
        <rect
          v-for="(e, i) in row.err" :key="i" :x="bx(i) + 0.5" :width="BW - 1" :y="PLOT + 18 + ri * 22" height="16" rx="1.5"
          :fill="e >= 1 ? 'var(--danger)' : heat(e * 2.5, 'var(--warn)')" class="errc"
        ><title>#{{ i }}: x = {{ x[i].toExponential(2) }}, Q(x) = {{ row.q[i].toExponential(2) }}, 误差 {{ (e * 100).toFixed(0) }}%</title></rect>
      </g>
    </svg>
    <p class="lab-note">
      下面两行色块的图例: <i class="sw z" /> 被冲成 0 (误差 100%) <i class="sw e" /> 有误差, 颜色越深误差越大 <i class="sw f" /> 上图的红色虚线 = 该 block 的 scale 下能表示的最小非零值的一半, 柱子低于它就归零。
    </p>

    <template #stats>
      <div class="kv"><span>整张量共用一个 scale: 被冲成 0 的元素</span><b :class="tensorRes.zeros ? 'bad' : 'good'">{{ tensorRes.zeros }} / 64</b></div>
      <div class="kv"><span>每 block 一个 scale: 被冲成 0 的元素</span><b :class="blockRes.zeros ? 'bad' : 'good'">{{ blockRes.zeros }} / 64</b></div>
      <div class="kv"><span>64 个元素的平均相对误差 (整张量 → block)</span><b :class="{ good: blockRes.mean < tensorRes.mean * 0.7 }">{{ pct(tensorRes.mean) }} → {{ pct(blockRes.mean) }}</b></div>
      <div class="kv"><span>每元素位宽 (把每个 block 的 scale 摊进去)</span><b>{{ (R.bits + R.scaleBits / block).toFixed(2) }} bit</b></div>
      <p class="lab-note">★ 当前格式的 scale 怎么定: <Tex :text="R.note" /></p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, fmtNum, heat, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const NEL = 64, BASE = 0.02
// 与 core/numerics.py:fake_quant_float 同一个算法: 饱和 → normal 区按尾数舍入 → subnormal 区固定步长
const FLOAT = { e4m3: [3, 448, -6], e2m1: [1, 6, 0] }
const fq = (v, fmt) => {
  const [m, max, eMin] = FLOAT[fmt]
  const a = Math.min(Math.abs(v), max)
  if (a === 0) return 0
  const e = Math.max(Math.floor(Math.log2(a)), eMin)
  const q = Math.round(a / 2 ** (e - m)) * 2 ** (e - m)
  return Math.sign(v) * q
}
const RECIPES = {
  fp8: { name: 'FP8 E4M3', fmt: 'e4m3', bits: 8, scaleBits: 32, scale: (amax) => amax / 448, note: '$\\text{scale} = \\text{amax} / 448$ (高精度存)。E4M3 自带 $2^{15}$ 以上的动态范围: 没有极端 outlier 时整张量 scale 就够用, DeepSeek-V3 的 block 128 是为 outlier 买的保险。' },
  mxfp4: { name: 'MXFP4 (E8M0 scale)', fmt: 'e2m1', bits: 4, scaleBits: 8, scale: (amax) => 2 ** (Math.floor(Math.log2(amax)) - 2), note: '$\\text{scale} = 2^{\\lfloor \\log_2 \\text{amax} \\rfloor - 2}$, 只能是 2 的幂: $\\text{amax}/\\text{scale}$ 落在 $[4, 8)$, 大于 6 的会被饱和。标准 block = 32。' },
  nvfp4: { name: 'NVFP4 (E4M3 scale)', fmt: 'e2m1', bits: 4, scaleBits: 8, scale: (amax, tmax) => { const ts = tmax / (448 * 6); return fq(amax / 6 / ts, 'e4m3') * ts }, note: 'block scale 本身是 E4M3 (带 3 位尾数), 再乘一个整张量的 FP32 scale: amax 几乎精确贴到 6。标准 block = 16。' },
}
const recipe = ref('mxfp4'), logBlock = ref(4), logOut = ref(1.5), outIdx = ref(21), seed = ref(3), svg = ref(null)
const { start } = useDrag()
const R = computed(() => RECIPES[recipe.value])
const block = computed(() => 2 ** logBlock.value)
const nBlocks = computed(() => NEL / block.value)
const log10 = Math.log10

const x = computed(() => {
  const rand = mulberry32(seed.value)
  return range(NEL).map((i) => (i === outIdx.value ? BASE * 10 ** logOut.value : randn(rand) * BASE || BASE))
})

// ★ 同一个量化器, 只是 "谁和谁共享 amax" 不同
const quantize = (xs, B, r) => {
  const tmax = Math.max(...xs.map(Math.abs))
  const q = [], floors = []
  for (let b0 = 0; b0 < xs.length; b0 += B) {
    const blk = xs.slice(b0, b0 + B)
    const s = r.scale(Math.max(...blk.map(Math.abs)), tmax) || 1
    floors.push(s * 2 ** (FLOAT[r.fmt][2] - FLOAT[r.fmt][0]) / 2)
    blk.forEach((v) => q.push(fq(v / s, r.fmt) * s))
  }
  const err = xs.map((v, i) => Math.min(1, Math.abs(q[i] - v) / Math.abs(v)))
  return { q, err, floors, zeros: q.filter((v) => v === 0).length, mean: sum(err) / err.length }
}
const tensorRes = computed(() => quantize(x.value, NEL, R.value))
const blockRes = computed(() => quantize(x.value, block.value, R.value))

const W = 640, PLOT = 170, H = PLOT + 66, X0 = 44, BW = (W - X0) / NEL
const LO = -5, HI = 4.6
const bx = (i) => X0 + i * BW
const py = (lg) => 6 + (1 - (clamp(lg, LO, HI) - LO) / (HI - LO)) * PLOT
const pyInv = (yy) => LO + (1 - (yy - 6) / PLOT) * (HI - LO)
const pct = (v) => (v * 100).toFixed(1) + '%'
</script>

<style scoped>
.blk { fill: transparent; }
.blk.alt { fill: var(--accent-soft); }
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.bar { fill: var(--text-dim); cursor: pointer; outline: none; }
.bar:hover { fill: var(--accent); }
.bar.out { fill: var(--eye); }
.floor { stroke: var(--danger); stroke-dasharray: 3 2; stroke-width: 1.5; }
.handle { fill: var(--eye); stroke: var(--bg-card); stroke-width: 2; }
.errc { stroke: var(--border); stroke-width: 0.5; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 3px 0 8px; vertical-align: -1px; }
.sw:first-child { margin-left: 0; }
.sw.z { background: var(--danger); }
.sw.e { background: color-mix(in srgb, var(--warn) 60%, transparent); }
.sw.f { height: 0; border-top: 2px dashed var(--danger); vertical-align: 3px; }
.handle:focus-visible { stroke: var(--text); }
</style>
