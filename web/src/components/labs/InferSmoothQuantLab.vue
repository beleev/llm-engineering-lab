<!--
  SmoothQuant 实验台 (对应 llm_infer/m25_weight_quant/smoothquant.py:smooth_scales / w8a8_matmul)。
  一件事: 同一个缩放恒等式 XW = (X/s)(s⊙W), 用来把激活的离群值"挪"给权重, 让 W8A8 两边都好量化。
-->
<template>
  <LabFrame
    title="SmoothQuant — 把离群值从激活挪给权重"
    sub="W8A8: 激活按 token 量化成 INT8, 权重按输出通道量化成 INT8。激活有 4 个输入通道放大了 50 倍。
      - 图 ①: 输出误差随 $\alpha$ 的三条曲线 (Python 实测)。点圆点或拖滑杆选 $\alpha$。
      - 图 ②: 8 个示意通道平滑后的 $\max|X_j|$ 和 $\max|W_j|$ (对数高度), 按 $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$ 现算。"
    module="llm_infer/m25"
    run="python -m llm_infer.m25_weight_quant.demo"
    :challenge="{
      ask: '把 α 拖到 1, 离群值全部挪给权重。W8A8 误差会比不平滑的 0.0278 更低吗?',
      answer: '不会: 0.0297, 比不平滑还差。最低点在 $\\alpha=0.5$。\n$\\alpha=0.5$ 时图 ② 每个通道的两根柱子一样高: $\\max|X_j|/s_j = \\max|W_j| \\cdot s_j = \\sqrt{\\max|X_j| \\cdot \\max|W_j|}$。难度在激活和权重之间平分。\n- $\\alpha=0.5$: W8A8 误差 0.0070, 比不平滑的 0.0278 降 4.0×。\n- $\\alpha=1$: 离群值整个压到权重上, 权重误差 0.0288, W8A8 0.0297, 比不平滑还差。\n和 AWQ 恒等式相同, 目的相反:\n- AWQ: 只量化权重, 放大重要通道是为了保护权重。\n- SmoothQuant: 要让激活也能 INT8。',
    }"
  >
    <template #controls>
      <LabSlider v-model="ai" label="迁移强度 α" :min="-1" :max="10" :format="(k) => (k < 0 ? '不平滑' : (k / 10).toFixed(1))" />
      <div class="row"><button type="button" @click="ai = 5">跳到最优 α = 0.5</button><button type="button" @click="seed++">换一组通道</button></div>
    </template>

    <svg viewBox="0 0 560 330" role="group" aria-label="误差随 α 的曲线与逐通道幅度">
      <text x="4" y="12" class="cap">① 实测, 不随「换一组通道」变化: 输出相对误差 (不平滑时 W8A8 = 0.0278, 虚线)</text>
      <line x1="44" x2="550" :y1="ey(NONE.xw)" :y2="ey(NONE.xw)" class="none" />
      <g v-for="s in SERIES" :key="s.key">
        <polyline :points="s.data.map((v, k) => `${ex(k)},${ey(v)}`).join(' ')" class="line" :class="s.key" />
        <g v-for="(v, k) in s.data" :key="k" class="pt" @click="ai = k">
          <circle :cx="ex(k)" :cy="ey(v)" r="8" class="hit" />
          <circle :cx="ex(k)" :cy="ey(v)" :r="k === ai ? 4.5 : 2.5" :class="s.key" />
        </g>
      </g>
      <line v-if="ai >= 0" :x1="ex(ai)" :x2="ex(ai)" y1="18" y2="150" class="cursor" />
      <text v-for="k in 11" :key="`t${k}`" :x="ex(k - 1)" y="162" class="tick">{{ ((k - 1) / 10).toFixed(1) }}</text>
      <text v-for="(s, j) in SERIES" :key="`l${j}`" :x="60 + j * 110" y="176" class="tick start" :class="s.key">● {{ s.name }}</text>

      <text x="4" y="198" class="cap">② 前端现算, ch3 放大 50 倍: 左 = max|X_j| / s_j, 右 = max|W_j| · s_j (对数高度)</text>
      <g v-for="(c, j) in ch.cols" :key="`c${j}`">
        <rect :x="cx(j)" :y="hy(c.x)" width="22" :height="316 - hy(c.x)" class="xb" :class="{ out: OUT.includes(j) }" />
        <rect :x="cx(j) + 24" :y="hy(c.w)" width="22" :height="316 - hy(c.w)" class="wb" />
        <text :x="cx(j) + 23" y="328" class="tick">ch{{ j }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>α = {{ ai < 0 ? '不平滑' : (ai / 10).toFixed(1) }}: 只量化 X</span><b>{{ row.x.toFixed(4) }}</b></div>
      <div class="kv"><span>只量化 W</span><b :class="row.w > 0.02 ? 'bad' : ''">{{ row.w.toFixed(4) }}</b></div>
      <div class="kv"><span>W8A8 (相对不平滑)</span><b :class="row.xw < NONE.xw * 0.5 ? 'good' : row.xw > NONE.xw ? 'bad' : ''">{{ row.xw.toFixed(4) }} ({{ (NONE.xw / row.xw).toFixed(1) }}×)</b></div>
      <div class="kv"><span>② 最大 ÷ 中位: 激活 / 权重</span><b :class="Math.max(ch.xr, ch.wr) > 10 ? 'bad' : 'good'">{{ ch.xr.toFixed(1) }} / {{ ch.wr.toFixed(1) }}</b></div>
      <div class="lab-note">
        <p>① 来自 demo 的 [B] 段 (W 256×256, 离群输入通道 [7, 50, 131, 200] ×50)。② 是 8 个示意通道, 按 smooth_scales 的公式现算, 不参与 ① 的误差。</p>
        <p>② 的比值: 激活那个是 per-token 量化的难度, 权重那个是 per-channel 量化的难度。哪个超过 10 都会标红。</p>
        <p>同一个 demo 里 GPTQ 走另一条路: 不挪幅度, 而是把舍入误差补给后面的行 (相关激活下 0.0776 → 0.0369)。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range } from '@/utils/labmath.js'

// 数据: python -m llm_infer.m25_weight_quant.demo 的 [B] 段, α = 0, 0.1, …, 1 共 11 行 + 不平滑
const XO = [0.0307, 0.0207, 0.0139, 0.0096, 0.0068, 0.0051, 0.0041, 0.0037, 0.0037, 0.0049, 0.0074]
const WO = [0.0072, 0.0051, 0.0038, 0.0036, 0.004, 0.0048, 0.0065, 0.009, 0.0131, 0.0192, 0.0288]
const XW = [0.0315, 0.0213, 0.0144, 0.0103, 0.0078, 0.007, 0.0076, 0.0097, 0.0136, 0.0199, 0.0297]
const NONE = { x: 0.027, w: 0.0067, xw: 0.0278 }
const SERIES = [
  { key: 'xo', name: '只量化 X', data: XO },
  { key: 'wo', name: '只量化 W', data: WO },
  { key: 'xw', name: 'W8A8', data: XW },
]
const OUT = [3], MAG = 50   // 示意通道里的离群通道和放大倍数, 倍数与 ① 的 demo 配置相同

const ai = ref(-1), seed = ref(1)   // ai = −1 表示不平滑
const row = computed(() => (ai.value < 0 ? NONE : { x: XO[ai.value], w: WO[ai.value], xw: XW[ai.value] }))

// 8 个示意通道: 激活幅度 ~1, 权重行幅度 ~0.1; 第 3 个通道放大 MAG 倍
const base = computed(() => {
  const r = mulberry32(seed.value * 977)
  return range(8).map((j) => ({ ax: (0.7 + 0.6 * r()) * (OUT.includes(j) ? MAG : 1), aw: 0.06 + 0.08 * r() }))
})
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return (s[3] + s[4]) / 2 }
const ch = computed(() => {
  const a = ai.value < 0 ? null : ai.value / 10
  const cols = base.value.map(({ ax, aw }) => {
    const s = a === null ? 1 : ax ** a / aw ** (1 - a)          // ★ s_j = max|X_j|^α / max|W_j|^(1−α)
    return { x: ax / s, w: aw * s }
  })
  const xs = cols.map((c) => c.x), ws = cols.map((c) => c.w)
  return { cols, xr: Math.max(...xs) / median(xs), wr: Math.max(...ws) / median(ws) }
})

const ex = (k) => 60 + k * 48
const ey = (v) => 150 - (v / 0.032) * 130
const cx = (j) => 30 + j * 66
const hy = (v) => 316 - ((Math.log10(v) + 2) / 4.5) * 110   // 对数轴: 0.01 … 10^2.5
</script>

<style scoped>
svg { min-width: 540px; }
.cap { font-size: 10px; fill: var(--text-muted); }
.none { stroke: var(--text-muted); stroke-dasharray: 5 4; }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.tick.start { text-anchor: start; }
polyline.line { fill: none; stroke-width: 2; }
.xo { stroke: var(--eye); fill: var(--eye); }
.wo { stroke: var(--right); fill: var(--right); }
.xw { stroke: var(--accent); fill: var(--accent); }
.line.xw { stroke-width: 3; }
.pt { cursor: pointer; }
.hit { fill: transparent; stroke: none; }
.cursor { stroke: var(--warn); stroke-dasharray: 3 3; }
.xb { fill: var(--eye); opacity: 0.75; }
.xb.out { fill: var(--warn); opacity: 1; }
.wb { fill: var(--right); opacity: 0.75; }
</style>
