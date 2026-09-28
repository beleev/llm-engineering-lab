<!--
  INT4 / AWQ 实验台 (对应 llm_infer/m08_quantization/int4_awq.py:awq_quantize)。
  一件事: 量化要最小化的是输出误差 ‖XW − XŴ‖ 而不是 ‖W − Ŵ‖ —— 激活大的输入通道, 权重误差被放大得最厉害, 应该优先保护。
-->
<template>
  <LabFrame
    title="INT4 权重量化 — RTN vs AWQ 激活感知缩放"
    sub="$W$ 是 32×8 的权重, $X$ 是 64 条校准激活, 其中一个输入通道是离群通道。
      - 图 ①: 每个输入通道的激活幅度 $\mathrm{mean}|x|$。拖虚线框里那根柱子改离群幅度, 点别的柱子把离群通道搬过去。
      - 图 ② ③: 输出误差随 $\alpha$ 的变化 (点圆点选 $\alpha$), 和每个输入通道对输出误差的贡献。"
    module="llm_infer/m08"
    run="python -m llm_infer.m08_quantization.demo"
    :challenge="{
      ask: 'α = 0 就是 RTN。先猜: α 一路拖到 1.0, 误差是单调下降吗?',
      answer: '不是单调的, 是 U 形。\n- 收益: $XW = (X/s) \\cdot (s \\odot W)$。把离群通道那一行权重放大 $s$ 倍再量化, 它的相对舍入误差缩小约 $s$ 倍。\n- 陪绑: 放大同时撑大了所在 group 的 min/max 范围, 同组其它行的格点变粗。\n$\\alpha$ 太大时陪绑的损失超过收益, 所以 $\\alpha$ 要在校准集上网格搜索。真实 demo: INT4 g32 RTN 0.0759 → AWQ 0.0273, 最优 $\\alpha=0.4$。\n把离群幅度拖到 1× (没有离群通道): 各通道 $s \\approx 1$, 曲线几乎是平的, AWQ 无事可做。\n把 group 从 32 调到 4: group 越小, 离群行只污染更少的邻居, RTN 本身就变好, AWQ 的相对收益缩小。代价是 scale/zero 的开销: $4 + 32/g$ bit/权重。',
    }"
  >
    <template #controls>
      <LabSlider v-model="bits" label="比特数" :min="2" :max="8" unit=" bit" />
      <LabSlider v-model="gi" label="group 大小 g" :min="0" :max="3" :format="(i) => GS[i]" />
      <LabSlider v-model="ak" label="AWQ 指数 α" :min="0" :max="10" :format="(k) => (k / 10).toFixed(1)" />
      <div class="row"><button type="button" @click="seed++">换一组 W / X</button><button type="button" @click="ak = best">跳到最优 α</button></div>
    </template>

    <svg ref="svg" viewBox="0 0 560 330" role="group" aria-label="激活幅度、误差曲线与逐通道误差贡献">
      <text x="4" y="10" class="cap">① 激活幅度 mean|x_i|, 离群通道 ×{{ mag }} (拖虚线框里的柱 / 点其它柱)</text>
      <g v-for="(a, i) in m.act" :key="`a${i}`">
        <rect v-if="i !== oc" :x="bx(i)" y="14" :width="BW - 3" height="92" class="hit" @click="oc = i" />
        <rect :x="bx(i)" :y="106 - ah(a)" :width="BW - 3" :height="ah(a)" :class="['abar', { out: i === oc }]" pointer-events="none" />
      </g>
      <!-- 图 ① 只有这一个 Tab 停靠点: 上下键改幅度, 左右键把离群通道搬到邻居 -->
      <rect :x="bx(oc) - 2" y="14" :width="BW + 1" height="92" class="grab draggable" tabindex="0" role="slider"
        :aria-label="`离群通道 ${oc} 的幅度; 左右键换通道`" :aria-valuenow="mag" aria-valuemin="1" aria-valuemax="20"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.up.prevent="mag = clamp(mag + 0.5, 1, 20)" @keydown.down.prevent="mag = clamp(mag - 0.5, 1, 20)"
        @keydown.left.prevent="oc = clamp(oc - 1, 0, DI - 1)" @keydown.right.prevent="oc = clamp(oc + 1, 0, DI - 1)" />

      <text x="4" y="130" class="cap">② 输出误差 ‖XW − XŴ‖ / ‖XW‖ 随 α (0 = RTN)</text>
      <line x1="30" x2="550" y1="215" y2="215" class="axis" />
      <polyline :points="m.errs.map((e, k) => `${ex(k)},${ey(e)}`).join(' ')" class="curve" />
      <g v-for="(e, k) in m.errs" :key="`e${k}`" class="pt" @click="ak = k">
        <circle :cx="ex(k)" :cy="ey(e)" r="10" class="hit" />
        <circle :cx="ex(k)" :cy="ey(e)" :r="k === ak ? 6 : 3.5" :class="{ cur: k === ak, best: k === best }" />
        <text :x="ex(k)" y="227" class="tick">{{ (k / 10).toFixed(1) }}</text>
      </g>
      <text :x="ex(best)" :y="ey(m.errs[best]) - 9" class="tick good">最优</text>

      <text x="4" y="248" class="cap">③ 通道 i 对输出误差的贡献 ‖x_i‖·‖ΔW_i‖ (灰 = RTN, 绿 = 当前 α)</text>
      <g v-for="i in DI" :key="`c${i}`">
        <rect :x="bx(i - 1)" :y="322 - ch(m.rtnC[i - 1])" :width="(BW - 3) / 2" :height="ch(m.rtnC[i - 1])" class="cr" />
        <rect :x="bx(i - 1) + (BW - 3) / 2" :y="322 - ch(m.awqC[i - 1])" :width="(BW - 3) / 2" :height="ch(m.awqC[i - 1])" class="ca" />
      </g>
    </svg>

    <template #stats>
      <div class="kv">
        <span>误差: RTN → AWQ (<Tex text="$\alpha$" /> = {{ (ak / 10).toFixed(1) }})</span>
        <b :class="m.errs[ak] < m.errs[0] ? 'good' : m.errs[ak] > m.errs[0] ? 'bad' : ''">{{ m.errs[0].toFixed(4) }} → {{ m.errs[ak].toFixed(4) }}</b>
      </div>
      <div class="kv"><span>最优 <Tex text="$\alpha$" /> / 误差降到</span><b>{{ (best / 10).toFixed(1) }} / {{ (m.errs[best] / m.errs[0] * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>等效 bit / 权重 (含 fp16 scale+zero)</span><b>{{ (bits + 32 / GS[gi]).toFixed(2) }}</b></div>
      <div class="kv"><span>离群通道的 <Tex text="$s_i$" /></span><b>{{ m.s[oc].toFixed(2) }}</b></div>
      <div class="lab-note">
        <p><Tex text="★ $\hat{W} = Q(s \odot W) / s$, $s_i = \mathrm{mean}|x_i|^\alpha$ (再按 $\sqrt{s_{\max} \cdot s_{\min}}$ 归一)。$1/s$ 离线折进上一层, 推理零开销。" /></p>
        <p><Tex text="量化器与 Python 相同: 每个输出通道内, 每 $g$ 个相邻输入通道共用 min/max (非对称 RTN)。$g=128$ 时等效 4.25 bit。" /></p>
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
import { argmax, clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const DI = 32, DO = 8, N = 64, GS = [4, 8, 16, 32], BW = 17
const bits = ref(4), gi = ref(3), ak = ref(4), mag = ref(10), oc = ref(5), seed = ref(1)
const svg = ref(null)
const { start } = useDrag()

// 基础数据只随 seed 变; 离群通道 = 把那一列激活乘 mag (拖动时其它数不跳)
const base = computed(() => {
  const r = mulberry32(seed.value * 131 + 1)
  return { W: range(DI).map(() => range(DO).map(() => randn(r) * 0.3)), X: range(N).map(() => range(DI).map(() => randn(r))) }
})

// 非对称 group-wise RTN, 沿输入维分组 (= quantize_groupwise); 忽略 Python 里 scale 存 fp16 的那点舍入
const quantDequant = (M, b, g) => {
  const out = M.map((row) => row.slice()), L = 2 ** b - 1
  for (let j = 0; j < DO; j++) for (let s0 = 0; s0 < DI; s0 += g) {
    const col = range(g).map((k) => M[s0 + k][j]), lo = Math.min(...col)
    const sc = Math.max((Math.max(...col) - lo) / L, 1e-4)
    col.forEach((v, k) => { out[s0 + k][j] = clamp(Math.round((v - lo) / sc), 0, L) * sc + lo })
  }
  return out
}

const m = computed(() => {
  const { W } = base.value, g = GS[gi.value]
  const X = base.value.X.map((x) => x.map((v, i) => (i === oc.value ? v * mag.value : v)))
  const act = range(DI).map((i) => X.reduce((a, x) => a + Math.abs(x[i]), 0) / N)
  const xn = range(DI).map((i) => Math.hypot(...X.map((x) => x[i])))
  const Y = X.map((x) => range(DO).map((j) => x.reduce((a, v, i) => a + v * W[i][j], 0)))
  const ny = Math.hypot(...Y.flat())
  const at = (k) => {
    let s = act.map((a) => a ** (k / 10))
    const c = Math.sqrt(Math.max(...s) * Math.min(...s)); s = s.map((v) => v / c)
    const Wh = quantDequant(W.map((row, i) => row.map((v) => v * s[i])), bits.value, g).map((row, i) => row.map((v) => v / s[i])) // ★
    const dW = W.map((row, i) => row.map((v, j) => v - Wh[i][j]))
    let e = 0
    for (const x of X) for (let j = 0; j < DO; j++) { let d = 0; for (let i = 0; i < DI; i++) d += x[i] * dW[i][j]; e += d * d }
    return { err: Math.sqrt(e) / ny, contrib: dW.map((row, i) => xn[i] * Math.hypot(...row)), s }
  }
  const all = range(11).map(at)
  return { act, errs: all.map((a) => a.err), rtnC: all[0].contrib, awqC: all[ak.value].contrib, s: all[ak.value].s }
})
const best = computed(() => argmax(m.value.errs.map((e) => -e)))

const bx = (i) => 8 + i * BW
const ah = (a) => clamp(a / 17, 0, 1) * 90
const onMove = ({ y }) => { mag.value = clamp(Math.round(((106 - y) / 90) * 17 / 0.8 * 2) / 2, 1, 20) }  // E|N(0,1)| ≈ 0.8
const eMax = computed(() => Math.max(...m.value.errs) * 1.15)
const ex = (k) => 40 + k * 50, ey = (e) => 215 - (e / eMax.value) * 75
const cMax = computed(() => Math.max(...m.value.rtnC, ...m.value.awqC))
const ch = (c) => (c / cMax.value) * 66
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 10px; fill: var(--text-muted); }
.axis { stroke: var(--border-strong); }
.hit { fill: transparent; cursor: pointer; }
.hit:hover { fill: var(--accent-soft); }
.abar { fill: var(--accent); opacity: 0.7; }
.abar.out { fill: var(--warn); opacity: 1; }
.grab { fill: transparent; stroke: var(--warn); stroke-dasharray: 3 3; stroke-width: 1; outline: none; }
.grab:focus-visible { stroke-width: 2.5; stroke-dasharray: none; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.pt { cursor: pointer; }
.pt circle:not(.hit) { fill: var(--bg-elev); stroke: var(--accent); stroke-width: 1.5; }
.pt circle.best { stroke: var(--left); fill: var(--left); }
.pt circle.cur { fill: var(--accent); stroke: var(--text); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.tick.good { fill: var(--left); }
.cr { fill: var(--text-dim); opacity: 0.6; }
.ca { fill: var(--left); }
</style>
