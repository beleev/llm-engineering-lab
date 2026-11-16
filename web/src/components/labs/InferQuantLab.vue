<!--
  INT4 / AWQ 实验台 (对应 llm_infer/m08_quantization/int4_awq.py:awq_quantize)。
  一件事: 量化要最小化的是输出误差 ‖XW − XŴ‖ 而不是 ‖W − Ŵ‖ —— 激活大的输入通道, 权重误差被放大得最厉害, 应该优先保护。
-->
<template>
  <LabFrame
    title="INT4 权重量化 — RTN 直接舍入 vs AWQ 先放大重要通道再舍入"
    sub="- 模拟什么: 把一层线性层的权重 $W$ 压成 4 bit 整数存起来 (权重量化), 推理时再还原回来乘, 目的是省显存和 decode 时读权重的带宽。最朴素的做法是 RTN (round-to-nearest): 每 $g$ 个相邻权重共用一个范围, 把每个数直接舍入到最近的格点。AWQ (激活感知权重量化) 多做一步: 量化前把「激活大的输入通道」对应的那一行权重乘以 $s_i$ 放大, 激活那一列同步除以 $s_i$, 乘积 $XW$ 不变, 但放大后那一行的舍入误差相对变小了。这个台子用一个 32×8 的 $W$ 和 64 条校准激活 $X$ (其中一个输入通道是离群通道, 数值特别大) 把这件事算一遍。
      - 图上是什么: 图 ① 一根柱子是一个输入通道的激活平均幅度 $\mathrm{mean}|x_i|$, 带虚线框的那根是离群通道。图 ② 是输出误差 $\|XW - X\hat{W}\| / \|XW\|$ 随 $\alpha$ 变化的曲线, 横轴 $\alpha$ 从 0 到 1, $\alpha = 0$ 就是 RTN; 标「最优」的点误差最低, 实心点是当前选中的 $\alpha$。图 ③ 每个输入通道一对柱子, 高度是这个通道对输出误差的贡献 $\|x_i\| \cdot \|\Delta W_i\|$ (激活越大、这一行权重的误差越大, 贡献越大); 灰柱是 RTN, 绿柱是当前 $\alpha$ 下的 AWQ。
      - 可以做什么: 拖「比特数」改用几比特存; 拖「group 大小 g」改几个相邻权重共用一个 scale; 拖「AWQ 指数 α」或点图 ② 的圆点改 $\alpha$, $s_i = \mathrm{mean}|x_i|^\alpha$, $\alpha$ 越大放大得越狠; 「换一组 W / X」换随机数; 「跳到最优 α」跳到图 ② 的最低点。在图 ① 里上下拖虚线框里的柱子改离群通道的幅度, 点别的柱子把离群通道搬到那一列。
      - 应该看到: 图 ③ 里 RTN (灰) 的误差贡献几乎全来自离群通道那一根。把 $\alpha$ 从 0 拖大, 那一根的绿柱明显变矮, 但其它通道的绿柱会略微长高 (被放大的行撑大了同组的范围, 同组其它行的格点变粗), 所以图 ② 是先降后升的 U 形。右侧「误差: RTN → AWQ」是两个数的直接对比, 降了标绿; 「最优 α / 那里的误差是 RTN 的」是最低点的位置和它相对 RTN 的百分比; 「每个权重实际占的 bit」把每组一个 fp16 的 scale 和一个 fp16 的偏移 zero (共 32 bit) 平摊进去; 「离群通道的放大倍数 s_i」是它当前被放大了几倍。"
    module="llm_infer/m08"
    run="python -m llm_infer.m08_quantization.demo"
    :challenge="{
      ask: 'α = 0 就是 RTN。先猜: α 一路拖到 1.0, 误差是单调下降吗?',
      answer: '不是单调的, 是 U 形: 先降后升, 最低点在中间。\n- 收益从哪来: $XW = (X/s) \\cdot (s \\odot W)$, 把离群通道那一行权重乘以 $s$ 再量化, 激活那一列除以 $s$, 乘积不变。舍入误差的大小只由这一组的 scale 决定, 这一行被放大了 $s$ 倍, 舍入误差相对这一行就缩小了约 $s$ 倍; 而这一行正是输出误差的大头, 因为它乘的是最大的激活。\n- 代价在哪: 这一行和同一个 group 里的另外 $g - 1$ 行共用一个 min/max。它被放大后, 整组的范围跟着被撑大, 同组其它行的格点变粗, 误差变大 (图 ③ 里其它通道的绿柱略微长高)。\n$\\alpha$ 越大 $s$ 越大: 一开始收益大于代价, 误差下降; 过了某一点代价反超, 误差回升。所以 AWQ 在校准集上把 $\\alpha$ 从 0 到 1 网格搜一遍, 取最低点。Python demo (W 256×256, 4 个离群通道 ×30): RTN 0.0759, AWQ $\\alpha = 0.4$ 时 0.0273, $\\alpha = 1$ 时 0.1173, 比 RTN 还差。\n再试两件事:\n- 把图 ① 的离群幅度拖到 1× (没有离群通道): 各通道激活差不多大, $s_i$ 都接近 1, 曲线几乎是平的, AWQ 没有东西可保护。\n- 把 group 从 32 调到 4: group 越小, 离群行只和 3 个邻居共用范围, 污染范围小, RTN 本身就好了很多, AWQ 的相对收益缩小。代价是每组要存一个 fp16 的 scale 和一个 fp16 的 zero, 每个权重平摊 $32/g$ bit: $g = 4$ 时等效 12 bit, 比 INT8 还贵; $g = 32$ 时 5 bit; $g = 128$ 时 4.25 bit。',
    }"
  >
    <template #controls>
      <LabSlider v-model="bits" label="比特数" :min="2" :max="8" unit=" bit" />
      <LabSlider v-model="gi" label="group 大小 g" :min="0" :max="3" :format="(i) => GS[i]" />
      <LabSlider v-model="ak" label="AWQ 指数 α" :min="0" :max="10" :format="(k) => (k / 10).toFixed(1)" />
      <div class="row"><button type="button" @click="seed++">换一组 W / X</button><button type="button" @click="ak = best">跳到最优 α</button></div>
    </template>

    <svg ref="svg" viewBox="0 0 560 330" role="group" aria-label="激活幅度、误差曲线与逐通道误差贡献">
      <text x="4" y="10" class="cap">① 各输入通道的激活幅度 mean|x_i|; 虚线框 = 离群通道 ×{{ mag }} (上下拖它改幅度, 点别的柱子搬过去)</text>
      <g v-for="(a, i) in m.act" :key="`a${i}`">
        <rect v-if="i !== oc" :x="bx(i)" y="14" :width="BW - 3" height="92" class="hit" @click="oc = i" />
        <rect :x="bx(i)" :y="106 - ah(a)" :width="BW - 3" :height="ah(a)" :class="['abar', { out: i === oc }]" pointer-events="none" />
      </g>
      <!-- 图 ① 只有这一个 Tab 停靠点: 上下键改幅度, 左右键把离群通道搬到邻居 -->
      <rect :x="bx(oc) - 2" y="14" :width="BW + 1" height="92" class="grab draggable" tabindex="0" role="slider"
        :aria-label="`离群通道 ${oc} 的激活幅度, 上下键改大小; 左右键把离群通道搬到邻居`" :aria-valuenow="mag" aria-valuemin="1" aria-valuemax="20"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.up.prevent="mag = clamp(mag + 0.5, 1, 20)" @keydown.down.prevent="mag = clamp(mag - 0.5, 1, 20)"
        @keydown.left.prevent="oc = clamp(oc - 1, 0, DI - 1)" @keydown.right.prevent="oc = clamp(oc + 1, 0, DI - 1)" />

      <text x="4" y="130" class="cap">② 输出误差 ‖XW − XŴ‖ / ‖XW‖ 随 α 的变化 (α = 0 就是 RTN; 点圆点选 α)</text>
      <line x1="30" x2="550" y1="215" y2="215" class="axis" />
      <polyline :points="m.errs.map((e, k) => `${ex(k)},${ey(e)}`).join(' ')" class="curve" />
      <g v-for="(e, k) in m.errs" :key="`e${k}`" class="pt" @click="ak = k">
        <circle :cx="ex(k)" :cy="ey(e)" r="10" class="hit" />
        <circle :cx="ex(k)" :cy="ey(e)" :r="k === ak ? 6 : 3.5" :class="{ cur: k === ak, best: k === best }" />
        <text :x="ex(k)" y="227" class="tick">{{ (k / 10).toFixed(1) }}</text>
      </g>
      <text :x="ex(best)" :y="ey(m.errs[best]) - 9" class="tick good">最优</text>

      <text x="4" y="248" class="cap">③ 每个输入通道对输出误差的贡献 ‖x_i‖·‖ΔW_i‖ (灰 = RTN, 绿 = 当前 α 的 AWQ)</text>
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
      <div class="kv"><span>最优 <Tex text="$\alpha$" /> / 那里的误差是 RTN 的</span><b>{{ (best / 10).toFixed(1) }} / {{ (m.errs[best] / m.errs[0] * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>每个权重实际占的 bit (含每组 fp16 scale + zero)</span><b>{{ (bits + 32 / GS[gi]).toFixed(2) }}</b></div>
      <div class="kv"><span>离群通道的放大倍数 <Tex text="$s_i$" /></span><b>{{ m.s[oc].toFixed(2) }}</b></div>
      <div class="lab-note">
        <p><Tex text="★ AWQ 的全部公式: $\hat{W} = Q(s \odot W) / s$, 其中 $Q$ 是普通的 RTN 量化, $s_i = \mathrm{mean}|x_i|^\alpha$ (再按 $\sqrt{s_{\max} \cdot s_{\min}}$ 归一, 让放大和缩小的通道各占一半)。先把每一行乘以 $s_i$, 量化, 再除回来。" /></p>
        <p><Tex text="推理时不用真的除: $1/s$ 离线折进上一层 (RMSNorm 的 gamma 或上一个 Linear), 所以 AWQ 的推理开销为零, 存的还是普通 INT4 格式。" /></p>
        <p><Tex text="量化器与 Python 相同: 每个输出通道内, 每 $g$ 个相邻输入通道共用一组 min/max (非对称 RTN)。每组存 fp16 的 scale 和 zero, 所以每个权重实际占 $\text{bits} + 32/g$ bit, $g = 128$ 时 4.25 bit。" /></p>
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
