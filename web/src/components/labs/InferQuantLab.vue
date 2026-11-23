<!--
  INT4 / AWQ 实验台 (对应 llm_infer/m08_quantization/int4_awq.py:awq_quantize)。
  一件事: 量化要最小化的是输出误差 ‖XW − XŴ‖ 而不是 ‖W − Ŵ‖ —— 激活大的输入通道, 权重误差被放大得最厉害, 应该优先保护。
-->
<template>
  <LabFrame
    title="INT4 权重量化 — RTN 直接舍入 vs AWQ 先放大重要通道再舍入"
    sub="- 模拟什么: 权重量化将线性层权重矩阵 $W$ 压缩为 4 bit 等低位宽整数存储, 推理计算时再反量化还原后参与矩阵乘, 用以降低模型显存占用及 decode 阶段读取权重的访存带宽开销。本实验在一个 32×8 的权重矩阵 $W$ 与 64 条校准激活样本 $X$ (其中一个输入通道为幅值显著偏大的离群通道) 上对比两种量化方法:
  - RTN (round-to-nearest): 基础最近邻舍入量化, 每 $g$ 个相邻权重共享同一量化区间并直接舍入至最近格点。
  - AWQ (激活感知权重量化): 在量化前将高幅值输入通道所对应的那一行权重乘以缩放因子 $s_i$ 放大, 同时将输入激活对应列除以 $s_i$; 数学上乘积 $XW$ 保持不变, 但该行权重放大后再执行舍入量化, 其相对量化误差显著减小。
- 图上是什么: 三幅联动图表:
  - 图 ①: 每根柱子代表一个输入通道在校准集上的激活平均幅值 $\mathrm{mean}|x_i|$, 带虚线框的柱子标示离群通道。
  - 图 ②: 相对输出误差 $\|XW - X\hat{W}\| / \|XW\|$ 随指数 $\alpha$ 变化的曲线, 横轴 $\alpha$ 取值从 0 到 1, 其中 $\alpha = 0$ 即退化为标准 RTN; 标有「最优」的圆点对应误差最低点, 实心圆点代表当前选定的 $\alpha$。
  - 图 ③: 每个输入通道对应一对柱子, 柱高代表该通道对总输出误差的贡献项 $\|x_i\| \cdot \|\Delta W_i\|$ (输入激活幅值越大、对应权重行量化误差越大, 对输出误差的贡献越大); 灰色柱为 RTN, 绿色柱为当前 $\alpha$ 下的 AWQ。
- 可以做什么: 按界面从上到下依次操作, 并可直接交互调整图 ①:
  - 拖动「比特数」修改权重量化位宽。
  - 拖动「group 大小 g」调整共享同一个 scale 的相邻权重数量。
  - 拖动「AWQ 指数 α」或点击图 ② 曲线上的圆点修改 $\alpha$ (缩放系数定义为 $s_i = \mathrm{mean}|x_i|^\alpha$, $\alpha$ 越大对显著通道的放大倍率越高)。
  - 点击「换一组 W / X」重新生成随机权重与校准激活。
  - 点击「跳到最优 α」直接定位至图 ② 误差曲线的最低点。
  - 在图 ① 中上下拖动带虚线框的柱子调整离群通道幅值, 或点击其余柱子将离群通道切换至对应列。
- 应该看到: 图 ③ 中 RTN (灰柱) 的输出误差几乎全部集中在离群通道那一列。将 $\alpha$ 从 0 逐步增大时, 离群通道对应的绿柱显著缩短, 但同组其余普通通道的绿柱会轻微升高 (被放大的显著权重行拉宽了所在分组的动态范围, 使其余行的量化格点变粗), 因此图 ② 的总误差曲线呈先降后升的 U 形。右侧「误差: RTN → AWQ」直接对比两者的相对输出误差, 误差下降时标绿。其余读数含义如下:
  - 「最优 α / 那里的误差是 RTN 的」展示最优指数位置及其相对 RTN 误差的百分比。
  - 「每个权重实际占的 bit」将每组额外存储的一个 fp16 scale 与一个 fp16 零点偏移 zero (合计 32 bit) 分摊计入单权重等效位宽。
  - 「离群通道的放大倍数 s_i」显示当前离群通道对应的缩放倍率。"
    module="llm_infer/m08"
    run="python -m llm_infer.m08_quantization.demo"
    :challenge="{
      ask: '当 $\\alpha = 0$ 时即为标准 RTN 量化。若将 $\\alpha$ 一路拖至 1.0, 输出相对误差会单调下降吗?',
      answer: '并非单调下降, 而是呈先降后升的 U 形曲线, 误差最低点位于区间中部。\n- 收益来源: 利用恒等变换 $XW = (X/s) \\cdot (s \\odot W)$, 将离群通道对应的那一行权重乘以 $s$ 后再量化, 并将输入激活对应列除以 $s$。由于舍入误差的绝对步长取决于整组的 scale, 显著权重行被放大 $s$ 倍后, 其相对舍入误差相应缩小约 $s$ 倍; 该行原本与最大幅值的输入激活相乘、占据总输出误差的主要部分, 保护该行即可显著压低总误差。\n- 伴随代价: 该行权重与同一 group 内的其余 $g - 1$ 个权重共享同一组 min/max 区间。当该行被过度放大时, 整组的动态范围随之被拉宽, 导致同组其余普通权重的量化格点变粗、量化误差上升 (表现为图 ③ 中其余通道的绿柱轻微升高)。\n随着 $\\alpha$ 增大、缩放倍率 $s$ 提高, 初期保护显著通道的收益大于连带损失, 总误差快速下降; 越过平衡点后同组普通通道受损的代价反超, 总误差重新回升。因此 AWQ 在工程实践中会在校准集上对 $\\alpha \\in [0, 1]$ 执行网格搜索以选取误差极小点。在 Python demo (权重维度 256×256, 含 4 个放大 30 倍的离群通道) 中: 标准 RTN 相对误差为 0.0759, AWQ 在 $\\alpha = 0.4$ 时降至 0.0273, 而在 $\\alpha = 1$ 时反而恶化至 0.1173, 劣于标准 RTN。\n还可进一步观察以下两组对照现象:\n- 将图 ① 离群通道幅度拖至 1× (即不存在离群通道): 各输入通道激活幅值相近, 所有 $s_i$ 均接近 1, 误差曲线趋于平坦, AWQ 失去需要重点保护的显著通道。\n- 将 group 大小从 32 调小至 4: 分组越细, 显著权重仅与组内其余 3 个相邻元素共享区间, 受波及范围收窄, RTN 自身的基准误差已大幅降低, AWQ 的相对增益随之缩小。但缩小分组的代价是每组均需额外存储 fp16 的 scale 与 zero (共 32 bit), 分摊至单个权重的元数据开销为 $32/g$ bit:\n  - $g = 4$ 时单权重等效占 12 bit, 存储开销甚至高于 INT8。\n  - $g = 32$ 时等效占 5 bit。\n  - $g = 128$ 时等效占 4.25 bit。',
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
        <p><Tex text="★ AWQ 核心公式: $\hat{W} = Q(s \odot W) / s$, 其中 $Q$ 表示常规 RTN 量化算子, 缩放因子定义为 $s_i = \mathrm{mean}|x_i|^\alpha$ (再除以 $\sqrt{s_{\max} \cdot s_{\min}}$ 归一化, 使放大与缩小的通道围绕 1 对称分布)。具体步骤:" /></p>
        <ol class="pts">
          <li><Tex text="先把每一行乘以 $s_i$。" /></li>
          <li>量化。</li>
          <li>再除回来。</li>
        </ol>
        <p><Tex text="实际部署推理时无需在线执行除法: 缩放因子 $1/s$ 可离线融合进前驱算子 (如 RMSNorm 的缩放参数 gamma 或前一层 Linear 权重), 因此 AWQ 在推理阶段不引入额外计算开销, 权重仍按标准 INT4 格式存储。" /></p>
        <p><Tex text="前端量化器实现与 Python demo 一致: 在每个输出通道内部, 每 $g$ 个相邻输入通道共享一组 min/max 区间并执行非对称 RTN 量化。由于每组需额外存储 fp16 精度的 scale 与 zero, 单个权重实际占用位宽为 $\text{bits} + 32/g$ bit, 当 $g = 128$ 时等效为 4.25 bit。" /></p>
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
