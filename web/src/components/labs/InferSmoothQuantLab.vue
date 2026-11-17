<!--
  SmoothQuant 实验台 (对应 llm_infer/m25_weight_quant/smoothquant.py:smooth_scales / w8a8_matmul)。
  一件事: 同一个缩放恒等式 XW = (X/s)(s⊙W), 用来把激活的离群值"挪"给权重, 让 W8A8 两边都好量化。
-->
<template>
  <LabFrame
    title="SmoothQuant — 把激活里的离群值挪一部分给权重, 让两边都好量化"
    sub="- 模拟什么: W8A8 量化: 权重 $W$ 和激活 $X$ 都压成 INT8, 矩阵乘就能走 INT8 运算单元, 比只量化权重更省算力。权重按输出通道一组、激活按 token (一行) 一组来量化。麻烦在激活: 真实 LLM 的激活里有少数几个通道 (这里是 4 个, 放大了 50 倍) 在每个 token 上都特别大, 每一行都含它们, 每一行的量化范围都被撑大, 其它通道的精度一起变差。SmoothQuant 用恒等式 $XW = (X/s)(s \odot W)$, 等于把离群值从激活「挪」给权重:
        - 把激活的第 $j$ 个通道除以 $s_j$ 压平。
        - 权重的第 $j$ 行乘以 $s_j$ 补回来。
        - $\alpha$ 决定挪多少, $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$, $\alpha$ 越大挪得越多。
      - 图上是什么: 两张图:
        - 图 ①: Python demo 实测的三条误差曲线, 横轴是 $\alpha$ 从 0 到 1。「只量化 X」(权重保持浮点)、「只量化 W」(激活保持浮点)、「W8A8」(两边都量化, 粗线)。虚线是不做平滑直接 W8A8 的误差 0.0278, 作为基准。
        - 图 ②: 8 个示意通道, 每个通道一对柱子。左边是平滑后激活这一列的最大值 $\max|X_j| / s_j$, 右边是平滑后权重这一行的最大值 $\max|W_j| \cdot s_j$, 高度是对数刻度; ch3 是放大 50 倍的离群通道, 用另一种颜色标出。这 8 个通道是前端按公式现算的示意, 不参与图 ① 的误差。
      - 可以做什么: 三个控件:
        - 拖「迁移强度 α」或点图 ① 的圆点选 $\alpha$, 最左一档是「不平滑」。
        - 「跳到最优 α = 0.5」直接跳到最低点。
        - 「换一组通道」重抽图 ② 的 8 个示意通道 (图 ① 不变)。
      - 应该看到: 右侧前三个误差读数来自图 ①, 「W8A8」后面的括号是它比不平滑好了几倍; 「最大 ÷ 中位」是图 ② 里两侧各自的最大值除以中位数, 哪边超过 10 就标红, 代表那边有离群值、难量化。把 $\alpha$ 从左往右拖:
        - 不平滑时: 图 ② 的 ch3 左柱 (激活) 远高于别的通道, 右侧「最大 ÷ 中位: 激活」很大, 说明激活这一侧难量化。
        - $\alpha$ 往右拖: ch3 的左柱降、右柱升, 离群值在往权重那边挪。
        - 到 $\alpha = 1$ 时: 换成权重那根柱子远高于其它通道, 「只量化 W」的误差飙升。
        - $\alpha = 0.5$ 时: 每个通道左右两根柱子一样高, 难度被平分, 图 ① 的 W8A8 曲线也在这里最低。"
    module="llm_infer/m25"
    run="python -m llm_infer.m25_weight_quant.demo"
    :challenge="{
      ask: '把 α 拖到 1, 离群值全部挪给权重。W8A8 误差会比不平滑的 0.0278 更低吗?',
      answer: '不会: $\\alpha = 1$ 时 W8A8 误差 0.0297, 比不平滑的 0.0278 还差一点。最低点在 $\\alpha = 0.5$, 误差 0.0070, 比不平滑降了 4.0 倍。\n全挪过去反而差, 因为离群值并没有消失, 只是换了个地方。$\\alpha = 1$ 时 $s_j = \\max|X_j|$:\n- 激活这边: 每个通道都被压到最大值为 1, 很好量化 (「只量化 X」0.0074)。\n- 权重这边: 第 $j$ 行乘了 $\\max|X_j|$, 离群通道那一行比别的行大 50 倍, 权重按输出通道分组时每一组都含这几行, 权重这边的量化范围被撑大, 「只量化 W」从 0.0067 飙到 0.0288。\n两边加起来就是 0.0297。\n$\\alpha = 0.5$ 时图 ② 每个通道左右两根柱子一样高: $\\max|X_j| / s_j = \\max|W_j| \\cdot s_j = \\sqrt{\\max|X_j| \\cdot \\max|W_j|}$, 离群值被平分给两边, 谁都不至于太难量化。\n和 AWQ 用的是同一个恒等式 $XW = (X/s)(s \\odot W)$, 目的相反:\n- AWQ 只量化权重, 把激活大的通道对应的权重行放大, 是为了保护权重。\n- SmoothQuant 要让激活也能 INT8, 把激活的离群通道压小, 是为了保护激活。\n真实模型里 $\\alpha$ 按层搜索, 常见 0.5~0.85; 这里只有一层, 最优恰好是 0.5 是这份数据决定的。',
    }"
  >
    <template #controls>
      <LabSlider v-model="ai" label="迁移强度 α" :min="-1" :max="10" :format="(k) => (k < 0 ? '不平滑' : (k / 10).toFixed(1))" />
      <div class="row"><button type="button" @click="ai = 5">跳到最优 α = 0.5</button><button type="button" @click="seed++">换一组通道</button></div>
    </template>

    <svg viewBox="0 0 560 330" role="group" aria-label="误差随 α 的曲线与逐通道幅度">
      <text x="4" y="12" class="cap">① Python 实测, 不随「换一组通道」变化: 输出相对误差随 α (虚线 = 不平滑的 W8A8 0.0278)</text>
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

      <text x="4" y="198" class="cap">② 8 个示意通道, ch3 放大 50 倍: 左 = 平滑后激活 max|X_j|/s_j, 右 = 平滑后权重 max|W_j|·s_j (对数高度)</text>
      <g v-for="(c, j) in ch.cols" :key="`c${j}`">
        <rect :x="cx(j)" :y="hy(c.x)" width="22" :height="316 - hy(c.x)" class="xb" :class="{ out: OUT.includes(j) }" />
        <rect :x="cx(j) + 24" :y="hy(c.w)" width="22" :height="316 - hy(c.w)" class="wb" />
        <text :x="cx(j) + 23" y="328" class="tick">ch{{ j }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>α = {{ ai < 0 ? '不平滑' : (ai / 10).toFixed(1) }} 时的输出误差: 只量化 X (权重保持浮点)</span><b>{{ row.x.toFixed(4) }}</b></div>
      <div class="kv"><span>只量化 W (激活保持浮点)</span><b :class="row.w > 0.02 ? 'bad' : ''">{{ row.w.toFixed(4) }}</b></div>
      <div class="kv"><span>W8A8 两边都量化 (括号: 比不平滑好几倍)</span><b :class="row.xw < NONE.xw * 0.5 ? 'good' : row.xw > NONE.xw ? 'bad' : ''">{{ row.xw.toFixed(4) }} ({{ (NONE.xw / row.xw).toFixed(1) }}×)</b></div>
      <div class="kv"><span>② 最大 ÷ 中位: 激活侧 / 权重侧</span><b :class="Math.max(ch.xr, ch.wr) > 10 ? 'bad' : 'good'">{{ ch.xr.toFixed(1) }} / {{ ch.wr.toFixed(1) }}</b></div>
      <div class="lab-note">
        <p>① 的数字来自 demo 的 [B] 段 (W 256×256, 离群输入通道 [7, 50, 131, 200] 放大 50 倍), 误差是相对误差 (量化前后输出之差的长度 ÷ 原输出的长度), 0.03 大约就是 3%。② 是 8 个示意通道, 按 smooth_scales 的公式现算, 不参与 ① 的误差。</p>
        <p>② 的比值:</p>
        <ul class="pts">
          <li><b>激活侧:</b> 「最大的通道比中等通道大多少倍」, 它就是 per-token 量化的难度 (每一行的范围都要照顾到最大的那个通道)。</li>
          <li><b>权重侧:</b> 同理是 per-channel 量化的难度。</li>
        </ul>
        <p>哪边超过 10 都标红。</p>
        <p>同一个 demo 里 GPTQ 走另一条路: 不挪幅度, 而是量化一行时把它的舍入误差补给后面还没量化的行 (相关激活下 0.0776 → 0.0369)。</p>
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
