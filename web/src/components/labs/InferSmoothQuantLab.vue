<!--
  SmoothQuant 实验台 (对应 llm_infer/m25_weight_quant/smoothquant.py:smooth_scales / w8a8_matmul)。
  一件事: 同一个缩放恒等式 XW = (X/s)(s⊙W), 用来把激活的离群值"挪"给权重, 让 W8A8 两边都好量化。
-->
<template>
  <LabFrame
    title="SmoothQuant — 把激活里的离群值挪一部分给权重, 让两边都好量化"
    sub="- 模拟什么: W8A8 量化将权重 $W$ 与输入激活 $X$ 同时压缩为 INT8 整数, 使矩阵乘可直接调用硬件 INT8 Tensor Core 运算单元, 比仅量化权重进一步提升算力吞吐。工程上硬件要求权重按输出通道分组量化、激活按 token (逐行) 分组量化; 难点在于真实大模型的激活张量中存在少数固定离群通道 (本实验包含 4 个幅值放大 50 倍的通道), 在所有 token 行上均呈现极大值, 导致每一行激活的动态范围都被严重拉宽、其余正常通道损失有效精度。SmoothQuant 利用数学恒等式 $XW = (X/s)(s \odot W)$ 将量化难度从激活侧部分迁移至权重侧:
  - 将输入激活的第 $j$ 个通道列除以缩放因子 $s_j$ 以压平离群尖峰。
  - 将权重矩阵对应的第 $j$ 行同步乘以 $s_j$ 以保持矩阵乘积不变。
  - 迁移指数 $\alpha$ 控制难度分担比例, 定义为 $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$, $\alpha$ 越大向权重侧转移的幅度越多。
- 图上是什么: 上下两幅视图:
  - 图 ①: Python demo 实测的三条相对误差随 $\alpha \in [0, 1]$ 变化的曲线, 分别为「只量化 X」(权重保持浮点)、「只量化 W」(激活保持浮点) 以及「W8A8」(权重与激活同时量化, 粗实线); 水平虚线标示不做平滑直接执行 W8A8 的基准误差 0.0278。
  - 图 ②: 8 个代表性通道的幅值对比柱状图, 每个通道并列左右两根柱子 (对数纵轴刻度), 左柱为平滑后该激活通道的最大绝对值 $\max|X_j| / s_j$, 右柱为平滑后对应权重行的最大绝对值 $\max|W_j| \cdot s_j$; 其中 ch3 为放大 50 倍的离群通道并以醒目颜色标出。这 8 个通道由前端按公式实时计算仅作直观示意, 不参与图 ① 实测误差曲线的计算。
- 可以做什么: 通过三个控件调节迁移参数:
  - 拖动「迁移强度 α」滑杆或点击图 ① 曲线上的圆点选定 $\alpha$, 最左端档位对应「不平滑」基线。
  - 点击「跳到最优 α = 0.5」直接定位至 W8A8 误差曲线最低点。
  - 点击「换一组通道」重新采样图 ② 的 8 个示意通道 (图 ① 实测曲线保持不变)。
- 应该看到: 右侧前三项误差读数取自图 ①, 「W8A8」括号内标注其相对不平滑基线的误差降低倍数; 「最大 ÷ 中位」统计图 ② 中激活侧与权重侧各自通道最大值与中位数之比, 超过 10 时标红提示该侧存在显著离群尖峰、难以直接低误差量化。将 $\alpha$ 从左向右逐步拖动时可观察到:
  - 处于「不平滑」档位时: 图 ② 中 ch3 的左侧激活柱远高于其余通道, 右侧「最大 ÷ 中位: 激活」读数悬殊, 表明量化瓶颈完全集中在激活侧。
  - 逐步增大 $\alpha$ 时: ch3 的左侧激活柱下降、右侧权重柱上升, 离群幅值向权重侧转移。
  - 拖至 $\alpha = 1$ 时: 离群尖峰被完全转嫁至权重行, 导致「只量化 W」的误差急剧上升。
  - 处于 $\alpha = 0.5$ 时: 每个通道的左右两根柱子高度完全齐平, 量化难度由激活与权重均匀分担, 图 ① 的 W8A8 综合误差也在此处达到最低。"
    module="llm_infer/m25"
    run="python -m llm_infer.m25_weight_quant.demo"
    :challenge="{
      ask: '将迁移强度 $\\alpha$ 拖至 1, 把激活离群幅度全部转移至权重侧。此时 W8A8 的相对误差会低于不平滑基线 (0.0278) 吗?',
      answer: '不会更低: 当 $\\alpha = 1$ 时 W8A8 相对误差为 0.0297, 反而略劣于不平滑基线的 0.0278; 误差极小点出现在 $\\alpha = 0.5$ 处, 此时误差降至 0.0070, 较不平滑基线降低了 4.0 倍。\n将离群幅度完全单向转移之所以适得其反, 是因为离群尖峰并未凭空消失, 而只是改变了所在的张量。当 $\\alpha = 1$ 时缩放因子取 $s_j = \\max|X_j|$:\n- 激活侧: 所有输入通道的最大绝对值均被归一化压平至 1, 极易量化 (「只量化 X」误差降至 0.0074)。\n- 权重侧: 第 $j$ 行权重乘以 $\\max|X_j|$ 后, 原离群通道对应的权重行幅值比其余普通行高出约 50 倍; 由于权重按输出通道 (列) 分组量化, 每个输出通道组内都包含这几个巨幅输入行, 导致所有输出通道的权重量化区间均被严重拉宽, 「只量化 W」误差从 0.0067 大幅升至 0.0288。\n两侧误差叠加后使最终 W8A8 误差升至 0.0297。\n当取折中值 $\\alpha = 0.5$ 时, 图 ② 中每个通道左右两根柱子高度完全相等, 满足 $\\max|X_j| / s_j = \\max|W_j| \\cdot s_j = \\sqrt{\\max|X_j| \\cdot \\max|W_j|}$, 几何均值将极端幅值均匀分摊给激活与权重两侧, 使双方均处于 INT8 可精细表示的动态范围内。\nSmoothQuant 与 AWQ 虽然共享同一等价变换 $XW = (X/s)(s \\odot W)$, 但两者的优化目标截然不同:\n- AWQ 仅对权重做低比特量化 (激活保持 fp16), 放大显著激活通道对应的权重行是为了减小该行权重的相对舍入误差以保护权重。\n- SmoothQuant 需要将激活与权重同时量化为 INT8, 缩小激活离群通道并将部分幅值分摊给权重是为了压平激活尖峰以保护激活量化精度。\n真实大模型中通常按层在校准集上搜索最优 $\\alpha$, 常见取值区间为 0.5~0.85; 本实验仅模拟单层合成数据, 最优点恰好落在 0.5 由该组数据的幅值分布决定。',
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
        <p>图 ① 的数据取自 Python demo 的 [B] 实验段 (权重矩阵维度 256×256, 离群输入通道 [7, 50, 131, 200] 幅值放大 50 倍), 纵轴相对误差定义为量化前后输出向量差值的范数除以原始输出范数 (0.03 约对应 3% 的相对偏差)。图 ② 展示的 8 个示意通道由前端按 smooth_scales 公式实时计算, 不参与图 ① 误差曲线的统计。</p>
        <p>② 的比值:</p>
        <ul class="pts">
          <li><b>激活侧:</b> 「最大的通道比中等通道大多少倍」, 它就是 per-token 量化的难度 (每一行的范围都要照顾到最大的那个通道)。</li>
          <li><b>权重侧:</b> 同理是 per-channel 量化的难度。</li>
        </ul>
        <p>哪边超过 10 都标红。</p>
        <p>同一 Python demo 中还对照了 GPTQ 的另一条技术路线: 不做跨张量幅值迁移, 而是在逐列/逐行量化权重时利用逆 Hessian 信息将当前权重的舍入误差补偿至后续尚未量化的权重上 (在存在通道相关性的校准激活下相对误差从 0.0776 降至 0.0369)。</p>
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
