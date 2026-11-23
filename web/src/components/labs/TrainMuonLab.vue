<!--
  Muon 的 Newton–Schulz 正交化 (对应 llm_train/m14_muon_optimizer:newton_schulz)。
  只讲一件事: NS 迭代只用矩阵乘, 就把梯度的所有奇异值推到 ~1 —— 每个方向迈同样大的步子。
  这里真的在 JS 里对一个 6×8 矩阵做 X ← aX + (bA + cA²)X, 奇异值是从迭代后的矩阵上量出来的。
-->
<template>
  <LabFrame
    title="Muon 的正交化 — Newton–Schulz 迭代把梯度的奇异值谱拉平, 让每个方向迈一样大的步"
    sub="- 模拟什么: 对权重矩阵的梯度 $G$ 作奇异值分解 $G = U\Sigma V^\top$, 奇异值对角阵 $\Sigma$ 刻画了梯度在各个正交奇异方向上的强度, 不同方向往往相差数个数量级 (条件数定义为最大奇异值除以最小奇异值)。标准 SGD 直接沿 $G$ 更新, 步长几乎全部集中在少数强奇异方向上; Adam 按矩阵元素逐一缩放, 虽能拉平各坐标分量量级却无法感知矩阵的奇异方向。Muon 优化器通过对更新矩阵作正交化处理, 将全部奇异值统一替换为 1, 使更新方向变为正交阵 $UV^\top$, 从而在每个奇异方向上迈出等幅步长。由于精确计算 SVD 开销过高, Muon 采用 Newton–Schulz (NS) 多项式迭代加以近似: 先将 $G$ 除以自身的 Frobenius 范数使全部奇异值不超过 1, 随后反复执行纯矩阵乘法迭代 $X \leftarrow aX + (bA + cA^2)X$ (其中 $A = XX^\top$), 等效于令每个奇异值独立经受五次多项式映射 $a\sigma + b\sigma^3 + c\sigma^5$ 并被逐步推向 1。本实验台在浏览器内对一个 6×8 矩阵实时运行该迭代, 图上奇异值均直接由每步迭代后的矩阵分解测得。
- 图上是什么: 上方为奇异值演化折线图, 下方为三行方向步长柱状图:
  - 上图中每条折线追踪一个奇异值 (共 6 个) 随迭代推进的轨迹, 横轴从左至右依次为原始梯度 G、Frobenius 范数归一化后以及 NS 第 1 步至第 10 步; 纵轴为奇异值幅值 (对数刻度)。
  - 浅绿色水平色带标出目标收敛区间 [0.7, 1.2]: Muon 并不要求奇异值精确收敛至 1, 落入该区间即可满足训练需求。
  - 竖直虚线标示当前选定的 NS 迭代步数。
  - 最左列圆点为可上下拖拽的初始奇异值调节手柄。
  - 下方三行柱状图对比同一梯度下三种优化器在 6 个奇异方向上的有效更新步长 (每行均按该行自身最大值归一化)。
  - SGD 行: 直接使用原始梯度矩阵更新。
  - Adam 行: 使用逐元素符号矩阵 $\mathrm{sign}(G)$, 对应无历史动量时 Adam 首步更新的形态。
  - Muon 行: 使用经当前步数 NS 迭代后的矩阵更新。
- 可以做什么: 控件从上到下依次为:
  - 拖动「NS 迭代步数」观察各阶奇异值经不同步数映射后的落点。
  - 拖动「梯度条件数」整体改变 6 个初始奇异值的跨度 (从最大值 1 至最小值 1/条件数按等比数列分布)。
  - 在上图中上下拖动最左列圆点手柄 (或聚焦后使用上下方向键) 单独微调某一个奇异值。
  - 点击「换一组奇异向量」重新随机生成正交基 $U$ 与 $V$, 同时保持奇异值不变。
  - 悬停或点击下方柱状图中的任一方向, 上图对应的奇异值折线将同步高亮。
- 应该看到: 当条件数设为 1e4 时, 最小奇异值从 1e-4 起步每步约放大 3.4 倍, 经 5 步迭代后升至 0.04 左右, 而较大奇异值在一两步内便稳定在 1.16 附近。对比下方三行柱状图可见: SGD 仅有一两根柱显著突出, Adam 高低参差且随坐标系旋转波动, Muon 则在六个方向上高度接近均匀。右侧读数含义如下:
  - 「输入 σ max / min」: 原始梯度矩阵的最大与最小奇异值。
  - 「NS 几步后 σ max / min」: 经当前选定步数迭代后的最大与最小奇异值。
  - 「落在 [0.7, 1.2] 的方向」: 进入目标区间的奇异值个数, 6 个方向全部进入时显示绿色。
  - 「与精确 UVᵀ 的余弦」: NS 近似矩阵与精确正交化矩阵之间的 Frobenius 内积余弦相似度, 越接近 1 代表方向逼近越准确。"
    module="llm_train/m14"
    run="python -m llm_train.m14_muon_optimizer.demo"
    :challenge="{
      ask: '将梯度条件数调至 1e4 并固定迭代 5 步: 最小奇异值 (1e-4) 被推高到了多少? 为什么 Muon 官方实现仅固定迭代 5 步, 而不继续迭代至精确收敛?',
      answer: '结论:\n- 最小奇异值被推高至 0.04 左右 (m14 实测为 0.041), 虽放大了数百倍但尚未到达 1。\n- 不继续迭代至精确收敛, 是因为优化更新并不需要精确正交矩阵, 5 步迭代的均衡效果已足够支撑训练, 继续增加步数只会徒增矩阵乘算力开销。\n逐项解析:\n- 0.04 的数值来源: 每一步迭代令各奇异值独立通过五次多项式 $a\\sigma + b\\sigma^3 + c\\sigma^5$, 其系数 (3.4445, −4.7750, 2.0315) 经过专门设计以最大化原点附近的斜率。当 $\\sigma$ 极小时高次项可忽略, 单步放大倍率约为一次项系数 $a \\approx 3.44$, 连续 5 步累计放大约 $3.44^5 \\approx 480$ 倍, 从而将 1e-4 量级的弱方向抬升至约 0.05 量级。将「NS 迭代步数」从 0 拖至 5 观察最下方折线, 可见其在对数坐标轴上近乎等距攀升。与此同时, 较大的奇异值在一两步内即被压缩至 1 附近, 随后在目标带 [0.7, 1.2] 内围绕 1 小幅振荡而不会精确收敛至 1。\n- 5 步即够用的工程权衡: Muon 的核心诉求是让各奇异方向的更新步长大致均衡, 并非追求严格的 $UV^\\top$ 正交阵 (右侧读数显示 5 步后与精确正交阵的余弦相似度已达约 0.88)。将最弱方向从 1e-4 抬升至 0.04 已实现了约 400 倍的条件数改善, 继续迭代的边际收益迅速递减。单步 NS 迭代仅含 3 次矩阵乘法, 5 步总开销占单步前向与反向传播总算力的比例不足 1%; 若强求全部奇异值完全收敛则会显著拉高优化器耗时。\n- 对照下方三行柱状图可直观区分三种更新机制:\n  - SGD: 直接沿原始梯度更新, 步长几乎完全集中在最强的一两个奇异方向上, 弱奇异方向几乎停滞。\n  - Adam: 借助逐元素 $\\mathrm{sign}(G)$ 拉平了矩阵各坐标元素的幅值, 但因逐元素非线性操作与坐标系强绑定, 各奇异方向实际分得的步长完全取决于奇异向量相对坐标轴的朝向, 点击「换一组奇异向量」即可看到柱高剧烈波动。\n  - Muon: 各奇异方向步长高度均衡, 且具备严格的正交旋转不变性, 更换奇异向量时柱状图分布保持不变。',
    }"
  >
    <template #controls>
      <LabSlider v-model="steps" label="NS 迭代步数" :min="0" :max="10" />
      <LabSlider v-model="cond" label="梯度条件数" :min="0.5" :max="5" :step="0.5" :format="(v) => '1e' + v" />
      <div class="row"><button type="button" @click="seed++">换一组奇异向量</button></div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="奇异值随 Newton-Schulz 迭代的变化, 最左一列的圆点可拖">
      <rect :x="X0" :width="W - X0 - 8" :y="py(Math.log10(1.2))" :height="py(Math.log10(0.7)) - py(Math.log10(1.2))" class="target" />
      <g v-for="t in [-6, -4, -2, 0]" :key="t">
        <line :x1="X0" :x2="W - 8" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">1e{{ t }}</text>
      </g>
      <text v-for="(lab, c) in colLabels" :key="c" :x="px(c)" :y="H - 6" class="tick" text-anchor="middle" :class="{ cur: c === steps + 1 }">{{ lab }}</text>
      <line :x1="px(steps + 1)" :x2="px(steps + 1)" y1="8" :y2="H - 20" class="curline" />
      <g v-for="(tr, i) in ns.traj" :key="i" :class="['line', { hot: lit === i }]" @mouseenter="hover = i" @mouseleave="hover = -1">
        <polyline :points="tr.map((s, c) => `${px(c)},${py(Math.log10(s))}`).join(' ')" />
        <circle v-for="(s, c) in tr.slice(1)" :key="c" :cx="px(c + 1)" :cy="py(Math.log10(s))" r="2.5" />
        <circle
          :cx="px(0)" :cy="py(Math.log10(tr[0]))" r="7" class="handle draggable" tabindex="0" role="slider"
          :aria-label="`第 ${i} 个奇异值`" :aria-valuenow="tr[0]"
          @pointerdown="start($event, { svg, onMove: ({ y }) => setSigma(i, 10 ** clamp(pyInv(y), -5, 0)) })"
          @keydown.up.prevent="setSigma(i, Math.min(1, sigma[i] * 2))" @keydown.down.prevent="setSigma(i, Math.max(1e-5, sigma[i] / 2))"
          @focus="pin = i"
        />
      </g>
    </svg>

    <div class="dirs">
      <div v-for="row in ns.rows" :key="row.name" class="dir-row">
        <span class="dir-name">{{ row.name }}</span>
        <span v-for="(v, i) in row.vals" :key="i" class="dir-bar" :class="{ hot: lit === i }" :title="`方向 ${i}: ${v.toFixed(3)}`"
              @mouseenter="hover = i" @mouseleave="hover = -1" @click="pin = pin === i ? -1 : i">
          <i :style="{ height: Math.max(2, Math.min(1, v) * 100) + '%', background: row.color }" />
        </span>
      </div>
      <p class="lab-note">下方三行柱状图展示同一梯度下三种优化更新在 6 个奇异方向上的相对步长 (每行均按本行最大值独立归一化, 因此仅限同行内部对比高低); 柱高越整齐说明各奇异方向步长越均匀。悬停或点击任一方向柱, 或通过 Tab 键聚焦上图左列圆点, 对应的奇异值折线均会同步高亮。</p>
    </div>

    <template #stats>
      <div class="kv"><span><Tex text="输入梯度的奇异值 $\sigma$ max / min" /></span><b>{{ sci(ns.s0max) }} / {{ sci(ns.s0min) }}</b></div>
      <div class="kv"><span><Tex :text="`NS ${steps} 步后的 $\\sigma$ max / min`" /></span><b>{{ ns.smax.toFixed(3) }} / {{ ns.smin < 0.01 ? sci(ns.smin) : ns.smin.toFixed(3) }}</b></div>
      <div class="kv"><span>落在目标带 [0.7, 1.2] 的方向 (6 个全进才算拉平)</span><b :class="ns.inBand === 6 ? 'good' : ns.inBand < 3 ? 'bad' : ''">{{ ns.inBand }} / 6</b></div>
      <div class="kv"><span><Tex text="与精确正交化结果 $UV^\top$ 的方向余弦 (1 = 完全一样)" /></span><b :class="{ good: ns.cos > 0.95 }">{{ ns.cos.toFixed(3) }}</b></div>
      <p class="lab-note">
        <Tex text="★ 迭代公式为 $X \leftarrow aX + (bA + cA^2)X$, 其中 $A = XX^\top$, 多项式系数取 $(a, b, c) = (3.4445, -4.7750, 2.0315)$。该变换严格保持左右奇异向量不变, 仅令每个奇异值独立执行奇次多项式映射 $a\sigma + b\sigma^3 + c\sigma^5$: 接近 0 的小奇异值被快速放大 (原点附近斜率约为 3.44), 接近 1 的大奇异值则被稳定约束在 1 附近。" />
        <Tex text="迭代前先除以矩阵的 Frobenius 范数 (对应图中「归一」列), 目的是确保全体初始奇异值满足 $\sigma \le 1$, 使后续多项式迭代始终处于收敛域内而不发生数值发散。" />
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const R = 6, C = 8, KMAX = 10
const steps = ref(5), cond = ref(4), seed = ref(1), svg = ref(null)
const hover = ref(-1), pin = ref(-1)                     // 悬停的方向 / 点击或聚焦固定的方向
const lit = computed(() => (hover.value >= 0 ? hover.value : pin.value))
const { start } = useDrag()
const logspace = (c) => range(R).map((i) => 10 ** (-c * i / (R - 1)))
const sigma = ref(logspace(cond.value))
watch(cond, (c) => { sigma.value = logspace(c) })
const setSigma = (i, v) => { sigma.value = sigma.value.map((s, j) => (j === i ? v : s)) }

const T = (m) => m[0].map((_, j) => m.map((row) => row[j]))
const mul = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
const lin = (ca, a, cb, b) => a.map((row, i) => row.map((v, j) => ca * v + cb * b[i][j]))
const fro = (m) => Math.sqrt(m.flat().reduce((s, v) => s + v * v, 0))
const dot = (a, b) => { const bf = b.flat(); return a.flat().reduce((s, v, i) => s + v * bf[i], 0) }
// Gram–Schmidt: n 个 dim 维的正交单位向量 (按行)
const orthoRows = (n, dim, rand) => {
  const rows = []
  while (rows.length < n) {
    let v = range(dim).map(() => randn(rand))
    for (const u of rows) { const p = v.reduce((s, x, k) => s + x * u[k], 0); v = v.map((x, k) => x - p * u[k]) }
    const nv = Math.hypot(...v)
    if (nv > 1e-6) rows.push(v.map((x) => x / nv))
  }
  return rows
}

const ns = computed(() => {
  const rand = mulberry32(seed.value * 131)
  const Ut = orthoRows(R, R, rand), Vt = orthoRows(R, C, rand) // Ut[i] = u_i, Vt[i] = v_i
  const U = T(Ut)
  const G = mul(U, Vt.map((row, i) => row.map((v) => v * sigma.value[i]))) // G = U Σ Vᵀ, [6, 8]
  const measure = (X) => range(R).map((i) => Math.abs(mul(mul([Ut[i]], X), T([Vt[i]]))[0][0])) // σ_i = u_iᵀ X v_i
  const a = 3.4445, b = -4.775, c = 2.0315
  let X = G.map((row) => row.map((v) => v / (fro(G) + 1e-7)))
  const hist = [measure(G), measure(X)]
  let atK = X
  for (let k = 1; k <= KMAX; k++) {
    const A = mul(X, T(X))                                   // [6, 6]
    X = lin(a, X, 1, mul(lin(b, A, c, mul(A, A)), X))        // ★ 整个优化器的核心就这一行
    hist.push(measure(X))
    if (k === steps.value) atK = X
  }
  const sv = hist[steps.value + 1]
  const exact = mul(U, Vt)
  const signG = G.map((row) => row.map(Math.sign))          // Adam 第一步: m/√v = sign(g), 逐元素
  const norm = (vals) => vals.map((v) => v / Math.max(...vals))
  return {
    traj: range(R).map((i) => hist.map((h) => Math.max(h[i], 1e-7))),
    s0max: Math.max(...hist[0]), s0min: Math.min(...hist[0]),
    smax: Math.max(...sv), smin: Math.min(...sv),
    inBand: sv.filter((s) => s >= 0.7 && s <= 1.2).length,
    cos: dot(atK, exact) / (fro(atK) * fro(exact)),
    rows: [
      { name: 'SGD: 梯度本身', vals: norm(hist[0]), color: 'var(--text-dim)' },
      { name: 'Adam: sign(G)', vals: norm(measure(signG)), color: 'var(--eye)' },
      { name: `Muon: NS ${steps.value} 步`, vals: norm(sv), color: 'var(--left)' },
    ],
  }
})

const W = 560, H = 260, X0 = 46, LO = -7, HI = 0.5
const colLabels = ['原始 G', '归一', ...range(KMAX).map((k) => String(k + 1))]
const px = (c) => X0 + 24 + (c / (KMAX + 1)) * (W - X0 - 48)
const py = (lg) => 10 + (1 - (clamp(lg, LO, HI) - LO) / (HI - LO)) * (H - 36)
const pyInv = (y) => LO + (1 - (y - 10) / (H - 36)) * (HI - LO)
const sci = (v) => v.toExponential(1)
</script>

<style scoped>
.target { fill: var(--left); opacity: 0.15; }
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.tick.cur { fill: var(--accent); font-weight: 700; }
.curline { stroke: var(--accent); stroke-dasharray: 3 3; opacity: 0.6; }
.line polyline { fill: none; stroke: var(--accent); stroke-width: 1.5; opacity: 0.55; }
.line circle { fill: var(--accent); }
.line.hot polyline { stroke-width: 3; opacity: 1; stroke: var(--warn); }
.line .handle { fill: var(--warn); stroke: var(--bg-card); stroke-width: 2; }
.dirs { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
.dir-row { display: grid; grid-template-columns: 110px repeat(6, minmax(0, 1fr)); gap: 4px; align-items: end; }
.dir-name { font-size: 11px; color: var(--text-muted); align-self: center; }
.dir-bar { cursor: pointer; height: 34px; display: flex; align-items: flex-end; border-bottom: 1px solid var(--border-strong); }
.dir-bar.hot { background: var(--accent-soft); }
.dir-bar i { display: block; width: 100%; border-radius: 2px 2px 0 0; }
</style>
