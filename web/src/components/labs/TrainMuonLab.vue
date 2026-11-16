<!--
  Muon 的 Newton–Schulz 正交化 (对应 llm_train/m14_muon_optimizer:newton_schulz)。
  只讲一件事: NS 迭代只用矩阵乘, 就把梯度的所有奇异值推到 ~1 —— 每个方向迈同样大的步子。
  这里真的在 JS 里对一个 6×8 矩阵做 X ← aX + (bA + cA²)X, 奇异值是从迭代后的矩阵上量出来的。
-->
<template>
  <LabFrame
    title="Muon 的正交化 — Newton–Schulz 迭代把梯度的奇异值谱拉平, 让每个方向迈一样大的步"
    sub="- 模拟什么: 权重矩阵的梯度 $G$ 做奇异值分解 $G = U\Sigma V^\top$, 奇异值 $\Sigma$ 表示梯度在各个「奇异方向」上有多强, 通常相差几个数量级 (条件数 = 最大除以最小)。SGD 直接拿 $G$ 更新, 步子几乎全压在头几个强方向上; Adam 逐元素缩放, 能拉平量级但不认识方向。Muon 优化器把更新矩阵正交化: 奇异值全换成 1, 更新量变成 $UV^\top$, 每个奇异方向迈同样大的步。精确做 SVD 太慢, Muon 用 Newton–Schulz (NS) 迭代近似: 先把 $G$ 除以 Frobenius 范数让所有奇异值 ≤ 1, 再反复做 $X \leftarrow aX + (bA + cA^2)X$ ($A = XX^\top$), 只用矩阵乘, 效果是每个奇异值各自走多项式 $a\sigma + b\sigma^3 + c\sigma^5$ 被推向 1。这个台子真的在浏览器里对一个 6×8 矩阵做这个迭代, 奇异值是从迭代后的矩阵上量出来的。
      - 图上是什么: 上图每条线是一个奇异值 (共 6 个) 随迭代的变化, 横轴从左到右是: 原始 G、除以范数归一之后、NS 第 1 步到第 10 步; 纵轴是奇异值大小, 对数刻度。浅绿色横带是目标区间 [0.7, 1.2]: Muon 不要求精确到 1, 落进这个带就够用。竖虚线标着当前选的步数。最左一列的圆点是可拖的手柄。下面三行柱子是同一个梯度下三种更新在 6 个奇异方向上各迈多大的步 (每行按自己的最大值归一): SGD 用梯度本身; Adam 用 $\mathrm{sign}(G)$, 也就是 Adam 第一步更新的样子; Muon 用 NS 迭代后的矩阵。
      - 可以做什么: 拖「NS 迭代步数」看奇异值走多少步被推到哪; 拖「梯度条件数」改 6 个奇异值的分布 (从最大 1 到最小 1/条件数等比排列); 在上图上下拖最左一列的圆点 (或聚焦后按上下方向键) 单独改某个奇异值; 「换一组奇异向量」换 $U$、$V$ 但不改奇异值。悬停或点击下面柱子里的某个方向, 上图对应的线会高亮。
      - 应该看到: 条件数 1e4 时, 最小奇异值从 1e-4 出发每步被放大约 3.4 倍, 5 步后到 0.04 左右, 大的早就压在 1.16 附近。右侧「输入 σ max / min」是原始梯度的最大和最小奇异值; 「NS 几步后 σ max / min」是迭代后的; 「落在 [0.7, 1.2] 的方向」是有几个奇异值进了目标带, 6 个全进变绿; 「与精确 UVᵀ 的余弦」是 NS 结果和真正正交化结果的方向余弦, 越接近 1 越像。下面三行柱子: SGD 只有一两根高, Adam 参差不齐, Muon 六根差不多高。"
    module="llm_train/m14"
    run="python -m llm_train.m14_muon_optimizer.demo"
    :challenge="{
      ask: '条件数调到 1e4, 迭代 5 步: 最小的那个奇异值 (1e-4) 被推到了多少? 为什么 Muon 官方只迭代 5 步而不迭代到收敛?',
      answer: '推到 0.04 左右 (m14 实测 0.041), 放大了几百倍但还没到 1; 不迭代到收敛是因为 Muon 不需要精确结果, 5 步已经够用, 再迭代白花算力。\n- 为什么是 0.04: 每一步每个奇异值各自走多项式 $a\\sigma + b\\sigma^3 + c\\sigma^5$, 系数 (3.4445, −4.7750, 2.0315) 是特意调过的。$\\sigma$ 很小时后两项可以忽略, 每步约放大 $a \\approx 3.44$ 倍, 5 步约 $3.44^5 \\approx 480$ 倍, 1e-4 的方向被抬到约 0.05 量级。拖「NS 迭代步数」从 0 到 5 看最下面那条线, 它在对数轴上几乎是等间距往上走的。大的奇异值则一两步就被压到 1 附近, 之后在 [0.7, 1.2] 里振荡, 不会真正收敛到 1。\n- 为什么 5 步就够: Muon 要的只是「各方向步长大致相同」, 不需要精确的 $UV^\\top$ (读数里 5 步后与精确结果的余弦约 0.88)。最小方向从 1e-4 变成 0.04 已经是 400 倍的改善, 再多迭代收益递减。每步只有 3 次矩阵乘, 5 步的开销相对一次前向反向不到 1%, 迭代到收敛就不是这个价钱了。\n- 对照下面三行柱子看三种更新的差别: SGD 用梯度本身, 更新几乎全压在头一两个奇异方向上, 其余方向几乎不动; Adam 的 $\\mathrm{sign}(G)$ 把每个元素的量级拉平了, 但它是逐元素操作, 各奇异方向分到多少全看坐标系, 点「换一组奇异向量」柱子就乱变; Muon 每个方向都迈得差不多大, 而且换奇异向量也不变, 因为正交化只看奇异值。',
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
      <p class="lab-note">三行柱子 = 同一个梯度下, 三种更新在 6 个奇异方向上各迈多大步 (每行按自己的最大值归一, 所以只能在一行内比高矮)。柱子越齐说明各方向步长越均匀。悬停或点击某个方向, 上图对应的线会高亮; 用 Tab 选中上图的圆点也一样。</p>
    </div>

    <template #stats>
      <div class="kv"><span><Tex text="输入梯度的奇异值 $\sigma$ max / min" /></span><b>{{ sci(ns.s0max) }} / {{ sci(ns.s0min) }}</b></div>
      <div class="kv"><span><Tex :text="`NS ${steps} 步后的 $\\sigma$ max / min`" /></span><b>{{ ns.smax.toFixed(3) }} / {{ ns.smin < 0.01 ? sci(ns.smin) : ns.smin.toFixed(3) }}</b></div>
      <div class="kv"><span>落在目标带 [0.7, 1.2] 的方向 (6 个全进才算拉平)</span><b :class="ns.inBand === 6 ? 'good' : ns.inBand < 3 ? 'bad' : ''">{{ ns.inBand }} / 6</b></div>
      <div class="kv"><span><Tex text="与精确正交化结果 $UV^\top$ 的方向余弦 (1 = 完全一样)" /></span><b :class="{ good: ns.cos > 0.95 }">{{ ns.cos.toFixed(3) }}</b></div>
      <p class="lab-note">
        <Tex text="★ 迭代公式 $X \leftarrow aX + (bA + cA^2)X$, $A = XX^\top$, 系数 $(a, b, c) = (3.4445, -4.7750, 2.0315)$。它不改奇异向量, 只让每个奇异值各自走多项式 $a\sigma + b\sigma^3 + c\sigma^5$: 小的被放大 (0 附近斜率约 3.44), 接近 1 的被压回 1 附近。" />
        <Tex text="先除以 Frobenius 范数 (「归一」那一列) 是为了让所有 $\sigma \le 1$, 多项式迭代才在收敛域内、不会发散。" />
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
