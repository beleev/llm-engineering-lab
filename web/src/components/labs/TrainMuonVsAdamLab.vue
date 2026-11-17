<!--
  Muon vs Adam (对应 llm_train/m14_muon_optimizer:run, 缩小到 16×16 在浏览器里真跑 150 步)。
  只讲一件事: Adam 的逐元素缩放只能修 "与坐标轴对齐" 的病态; 把同一个病态问题旋转一下, Adam 就修不动了, 而 Muon 的正交化与基无关。
  10 × 150 步不放在 computed 里: 参数变了才在渲染之后算一次, 结果按 (旋转, 种子) 缓存。
-->
<template>
  <LabFrame
    title="Muon vs Adam — 同一个病态问题, 把病态方向转一下, 谁还修得动?"
    sub="- 模拟什么: 一个病态的矩阵回归问题: 要学一个 16×16 的矩阵, 输入数据在 16 个主轴方向上的尺度从 1 等比降到 0.01 (条件数 1e4), 也就是有的方向梯度很大、有的方向梯度很小, 一个统一的步长管不住所有方向。这个台子让两种优化器各扫 5 个学习率、各跑 150 步, 真的在你的浏览器里跑 (bias 两边都用 Adam, 学习率线性退火到 0, 与 Python demo 一致)。两种优化器的办法:
        - Adam: 逐元素缩放, 每个参数按自己梯度的二阶矩 $1/\sqrt{v}$ 调步长, 这只能修「病态方向恰好沿坐标轴」的情况。
        - Muon: 把整个更新矩阵用 Newton–Schulz 迭代正交化, 让每个奇异方向迈同样大的步, 和坐标轴怎么摆无关。
      - 图上是什么: 横轴是训练步数 (0 到 150), 纵轴是 loss, 对数刻度。
        - 两条折线是 Adam 和 Muon 各自在选中学习率下的 loss 曲线, 颜色对应控件区的名字, 线尾标着学习率和最终 loss。
        - 控件区的两排格子是两种优化器在 5 个学习率下的最终 loss, 带绿色描边的是各自最好的那个, 高亮的是当前画在图上的。
      - 可以做什么: 控件从上到下:
        - 点「病态方向与坐标轴对齐」或「随机旋转 (不对齐)」切换问题的摆法 (难度不变, 只是把病态方向转到坐标轴之外)。
        - 「换一组数据」换随机种子重跑 (要等一下, 10 条曲线 × 150 步)。
        - 点任一学习率格子把那条曲线画到图上。
        - 「选最好」回到各自最好的。
      - 应该看到: 轴对齐时 Adam 的最好成绩和 Muon 相当甚至更好; 切到随机旋转, Adam 的最好成绩明显变差, Muon 几乎不变。右侧读数:
        - 「Adam 最好 / Muon 最好」: 各自 5 个学习率里最低的最终 loss (括号里是对应的学习率)。
        - 「各自最好的相比」: 两者之比, 写成谁好几倍。
        - 「图上画的是不是各自最好」: 提醒你现在比的是不是公平的那一对, 不是的话「不是」变红。"
    module="llm_train/m14"
    run="python -m llm_train.m14_muon_optimizer.demo"
    :challenge="{
      ask: '先看「轴对齐」: 谁赢? 再切到「随机旋转」: 问题的难度 (条件数) 一点没变, 为什么 Adam 的最好成绩明显变差, Muon 几乎不受影响?',
      answer: '答案:\n- 轴对齐时 Adam 赢或打平。\n- 旋转之后 Muon 赢, 因为 Adam 的缩放是按坐标轴做的, Muon 的是按奇异方向做的。\n为什么:\n- 轴对齐: 病态方向就是 16 个输入坐标本身, 每个坐标的梯度量级各不相同, 正好一个坐标一个尺度。Adam 对每个参数单独按 $1/\\sqrt{v}$ 缩放步长, 等于逐坐标把量级拉平, 这是它的主场 (换几组数据看, 从小胜到大胜都有; Python demo 用 32×32 时 Adam 好 1.9 倍)。\n- 旋转之后: 问题的条件数没变, 但每个病态方向变成了所有坐标的线性组合。任何一个坐标上的梯度都混着大尺度和小尺度方向的贡献, 逐元素的缩放不知道该放大还是缩小, 无从下手, 所以 Adam 的最好成绩明显变差。\n- Muon 为什么不受影响: Newton–Schulz 正交化作用在奇异值上, 旋转数据只改变梯度的奇异向量 (方向怎么摆), 不改变奇异值 (各方向多强), 所以 Muon 的更新在两种摆法下几乎一样, 轨迹几乎不变 (Python demo: 旋转时 Muon 好 9.2 倍)。\n真实网络里权重矩阵的病态方向没有理由恰好沿坐标轴, 更像旋转后的情况, 这就是 Muon 在 LLM 预训练上省算力的来源。比较时记得看右侧「图上画的是不是各自最好」, 两边都调到最好的学习率再比才公平, 否则比的是调参。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !rotate }" @click="rotate = false">病态方向与坐标轴对齐</button>
        <button type="button" :class="{ active: rotate }" @click="rotate = true">随机旋转 (不对齐)</button>
        <button type="button" @click="seed++">换一组数据</button>
      </div>
      <p v-if="!sweep" class="lab-note">正在跑 10 × 150 步…</p>
      <div v-for="o in sweep ? OPTS : []" :key="o.key" class="row lr-row">
        <span class="oname" :style="{ color: o.color }">{{ o.name }}</span>
        <button
          v-for="(lr, i) in LRS" :key="lr" type="button" class="chip mono" :class="{ active: pick[o.key] === i, best: sweep.best[o.key] === i }"
          :title="`lr = ${lr}, 最终 loss ${sweep[o.key][i].final.toExponential(2)}`" @click="pick[o.key] = i"
        >lr {{ lr }}<br /><small>{{ sweep[o.key][i].final.toExponential(1) }}</small></button>
        <button type="button" class="chip" @click="pick[o.key] = sweep.best[o.key]">选最好</button>
      </div>
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="loss 随训练步数的变化 (对数纵轴)">
      <g v-for="t in [-6, -5, -4, -3, -2, -1, 0, 1]" :key="t">
        <line :x1="X0" :x2="W - 8" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">1e{{ t }}</text>
      </g>
      <text :x="W - 8" :y="H - 4" class="tick" text-anchor="end">训练步数 → 150</text>
      <template v-if="sweep">
        <polyline v-for="o in OPTS" :key="o.key" :points="curve(o.key)" fill="none" :stroke="o.color" stroke-width="2" />
        <g v-for="o in OPTS" :key="'e' + o.key">
          <circle :cx="px(STEPS - 1)" :cy="py(lg(cur(o.key).final))" r="4" :fill="o.color" />
          <text :x="px(STEPS - 1) - 8" :y="py(lg(cur(o.key).final)) - 7" class="endl" text-anchor="end" :fill="o.color">{{ o.name }} lr {{ LRS[pick[o.key]] }}: {{ cur(o.key).final.toExponential(1) }}</text>
        </g>
      </template>
    </svg>

    <template #stats>
      <template v-if="sweep">
        <div class="kv"><span>Adam 5 个学习率里最好的最终 loss (lr {{ LRS[sweep.best.adam] }})</span><b>{{ bestOf('adam').toExponential(2) }}</b></div>
        <div class="kv"><span>Muon 5 个学习率里最好的最终 loss (lr {{ LRS[sweep.best.muon] }})</span><b>{{ bestOf('muon').toExponential(2) }}</b></div>
        <div class="kv"><span>各自最好的相比 (loss 之比)</span><b>{{ ratio > 1 ? `Muon 好 ${ratio.toFixed(1)}×` : `Adam 好 ${(1 / ratio).toFixed(1)}×` }}</b></div>
        <div class="kv"><span>图上画的是不是各自最好的学习率</span><b :class="{ bad: !onBest }">{{ onBest ? '是' : '不是' }}</b></div>
      </template>
      <div class="lab-note">
        <p>★ 公平比较 = 各自调到最好的学习率再比, 否则比的是调参而不是优化器。格子里的小字是该学习率的最终 loss, 绿色描边的是各自最好的那个; 发散成 NaN 的学习率不参与「最好」的比较。</p>
        <p>两边的 bias 都用 Adam, 因为 Muon 只管 2-D 矩阵, 向量参数没有「正交化」可言。学习率线性退火到 0, 与 Python demo 相同。</p>
        <p>优化器状态: Adam 每个矩阵存 2 份 (一阶矩、二阶矩), Muon 只存 1 份 (动量), 反而更省显存。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref, shallowRef, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32, randn, range } from '@/utils/labmath.js'

const D = 16, B = 32, STEPS = 150, LRS = [0.03, 0.1, 0.3, 1, 3]
const OPTS = [{ key: 'adam', name: 'Adam', color: 'var(--eye)' }, { key: 'muon', name: 'Muon', color: 'var(--left)' }]
const rotate = ref(true), seed = ref(1), pick = reactive({ adam: 0, muon: 0 })

// 扁平 Float64Array 上的矩阵乘: C[n×m] = A[n×k] · B[k×m] (tA/tB = 是否先转置)
const mm = (A, Bm, n, k, m, tA = false, tB = false) => {
  const C = new Float64Array(n * m)
  for (let i = 0; i < n; i++) for (let l = 0; l < k; l++) {
    const a = tA ? A[l * n + i] : A[i * k + l]
    if (a !== 0) for (let j = 0; j < m; j++) C[i * m + j] += a * (tB ? Bm[j * k + l] : Bm[l * m + j])
  }
  return C
}
// ★ 与 Python 的 newton_schulz 相同 (方阵, 不需要转置分支)
const newtonSchulz = (G) => {
  const a = 3.4445, b = -4.775, c = 2.0315
  const nrm = Math.sqrt(G.reduce((s, v) => s + v * v, 0)) + 1e-7 // Frobenius 归一 → 奇异值全部 ≤ 1
  let X = G.map((v) => v / nrm)
  for (let s = 0; s < 5; s++) {
    const A = mm(X, X, D, D, D, false, true), AA = mm(A, A, D, D, D)
    const P = A.map((v, i) => b * v + c * AA[i]), PX = mm(P, X, D, D, D)
    X = X.map((v, i) => a * v + PX[i])
  }
  return X
}
const adam = (p, g, m, v, t, lr) => {
  const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t
  for (let i = 0; i < p.length; i++) {
    m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]
    p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8)
  }
}

const runOne = (opt, lr, rot, sd) => {
  const rand = mulberry32(sd * 1009 + 14)
  const scale = range(D).map((i) => 10 ** ((-2 * i) / (D - 1))) // 主轴标准差 1 → 0.01
  // 随机正交矩阵 Q (Gram–Schmidt); 不旋转时为单位阵
  const Q = new Float64Array(D * D)
  for (let i = 0; i < D; i++) {
    const v = range(D).map(() => randn(rand))
    for (let p = 0; p < i; p++) { let d = 0; for (let j = 0; j < D; j++) d += v[j] * Q[p * D + j]; for (let j = 0; j < D; j++) v[j] -= d * Q[p * D + j] }
    const n = Math.hypot(...v); for (let j = 0; j < D; j++) Q[i * D + j] = rot ? v[j] / n : i === j ? 1 : 0
  }
  const Wt = Float64Array.from({ length: D * D }, () => randn(rand)), bt = Float64Array.from({ length: D }, () => randn(rand))
  const Wm = new Float64Array(D * D), bias = new Float64Array(D), buf = new Float64Array(D * D)
  const mW = new Float64Array(D * D), vW = new Float64Array(D * D), mb = new Float64Array(D), vb = new Float64Array(D)
  const losses = []
  for (let t = 1; t <= STEPS; t++) {
    const raw = Float64Array.from({ length: B * D }, (_, i) => randn(rand) * scale[i % D])
    const x = mm(raw, Q, B, D, D)
    const diff = mm(x, Wm, B, D, D), target = mm(x, Wt, B, D, D)
    let loss = 0
    for (let i = 0; i < B * D; i++) { diff[i] += bias[i % D] - target[i] - bt[i % D]; loss += diff[i] * diff[i] }
    losses.push(loss / (B * D))
    const k = 2 / (B * D), gW = mm(x, diff, D, B, D, true).map((v) => v * k), gb = new Float64Array(D)
    for (let i = 0; i < B * D; i++) gb[i % D] += diff[i] * k
    const lrT = lr * (1 - t / STEPS)
    if (opt === 'muon') {
      for (let i = 0; i < D * D; i++) buf[i] = 0.95 * buf[i] + gW[i]
      const O = newtonSchulz(gW.map((g, i) => g + 0.95 * buf[i])), s = lrT * 0.2 * Math.sqrt(D)
      for (let i = 0; i < D * D; i++) Wm[i] -= s * O[i]
    } else adam(Wm, gW, mW, vW, t, lrT)
    adam(bias, gb, mb, vb, t, 0.05 * (1 - t / STEPS))
  }
  return { losses, final: losses[STEPS - 1] }
}

// 2 × 5 × 150 步。发散成 NaN 的当作 +∞, 不参与「最好」的比较
const fin = (x) => (Number.isFinite(x) ? x : Infinity)
const runSweep = (rot, sd) => {
  const out = { best: {} }
  for (const o of OPTS) {
    out[o.key] = LRS.map((lr) => runOne(o.key, lr, rot, sd))
    out.best[o.key] = out[o.key].reduce((bi, r, i, arr) => (fin(r.final) < fin(arr[bi].final) ? i : bi), 0)
  }
  return out
}
// ★ 只在切换旋转 / 换数据时算一次, 放到下一个任务里跑, 不堵首屏。算完把图上的 lr 拨到各自最好的那个
const sweep = shallowRef(null), cache = new Map()
watch([rotate, seed], ([rot, sd]) => {
  const key = `${rot}-${sd}`
  const show = () => {
    if (key !== `${rotate.value}-${seed.value}`) return   // 等的时候又切走了
    sweep.value = cache.get(key)
    Object.assign(pick, sweep.value.best)
  }
  if (cache.has(key)) show()
  else setTimeout(() => { cache.set(key, runSweep(rot, sd)); show() }, 0)
}, { immediate: true })
const cur = (key) => sweep.value[key][pick[key]]
const bestOf = (key) => sweep.value[key][sweep.value.best[key]].final
const ratio = computed(() => bestOf('adam') / bestOf('muon'))
const onBest = computed(() => OPTS.every((o) => pick[o.key] === sweep.value.best[o.key]))

const W = 560, H = 250, X0 = 40, LO = -6, HI = 1.5
const px = (s) => X0 + (s / (STEPS - 1)) * (W - X0 - 12)
const py = (lg) => 10 + (1 - (Math.min(HI, Math.max(LO, lg)) - LO) / (HI - LO)) * (H - 30)
const lg = (l) => Math.log10(Number.isFinite(l) && l > 0 ? l : 1e9)   // 发散的点画到图顶
const curve = (key) => cur(key).losses.map((l, s) => `${px(s)},${py(lg(l))}`).join(' ')
</script>

<style scoped>
.lr-row { gap: 6px; }
.oname { font-size: 12px; font-weight: 600; min-width: 52px; }
.chip { min-height: 36px; padding: 2px 8px; font-size: 11px; line-height: 1.25; }
.chip small { font-size: 9px; opacity: 0.8; }
.chip.best { box-shadow: 0 0 0 2px var(--left); }
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.endl { font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
</style>
