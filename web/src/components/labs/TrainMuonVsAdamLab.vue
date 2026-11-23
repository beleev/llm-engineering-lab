<!--
  Muon vs Adam (对应 llm_train/m14_muon_optimizer:run, 缩小到 16×16 在浏览器里真跑 150 步)。
  只讲一件事: Adam 的逐元素缩放只能修 "与坐标轴对齐" 的病态; 把同一个病态问题旋转一下, Adam 就修不动了, 而 Muon 的正交化与基无关。
  10 × 150 步不放在 computed 里: 参数变了才在渲染之后算一次, 结果按 (旋转, 种子) 缓存。
-->
<template>
  <LabFrame
    title="Muon vs Adam — 同一个病态问题, 把病态方向转一下, 谁还修得动?"
    sub="- 模拟什么: 考察一个病态条件的矩阵回归任务: 目标是拟合一个 16×16 的权重矩阵, 输入特征在 16 个主轴方向上的标准差从 1 等比衰减至 0.01 (Hessian 条件数达 1e4), 导致不同方向上的梯度幅值相差悬殊, 单一全局步长无法兼顾强弱方向。本实验台在浏览器内实时运行两种优化器, 各自扫描 5 组候选学习率并训练 150 步 (偏置项 bias 统一采用 Adam 更新, 学习率全程线性退火至 0, 与离线 Python demo 配置严格一致)。对比两种优化器的预条件机制:
  - Adam: 执行逐元素自适应缩放, 每个矩阵参数独立除以自身梯度二阶矩的平方根 $\sqrt{v}$, 该机制仅在病态主轴恰好与参数坐标轴对齐时才能准确解耦各方向尺度。
  - Muon: 通过 Newton–Schulz 迭代对整个矩阵更新量作正交化处理, 使每个奇异方向获得等幅更新步长, 更新行为与坐标系旋转朝向无关。
- 图上是什么: 横轴为训练步数 (范围为 0 至 150), 纵轴为回归训练 loss (对数刻度)。
  - 两条实线分别绘制 Adam 与 Muon 在当前选定学习率下的训练损失收敛轨迹, 颜色与上方控件区标识对应, 曲线末端标注所选学习率及最终收敛 loss。
  - 控件区的两排色块列出两种优化器在 5 档候选学习率下的最终 loss, 带绿色描边的格子标识该优化器的最优学习率档位, 高亮底色标识当前绘制在图上的学习率档位。
- 可以做什么: 控件从上到下依次为:
  - 点击「病态方向与坐标轴对齐」或「随机旋转 (不对齐)」切换输入协方差主轴的摆放朝向 (矩阵条件数与本征优化难度完全不变, 仅通过随机正交矩阵将病态主轴旋转至坐标轴之外)。
  - 点击「换一组数据」更换随机种子重新训练全部配置 (共需实时计算 10 条曲线乘 150 步, 需短暂等待)。
  - 点击任意学习率格子将该档位对应的损失曲线切换至主图显示。
  - 点击「选最好」一键将两种优化器同时切回各自的最优学习率档位。
- 应该看到: 在坐标轴对齐配置下, Adam 的最优最终 loss 与 Muon 持平甚至略优; 切换至随机旋转配置后, Adam 的最优成绩显著退化, 而 Muon 的收敛曲线几乎不受影响。右侧读数含义如下:
  - 「Adam 最好 / Muon 最好」: 各自在 5 档候选学习率中取得的最低最终 loss (括号内注明对应的最优学习率)。
  - 「各自最好的相比」: 两种优化器最优最终 loss 的比值, 直接给出领先倍数。
  - 「图上画的是不是各自最好」: 提示当前主图对比是否处于各自调优后的公平基准下, 若未同时选中最优档位则「不是」标红提示。"
    module="llm_train/m14"
    run="python -m llm_train.m14_muon_optimizer.demo"
    :challenge="{
      ask: '先观察「病态方向与坐标轴对齐」模式: 哪种优化器表现更优? 再切换至「随机旋转 (不对齐)」模式: 问题的条件数与本征难度并未改变, 为什么 Adam 的最优成绩显著退化, 而 Muon 几乎不受影响?',
      answer: '结论:\n- 坐标轴对齐时 Adam 略优或与 Muon 基本持平。\n- 随机旋转后 Muon 显著胜出, 因为 Adam 的自适应缩放绑定在参数坐标轴上, 而 Muon 的正交化直接作用于矩阵的奇异方向。\n机制解析:\n- 坐标轴对齐场景: 输入数据的 16 个病态主方向恰好对应矩阵的 16 个坐标分量, 每个坐标轴各自对应单一尺度。Adam 对每个矩阵元素独立乘以 $1/\\sqrt{v}$ 调整步长, 恰好等价于沿主轴逐一拉平曲率差异, 正好处于其逐元素对角预条件假设成立的理想情形 (更换多组随机种子观察可见 Adam 从小幅领先到显著领先不等; 离线 Python demo 在 32×32 规模下实测 Adam 最优 loss 领先 1.9 倍)。\n- 随机旋转场景: 优化问题的特征值谱与条件数丝毫未变, 但每个病态主方向变成了全体坐标分量的稠密线性组合。此时单个矩阵元素的梯度同时混杂了强曲率方向与弱曲率方向的投影分量, 逐元素缩放既无法单独放大弱方向也无法单独抑制强方向, 导致 Adam 的最优收敛成绩大幅恶化。\n- Muon 保持旋转不变的原因: Newton–Schulz 迭代直接将矩阵的全体奇异值推向统一量级, 对输入空间施加正交旋转仅旋转梯度的奇异向量而完全不改变奇异值谱, 因此 Muon 在旋转前后的更新轨迹几乎完全一致 (离线 Python demo 实测随机旋转下 Muon 最优 loss 领先 Adam 达 9.2 倍)。\n真实神经网络隐藏层权重矩阵的病态曲率方向并无理由恰好与参数坐标轴对齐, 其几何结构更接近随机旋转后的耦合情形, 这正是 Muon 在大模型预训练中提升算力效率的核心机理。对比实验时需确认右侧「图上画的是不是各自最好」处于开启状态: 唯有双方均取各自网格搜索的最优学习率时比较才具备公平性, 否则测量的只是学习率失配误差。',
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
        <p>★ 公平比较优化器要求双方各自取最优学习率后再作对比, 否则反映的是学习率调参偏差而非算法本身的性能差异。色块内小字显示对应学习率的最终收敛 loss, 绿色描边标出各自的最优档位; 训练发散为 NaN 的档位自动排除在最优评选之外。</p>
        <p>两种方案中的偏置项 bias 统一采用 Adam 更新, 因为 Muon 的矩阵正交化仅适用于 2-D 权重矩阵, 一维向量参数不存在矩阵奇异值分解结构。学习率全程线性退火至 0, 与离线 Python demo 保持一致。</p>
        <p>优化器显存开销对比: Adam 需为每个权重矩阵维护 2 份状态张量 (一阶动量与二阶矩), 而 Muon 仅需维护 1 份一阶动量张量, 常驻状态显存反而减半。</p>
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
