<!--
  QAT 实验台 (对应 llm_finetune/methods/qat.py: fake_quant / quantize_dequantize)。
  只讲一件事: RTN 对每个权重各自就近取整, 误差互不相干; QAT 让下游 loss 看见量化后的权重,
  STE 把梯度传回影子权重, 影子权重越过格点边界, 让一些误差反号, 和相邻的误差互相抵消。
  玩具是一个 32 维线性层的二次损失 L = (ŵ − w*)ᵀ Σ (ŵ − w*), Σ_ij = ρ^|i−j| 是输入特征的相关性。
  量化与 Python 相同: 每 g 个权重一组, scale = 组内 absmax / q_max, 对称取整到 [−q_max, q_max];
  简化: scale 按 w* 定好后固定, Python 每步按当前影子权重重算。
-->
<template>
  <LabFrame
    title="QAT: 让取整误差互相抵消"
    sub="- 模拟什么: 一个 32 维线性层已经训好, 权重是 w*。部署要把它量化到低位宽: 每 g 个权重一组, 组内按最大绝对值定 scale, 每个权重取整成 scale 的整数倍 (INT2 只有 −1, 0, +1 三档, INT4 有 −7 到 +7)。玩具里 scale 按 w* 定好后固定。RTN (训练后量化) 各自就近取整; QAT 从 w* 出发再训 300 步, 前向用量化后的 ŵ 算 loss, 反向用 STE 把梯度原样交给高精度的影子权重。loss 是 ŵ 偏离 w* 带来的输出误差, 输入特征相邻维度的相关系数是 ρ。
- 图上是什么: 上图是 QAT 训练过程中量化后权重的 loss (对数纵轴), 虚线是 RTN 的 loss, 圆点是当前查看的步数。下图是 32 个权重的取整误差 ŵ − w*: 上排是 RTN, 下排是 QAT 在当前步数的结果, 向上为正, 向下为负。
- 可以做什么: 选位宽和组大小; 拖「输入相关性 ρ」滑杆; 拖上图的圆点或「查看第几步」滑杆, 看 QAT 走到哪一步; 换一组权重。
- 应该看到: ρ 大时 QAT 的误差正负交替, 相邻误差在输出里互相抵消, loss 多半降到 RTN 的几分之一 (换几组权重, 个别组会不降); ρ = 0 时各维独立, 就近取整已经最优, QAT 学不到东西, 末步还比 RTN 差。右侧读数:
  - 「RTN loss」与「QAT loss」是两种做法量化后的输出误差, QAT 低于 RTN 时变绿。
  - 「每个权重的位数」是 bits + 16/g (每组一个 fp16 scale), 组越小越准, 但 scale 的开销越大。"
    module="llm_finetune/methods/qat.py"
    run="python -m llm_finetune.run_finetune.qat.train_qat"
    :challenge="{
      ask: 'INT2, g=8, 把 ρ 从 0.9 拖到 0。QAT 相对 RTN 的优势会怎样变化? 为什么?',
      answer: '结论: 默认这组权重, ρ=0.9 时 QAT 的 loss 是 RTN 的 0.33; ρ=0 时变成 1.16, 比 RTN 还差。\n- 输入相关时, 输出误差是 (ŵ − w*)ᵀ Σ (ŵ − w*), 相邻两个误差一正一负会互相抵消。RTN 各自就近取整, 看不到这种抵消; QAT 的梯度经过 Σ, 会把一部分影子权重推过格点边界, 故意取 \'远\' 的那个格点, 换来整体误差更小。看下排的误差条, 正负交替得比上排规律。\n- 输入不相关时 Σ = I, loss 就是各维误差的平方和, 就近取整已经最优, 没有可以抵消的东西。STE 的梯度一直把影子权重往误差的反方向推, 推过边界后换到另一个格点, 误差更大, 又被推回来, 最后停在边界附近, 有一半的机会停在较差的那一侧。\n- Python 的语言模型实验里, INT2 g=32 直接 RTN 的 EM 0.176, QAT 100 步后 0.961; 多训 100 步但不加假量化的对照只有 0.164。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="n in [2, 3, 4]" :key="'b' + n" type="button" :class="{ active: bits === n }" @click="bits = n">INT{{ n }}</button>
        <button v-for="n in [4, 8, 16, 32]" :key="'g' + n" type="button" :class="{ active: group === n }" @click="group = n">g={{ n }}</button>
        <button type="button" @click="seed++">换一组权重</button>
      </div>
      <LabSlider v-model="rho" label="输入相关性 ρ" :min="0" :max="0.95" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="t" label="查看第几步" :min="0" :max="STEPS" :step="1" />
    </template>

    <svg ref="svgEl" viewBox="0 0 560 300" role="group" aria-label="QAT 的 loss 曲线与两种取整误差">
      <text x="8" y="14" class="t">量化后权重的 loss (对数刻度), 虚线 = RTN</text>
      <line x1="40" x2="540" :y1="ly(run.rtn)" :y2="ly(run.rtn)" stroke="var(--eye)" stroke-dasharray="5 4" />
      <polyline :points="run.curve.map((l, i) => `${lx(i)},${ly(l)}`).join(' ')" fill="none" stroke="var(--accent)" stroke-width="1.5" />
      <circle class="draggable" :cx="lx(t)" :cy="ly(run.curve[t])" r="7" fill="var(--accent)" stroke="var(--bg-card)" stroke-width="2"
              tabindex="0" role="slider" aria-label="查看第几步" :aria-valuenow="t" aria-valuemin="0" :aria-valuemax="STEPS"
              @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => (t = Math.round(clamp((x - 40) / 500 * STEPS, 0, STEPS))) })"
              @keydown.right.prevent="t = Math.min(STEPS, t + 10)" @keydown.left.prevent="t = Math.max(0, t - 10)" />
      <text x="540" y="140" text-anchor="end" class="t">第 {{ t }} 步</text>
      <line x1="40" x2="540" y1="150" y2="150" stroke="var(--border)" />
      <text x="8" y="172" class="t">取整误差 ŵ − w*</text>
      <g v-for="(row, ri) in errRows" :key="ri">
        <text x="8" :y="ri ? 262 : 206" class="t">{{ ri ? 'QAT' : 'RTN' }}</text>
        <line x1="40" x2="540" :y1="ri ? 258 : 202" :y2="ri ? 258 : 202" stroke="var(--border)" />
        <rect v-for="(e, i) in row" :key="i" :x="44 + i * 15.5" :y="(ri ? 258 : 202) - Math.max(e, 0) * EK" width="10"
              :height="Math.abs(e) * EK + 0.5" :fill="ri ? 'var(--accent)' : 'var(--eye)'" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>RTN loss</span><b>{{ run.rtn.toFixed(3) }}</b></div>
      <div class="kv"><span>QAT loss (第 {{ t }} 步)</span><b :class="run.curve[t] < run.rtn ? 'good' : 'bad'">{{ run.curve[t].toFixed(3) }}</b></div>
      <div class="kv"><span>QAT / RTN</span><b :class="run.curve[t] < run.rtn ? 'good' : 'bad'">{{ (run.curve[t] / run.rtn).toFixed(2) }}</b></div>
      <div class="kv"><span>每个权重的位数 bits + 16/g</span><b>{{ (bits + 16 / group).toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>Python 实测 (reverse 任务的 2 层模型, 留出集 EM), 实测, 不随左侧变化:</p>
        <ul class="pts">
          <li><b>RTN:</b> INT4 各组大小都是 0.996; INT2 g=8 / 32 / 64 为 0.559 / 0.176 / 0.004。</li>
          <li><b>QAT 100 步再 RTN:</b> INT2 g=8 / 32 / 64 为 0.977 / 0.961 / 0.953; 不加假量化多训 100 步的对照 0.484 / 0.164 / 0.000。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const D = 32, STEPS = 300, LR = 0.0025, EK = 30
const bits = ref(2), group = ref(8), rho = ref(0.9), t = ref(STEPS), seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

// 与 quantize_dequantize 相同的分组对称取整; 简化: scale 按 w* 的组内 absmax 定好后固定 (Python 每步按影子权重重算)
const quant = (w, ref) => {
  const qm = 2 ** (bits.value - 1) - 1, out = []
  for (let s = 0; s < D; s += group.value) {
    const blk = w.slice(s, s + group.value)
    const sc = Math.max(...ref.slice(s, s + group.value).map(Math.abs), 1e-8) / qm
    blk.forEach((x) => out.push(clamp(Math.round(x / sc), -qm, qm) * sc))
  }
  return out
}

const run = computed(() => {
  const rand = mulberry32(seed.value * 104729)
  const ws = range(D).map(() => randn(rand))
  const S = range(D).map((i) => range(D).map((j) => rho.value ** Math.abs(i - j)))
  const Sv = (e) => S.map((row) => row.reduce((a, s, j) => a + s * e[j], 0))
  const loss = (wh) => { const e = wh.map((x, i) => x - ws[i]); return Sv(e).reduce((a, x, i) => a + x * e[i], 0) }
  const rtnW = quant(ws, ws)
  let w = ws.slice()
  const curve = [], snaps = []
  for (let k = 0; k <= STEPS; k++) {
    const wh = quant(w, ws)
    curve.push(loss(wh)); snaps.push(wh)
    // ★ STE: ∂L/∂w := ∂L/∂ŵ = 2Σ(ŵ − w*), 取整当恒等; 学习率线性衰减到 0
    const g = Sv(wh.map((x, i) => x - ws[i]))
    w = w.map((x, i) => x - LR * (1 - k / STEPS) * 2 * g[i])
  }
  return { ws, rtnW, rtn: loss(rtnW), curve, snaps }
})

const errRows = computed(() => [run.value.rtnW, run.value.snaps[t.value]].map((wh) => wh.map((x, i) => clamp(x - run.value.ws[i], -0.9, 0.9))))
const lmax = computed(() => Math.max(run.value.rtn, ...run.value.curve) * 1.5)
const lmin = computed(() => Math.max(1e-6, Math.min(run.value.rtn, ...run.value.curve) / 1.5))
const lx = (i) => 40 + (i / STEPS) * 500
const ly = (l) => 130 - (Math.log(l / lmin.value) / Math.log(lmax.value / lmin.value)) * 105
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
