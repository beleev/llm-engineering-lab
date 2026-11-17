<!--
  Mamba 选择性扫描实验台 (对应 llm_models/layers/sparse/ssm.py:SelectiveSSM.forward)。
  只讲一件事: h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B·x_t 里的 Δ_t 由输入决定 ——
  Δ 大 = "忘掉过去, 记住这个"; Δ 小 = "这个 token 当没看见, 状态原样传下去"。
-->
<template>
  <LabFrame
    title="Mamba 选择性扫描 — 每个 token 自带的步长 Δ 决定它被记住还是被略过"
    sub="- 模拟什么: Mamba 是一种状态空间模型 (SSM): 它不存历史 token, 只维护一个固定大小的隐状态 $h$, 每读一个 token 更新一次: $h_t = \exp(-\Delta_t \cdot a) \cdot h_{t-1} + \Delta_t \cdot x_t$。第一项是旧状态保留多少 (Δ 越大保留越少), 第二项是当前 token 写入多少 (Δ 越大写得越多)。「选择性」指 $\Delta_t$ 由当前 token 自己算出来 (softplus(Linear(x_t))), 所以模型可以给重要的 token 大 Δ (清掉过去、记住它), 给废话 Δ ≈ 0 (当没看见, 状态原样传下去)。对照组是老式的 LTI SSM (如 S4): 所有 token 共用一个 Δ。这句话「密码 是 7 嗯 那个 就是 然后 呢 请 回答」里只有「7」值得记。
      - 图上是什么: 从上到下三部分:
        - 上排柱子: 每个 token 的 $\Delta_t$ (选择性模式下柱顶圆点可以拖), 「7」那根是高亮色。
        - 中间折线: 隐状态 $h_t$ 的大小随 token 变化 (这里把每个 token 的输入 $x$ 都取 1, 只看 Δ 的作用)。
        - 下排柱子: 最终状态 $h_T$ 里每个 token 占的比例, 加起来 100%; 算法是「它写入的量 × 它之后每一步的保留率」。
      - 可以做什么: 按界面顺序:
        - 「选择性」和「LTI」: 切换 Δ 是每个 token 一个还是全局一个。
        - 「一键理想选择」: 把「7」的 Δ 设成 3、其余设成 0.02。
        - 「全部重置为 0.5」: 回到起点。
        - 拖「衰减速率 a」: 改 A = −a, a 越大, 同样的 Δ 下旧状态忘得越快。
        - 「全局 Δ」滑杆: LTI 模式下出现。
        - 拖 Δ 柱: 选择性模式下直接上下拖任意一根 Δ 柱 (键盘上下键每次 0.1)。
      - 应该看到: 先对比两种模式, 再看右侧读数:
        - 选择性模式下: 把「7」的 Δ 拖高、把它后面 7 个废话的 Δ 拖到接近 0, 下排「7」的占比能超过 90%; 折线在「7」处跳起, 之后几乎水平 (废话既不写入也不衰减)。
        - LTI 模式下: 无论怎么拖全局 Δ, 「7」的占比都上不去。
        - 「『7』在最终状态的占比」: 超过 50% 算成功。
        - 「『7』的信号强度」: 它的绝对贡献 (写入量乘后续保留率)。
        - 「经过『7』时旧状态保留 exp(−Δ·a)」接近 0 表示读到「7」时之前的内容被清空, 接近 1 表示几乎全留。"
    module="llm_models/layers/sparse/ssm.py"
    run="python -m llm_models.run_models.language_models.mamba.infer_mamba"
    :challenge="{
      ask: '先切到「LTI (所有 token 共用一个 Δ)」, 把全局 Δ 从最小拖到最大都试一遍。「7」的占比最高能到多少? 为什么? 再切回「选择性」, 只改两类 token 的 Δ 就让占比超过 90%。',
      answer: '答案: LTI 下「7」的占比最高只有 10%, 而且要 Δ 拖到最小才到。选择性下给「7」大 Δ、给 7 个废话 Δ ≈ 0, 占比就超过 90%。\n为什么 LTI 不行:\n- LTI 下所有 token 共用一个 Δ, token j 对最终状态的贡献是 $\\Delta \\cdot \\exp(-\\Delta \\cdot a \\cdot (T-1-j))$: 写入量都是 Δ, 之后每过一个 token 就乘一次保留率 $\\exp(-\\Delta \\cdot a)$, 越早的 token 被衰减的次数越多。\n- 「7」后面还有 7 个 token, 所以它的占比永远低于它后面的任何一个废话。Δ 拖大, 最后一个 token 独占; Δ 拖到 0, 保留率趋近 1, 每个 token 占比相同, 「7」是 10 个里的 1 个, 占 10%, 这是 LTI 下的上限。\n- 线性时不变系统 (LTI) 只能按「距离现在多远」加权, 不能按「内容重不重要」加权。\n为什么选择性可以: Mamba 让 $\\Delta_t = \\mathrm{softplus}(\\mathrm{Linear}(x_t))$ 由当前输入决定:\n- 给「7」大 Δ: 写入强, 同时保留率 $\\exp(-\\Delta \\cdot a) \\approx 0$ 把之前的状态清掉 (看右侧第三个读数)。\n- 给废话 Δ ≈ 0: 写入约为 0, 且保留率约为 1, 状态原样通过, 「7」的信号不被稀释。\n同一个随输入变化的 Δ 同时控制写入门和遗忘门。代价: 系统变成时变的, 不能像 S4 那样写成一个固定卷积核并行训练, 只能用并行 scan。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: selective }" @click="selective = true">选择性 (每个 token 自己的 Δ)</button>
        <button type="button" :class="{ active: !selective }" @click="selective = false">LTI (所有 token 共用一个 Δ, 如 S4)</button>
        <button type="button" @click="ideal">一键理想选择</button>
        <button type="button" @click="flat">全部重置为 0.5</button>
      </div>
      <LabSlider v-model="a" label="衰减速率 a (A = −a)" :min="0.2" :max="3" :step="0.1" :format="(t) => t.toFixed(1)" />
      <LabSlider v-if="!selective" v-model="dGlobal" label="全局 Δ" :min="0.02" :max="3" :step="0.02" :format="(t) => t.toFixed(2)" />
    </template>

    <svg ref="svg" viewBox="0 0 540 330" role="group" aria-label="选择性扫描">
      <text x="6" y="12" class="sec">每个 token 的步长 Δ_t (选择性模式下可拖)</text>
      <text x="6" y="142" class="sec">隐状态 h_t 的大小</text>
      <text x="6" y="242" class="sec">每个 token 在最终状态 h_T 中的占比</text>
      <line :x1="X0" :x2="X1" :y1="DY1" :y2="DY1" class="base" /><line :x1="X0" :x2="X1" :y1="HY1" :y2="HY1" class="base" /><line :x1="X0" :x2="X1" :y1="CY1" :y2="CY1" class="base" />
      <g v-for="(tk, j) in TOKENS" :key="j">
        <rect :x="cx(j) - 14" :y="dy(sim.d[j])" width="28" :height="DY1 - dy(sim.d[j])" class="dbar" :class="{ key: j === KEY }" />
        <text :x="cx(j)" :y="dy(sim.d[j]) - 4" class="num">{{ sim.d[j].toFixed(2) }}</text>
        <g v-if="selective" class="draggable" tabindex="0" role="slider" :aria-label="`token ${tk} 的 Δ`" :aria-valuenow="deltas[j]"
          @pointerdown="start($event, { svg, onMove: ({ y }) => setD(j, y) })"
          @keydown.up.prevent="bump(j, 0.1)" @keydown.down.prevent="bump(j, -0.1)">
          <rect :x="cx(j) - 18" :y="DY0 - 6" width="36" :height="DY1 - DY0 + 12" fill="transparent" />
          <circle :cx="cx(j)" :cy="dy(sim.d[j])" r="6" class="handle" />
        </g>
        <rect :x="cx(j) - 14" :y="CY1 - sim.share[j] * (CY1 - CY0)" width="28" :height="sim.share[j] * (CY1 - CY0)" class="cbar" :class="{ key: j === KEY }" />
        <text :x="cx(j)" :y="CY1 - sim.share[j] * (CY1 - CY0) - 3" class="num">{{ (sim.share[j] * 100).toFixed(0) }}%</text>
        <text :x="cx(j)" y="322" class="tok" :class="{ key: j === KEY }">{{ tk }}</text>
      </g>
      <polyline :points="hLine" class="hline" />
      <circle v-for="(h, j) in sim.h" :key="'h' + j" :cx="cx(j)" :cy="hy(h)" r="3" class="hdot" />
    </svg>

    <template #stats>
      <div class="kv"><span>「7」在最终状态的占比 (超过 50% 算成功)</span><b :class="sim.share[KEY] > 0.5 ? 'good' : 'bad'">{{ (sim.share[KEY] * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>「7」的信号强度 (写入量 × 之后的保留率)</span><b>{{ sim.contrib[KEY].toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="读到「7」时旧状态保留 $\exp(-\Delta \cdot a)$ (0 = 清空, 1 = 全留)" /></span><b>{{ Math.exp(-sim.d[KEY] * a).toFixed(2) }}</b></div>
      <p class="lab-note">隐状态的大小是固定的 (O(1)), 读 10 个 token 和读 10 万个 token 存的东西一样多。KV cache 则随 token 数一直增长。</p>
      <div class="lab-note">
        <template v-if="selective">
          <p><Tex text="每个 $\Delta_t$ 同时干两件事:" /></p>
          <ul class="pts">
            <li><Tex text="决定写入强度 $\Delta_t \cdot x_t$。" /></li>
            <li><Tex text="也决定对旧状态的保留率 $\exp(-\Delta_t \cdot a)$。" /></li>
          </ul>
          <p><Tex text="把废话 token 的 $\Delta$ 拖到 0 附近, 它既不写入也不衰减, 前面的信息「穿过」它原样传到后面。" /></p>
        </template>
        <p v-else><Tex text="LTI 模式: 所有 token 共用一个 $\Delta$, 每个 token 的贡献只取决于它离结尾多远, 最后一个 token 永远占大头。「7」的占比上限是 $1/10$ (全局 $\Delta \to 0$ 时每个 token 占比相同)。" /></p>
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
import { clamp, sum } from '@/utils/labmath.js'

const TOKENS = ['密码', '是', '7', '嗯', '那个', '就是', '然后', '呢', '请', '回答'], KEY = 2, DMAX = 3
const selective = ref(true), a = ref(1), dGlobal = ref(0.5)
const deltas = ref(TOKENS.map(() => 0.5))
const svg = ref(null)
const { start } = useDrag()

const sim = computed(() => {
  const d = selective.value ? deltas.value : TOKENS.map(() => dGlobal.value)
  const h = []
  let cur = 0
  for (let t = 0; t < d.length; t++) { cur = Math.exp(-d[t] * a.value) * cur + d[t] * 1; h.push(cur) } // ★ h_t = Ā_t·h_{t-1} + Δ_t·B·x_t (B = x = 1)
  // token j 在 h_T 里的贡献 = 写入量 × 之后每一步的保留率
  const contrib = d.map((dj, j) => dj * Math.exp(-a.value * sum(d.slice(j + 1))))
  const tot = sum(contrib) || 1
  return { d, h, contrib, share: contrib.map((c) => c / tot) }
})

const X0 = 60, X1 = 530, DY0 = 22, DY1 = 120, HY0 = 150, HY1 = 222, CY0 = 256, CY1 = 306
const cx = (j) => X0 + 24 + j * ((X1 - X0 - 48) / (TOKENS.length - 1))
const dy = (v) => DY1 - (v / DMAX) * (DY1 - DY0)
const hMax = computed(() => Math.max(1, ...sim.value.h))
const hy = (v) => HY1 - (v / hMax.value) * (HY1 - HY0)
const hLine = computed(() => sim.value.h.map((h, j) => `${cx(j)},${hy(h).toFixed(1)}`).join(' '))
const setD = (j, y) => { deltas.value = deltas.value.map((v, i) => (i === j ? clamp(Math.round(((DY1 - y) / (DY1 - DY0)) * DMAX * 50) / 50, 0.02, DMAX) : v)) }
const bump = (j, by) => { deltas.value = deltas.value.map((v, i) => (i === j ? clamp(v + by, 0.02, DMAX) : v)) }
const ideal = () => { selective.value = true; deltas.value = TOKENS.map((_, j) => (j === KEY ? 3 : 0.02)) }
const flat = () => { deltas.value = TOKENS.map(() => 0.5) }
</script>

<style scoped>
.sec { font-size: 10px; fill: var(--text-dim); }
.base { stroke: var(--border-strong); }
.dbar { fill: color-mix(in srgb, var(--accent) 35%, transparent); stroke: var(--accent); }
.dbar.key, .cbar.key { fill: color-mix(in srgb, var(--eye) 45%, transparent); stroke: var(--eye); }
.cbar { fill: color-mix(in srgb, var(--text-dim) 35%, transparent); stroke: var(--text-dim); }
.handle { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.draggable:focus-visible .handle { stroke: var(--text); }
.num { text-anchor: middle; font-size: 9px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.tok { text-anchor: middle; font-size: 11px; fill: var(--text-muted); }
.tok.key { fill: var(--eye); font-weight: 700; }
.hline { fill: none; stroke: var(--left); stroke-width: 2; }
.hdot { fill: var(--left); }
</style>
