<!--
  Mamba 选择性扫描实验台 (对应 llm_models/layers/sparse/ssm.py:SelectiveSSM.forward)。
  只讲一件事: h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B·x_t 里的 Δ_t 由输入决定 ——
  Δ 大 = "忘掉过去, 记住这个"; Δ 小 = "这个 token 当没看见, 状态原样传下去"。
-->
<template>
  <LabFrame
    title="Mamba 选择性扫描 — 每个 token 自带的步长 Δ 决定它被记住还是被略过"
    sub="- 模拟什么: Mamba 是一类选择性状态空间模型 (SSM), 它不缓存历史 token 序列, 仅维护固定维度的隐状态 $h$, 每读入一个 token 按离散化递推式 $h_t = \exp(-\Delta_t \cdot a) \cdot h_{t-1} + \Delta_t \cdot x_t$ 更新一次: 第一项控制历史状态的保留比例 ($\Delta_t$ 越大保留越少), 第二项控制当前输入的写入强度 ($\Delta_t$ 越大写入越多)。所谓「选择性」, 是指步长 $\Delta_t$ 由当前 token 动态计算得出, 即 $\mathrm{softplus}(\mathrm{Linear}(x_t))$, 使模型能够对关键信息赋予较大的 $\Delta_t$ (清除旧状态并强力写入), 对无关冗余词赋予接近 0 的 $\Delta_t$ (跳过写入并将已有状态原样向后传递)。对照组为传统线性时不变状态空间模型 (LTI SSM, 如 S4), 所有 token 共享同一个固定步长 $\Delta$。示例序列「密码 是 7 嗯 那个 就是 然后 呢 请 回答」中, 仅有「7」是需要长期保留的关键信息。
- 图上是什么: 自上而下分为三个联动视图:
  - 上排柱图: 各位置 token 的步长 $\Delta_t$ (选择性模式下可直接拖动柱顶圆点), 关键位置「7」以高亮色标出。
  - 中间折线: 隐状态幅值 $h_t$ 随序列推进的变化轨迹 (此处将各位置的输入标量 $x_t$ 统一设为 1, 以便单独观察 $\Delta_t$ 的门控作用)。
  - 下排柱图: 最终时刻隐状态 $h_T$ 中各历史 token 的贡献占比, 总和为 100%, 单项贡献等于该位置的初始写入量乘以其后每一步的保留率。
- 可以做什么: 按界面控件顺序操作:
  - 「选择性」与「LTI」: 切换步长 $\Delta$ 是按每个 token 独立取值还是全局共享单一数值。
  - 「一键理想选择」: 将「7」的 $\Delta$ 设为 3, 其余位置的 $\Delta$ 设为 0.02。
  - 「全部重置为 0.5」: 将所有位置的 $\Delta$ 恢复为初始值 0.5。
  - 拖动「衰减速率 a」: 调节连续状态矩阵参数 $A = -a$; $a$ 越大, 相同 $\Delta$ 下历史状态衰减越快。
  - 「全局 Δ」滑杆: 仅在 LTI 模式下显示, 用于统一调节全局步长。
  - 拖动 Δ 柱: 在选择性模式下直接上下拖动任意位置的 $\Delta$ 柱 (选中后按键盘上下方向键每次增减 0.1)。
- 应该看到: 对比两种模式下的状态构成与右侧读数:
  - 选择性模式下: 将「7」的 $\Delta$ 调高, 并将其后 7 个冗余词的 $\Delta$ 调至接近 0, 下排柱图中「7」的最终占比可超过 90%; 中间折线在「7」处显著跃升, 随后基本保持水平 (冗余词既不写入也不触发衰减)。
  - LTI 模式下: 无论如何调节全局 $\Delta$, 「7」在最终状态中的占比始终无法提升。
  - 「『7』在最终状态的占比」: 超过 50% 视为有效保留目标信息。
  - 「『7』的信号强度」: 「7」对 $h_T$ 的绝对贡献幅值 (初始写入量乘以后续各步保留率之积)。
  - 「经过『7』时旧状态保留 exp(−Δ·a)」: 接近 0 表示读入「7」时前缀旧状态被基本清空, 接近 1 表示前缀状态几乎原样保留。"
    module="llm_models/layers/sparse/ssm.py"
    run="python -m llm_models.run_models.language_models.mamba.infer_mamba"
    :challenge="{
      ask: '先切换至「LTI (所有 token 共用一个 Δ)」模式, 将全局 Δ 从最小值一路拖至最大值: 「7」在最终状态中的占比最高能达到多少? 原因是什么? 随后切回「选择性」模式, 尝试仅调整两类 token 的 Δ 使「7」的占比超过 90%。',
      answer: '答案: LTI 模式下「7」的最终占比最高仅为 10%, 且必须将全局 $\\Delta$ 调至趋近最小值才能达到; 切回选择性模式后, 只需调大「7」的 $\\Delta$ 并将后续 7 个冗余词的 $\\Delta$ 调至接近 0, 占比即可超过 90%。\nLTI 模式受限的原因:\n- 在 LTI 模式下, 所有位置共享同一个 $\\Delta$, 第 $j$ 个 token 对最终状态 $h_T$ 的贡献为 $\\Delta \\cdot \\exp(-\\Delta \\cdot a \\cdot (T - 1 - j))$: 各位置初始写入量均为 $\\Delta$, 此后每经过一个后续 token 就要乘一次保留率 $\\exp(-\\Delta \\cdot a)$, 越靠前的 token 经历的指数衰减次数越多。\n- 「7」之后仍有 7 个冗余 token, 因此其贡献占比必然严格低于其后任意一个冗余词。若调大 $\\Delta$, 末尾最后一个 token 将独占状态; 若令 $\\Delta$ 趋近于 0, 单步保留率趋近于 1, 全部 10 个 token 趋于等权平均, 「7」作为其中之一最高只能占 10%, 这便是 LTI 模式的理论上界。\n- 线性时不变系统 (LTI) 的权重仅取决于位置距序列末尾的相对距离, 无法根据输入内容本身的重要程度做选择性记忆。\n选择性机制生效的原因: Mamba 令 $\\Delta_t = \\mathrm{softplus}(\\mathrm{Linear}(x_t))$ 随当前输入动态变化:\n- 对关键 token「7」赋予较大的 $\\Delta$: 既增强当步写入, 又使保留率 $\\exp(-\\Delta \\cdot a) \\approx 0$ 从而清空前缀无关状态 (对应右侧第三项读数)。\n- 对后续冗余 token 赋予接近 0 的 $\\Delta$: 当步写入量趋近于 0, 同时保留率趋近于 1, 隐状态无损穿透冗余区间, 「7」的信号不会被稀释。\n由此, 同一个输入相关的 $\\Delta_t$ 同时充当了写入门与遗忘门; 相应代价是状态转移变为时变系统, 无法像 S4 那样等价展开为固定全局卷积核并行训练, 需改用硬件感知的并行前缀扫描 (parallel scan) 算法。',
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
      <p class="lab-note">Mamba 隐状态的空间开销为常数级 ($O(1)$), 处理 10 个 token 与处理 10 万个 token 所需的缓存体积完全相同; 标准注意力的 KV cache 则随序列长度线性增长。</p>
      <div class="lab-note">
        <template v-if="selective">
          <p><Tex text="每个位置的步长 $\Delta_t$ 同时承担两项门控职能:" /></p>
          <ul class="pts">
            <li><Tex text="控制当前输入的写入强度 $\Delta_t \cdot x_t$。" /></li>
            <li><Tex text="同时控制对历史隐状态的保留率 $\exp(-\Delta_t \cdot a)$。" /></li>
          </ul>
          <p><Tex text="将冗余 token 的 $\Delta$ 调至 0 附近时, 该步既不写入新分量也不衰减旧状态, 前序信息得以原样向后传递。" /></p>
        </template>
        <p v-else><Tex text="LTI 模式: 所有 token 共享单一 $\Delta$, 各位置的最终权重仅取决于距序列末尾的距离, 末尾 token 始终占据最大份额; 「7」的占比上界为 $1/10$ (当全局 $\Delta \to 0$ 时各位置趋于等权)。" /></p>
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
