<!--
  重要性比率 + clip 实验台 (PPO / GRPO 的 clipped surrogate, DAPO 的 clip-higher)。
  只讲一件事: 目标 min(ρÂ, clip(ρ, 1−ε_low, 1+ε_high)Â) 在阴影区里是平的 —— 梯度为 0,
  这个 token 本轮不再被推。上界只拦"想继续涨"的 token, 而真正会撞到它的只有低概率 (探索) token。
-->
<template>
  <LabFrame
    title="ratio clip — 一批样本更新多轮时, 哪些 token 这一步已经不许再推"
    sub="- 模拟什么: PPO 与 GRPO 为复用同一批高成本在线采样执行多轮参数更新, 自第二轮内层迭代起当前策略已偏离采样策略, 需引入重要性比率 $\rho = \pi_{\text{new}}/\pi_{\text{old}}$ 修正分布偏移。为防止单轮策略更新幅度过大, 代理目标取 $\min(\rho\hat{A},\ \text{clip}(\rho, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\hat{A})$: 当 $\rho$ 沿提升目标函数的有利方向越出区间 $[1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}}]$ 时切换为常数截断项, 对 $\rho$ 的导数归零, 暂停继续推高或压低该 token。优势值 $\hat{A}$ 为正表示应提高该 token 概率, 为负表示应压低其概率; DAPO 的 clip-higher 机制则单独放宽上界 $\varepsilon_{\text{high}}$。
- 图上是什么: 横轴为重要性比率 $\rho$, 图中各视觉元素对应关系如下:
  - 竖直虚线: $\rho = 1$, 对应新旧策略概率相等的初始位置。
  - 两条黄色竖线: 两侧裁剪边界 $1-\varepsilon_{\text{low}}$ 与 $1+\varepsilon_{\text{high}}$。
  - 灰色虚线: 未加裁剪的原始线性目标 $\rho\hat{A}$。
  - 紫色折线: 经 $\min$ 与 $\text{clip}$ 截断后的实际代理目标曲线。
  - 黄色阴影区: 目标函数斜率归零、梯度截断为 0 的区域。
  - 右侧灰色区域: 满足 $\rho \gt 1/\pi_{\text{old}}$, 对应新概率 $\pi_{\text{new}} \gt  1$ 的概率公理不可达区域。
  - 可拖动圆点: 当前 token 所处的比率 $\rho$, 绿色表示仍具非零梯度, 黄色表示已触发裁剪。
- 可以做什么: 按界面自上而下可进行以下操作:
  - 切换「$\hat{A} \gt 0$ / $\hat{A} \lt 0$」按钮: 选择正优势或负优势样本情形。
  - 点击「PPO/GRPO 对称 0.2」或「DAPO clip-higher 0.2/0.28」: 载入两组裁剪阈值预设。
  - 拖动「ε_low」与「ε_high」滑杆: 独立调整左右裁剪边界宽度。
  - 拖动「π_old」滑杆: 在 0.01 至 0.89 范围内调节旧策略赋予该 token 的概率, 决定右侧灰色不可达区域的起始位置。
  - 左右拖动圆点 (或聚焦后使用方向键): 改变当前重要性比率 $\rho$。
- 应该看到: 当 $\hat{A} \gt 0$ 时仅右侧出现黄色截断阴影 (仅限制概率上浮), 当 $\hat{A} \lt 0$ 时仅左侧出现黄色截断阴影 (仅限制概率下压)。右侧读数含义如下:
  - 第一行: 将当前比率 $\rho$ 换算为新策略概率 $\rho\,\pi_{\text{old}}$。
  - 「这个 token 的梯度」: 未裁剪时显示绿色 ±1·|Â|, 触发裁剪后显示红色 0。
  - 「上界内 π 最多涨到」: 单轮裁剪窗口允许新策略达到的概率上限 $\pi_{\text{old}}(1+\varepsilon_{\text{high}})$。
  - 「上界碰得到吗」: 当 $\pi_{\text{old}}$ 较大时, 概率上限 $1/\pi_{\text{old}}$ 小于 $1+\varepsilon_{\text{high}}$, 裁剪上界落入不可达区, 该高概率 token 实际上不受上界约束。"
    module="llm_finetune/methods/grpo.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '固定优势 Â 大于 0, 按顺序完成以下观察:\n1. 将 π_old 拖至最右侧 (约 0.89): 该 token 能否触及 $1+\\varepsilon_{\\text{high}}$ 裁剪上界?\n2. 将 π_old 拖至 0.01, 在对称裁剪阈值 $\\varepsilon = 0.2$ 下, 该 token 单轮更新最多能升至多大概率?\n3. 切换为 DAPO 的非对称阈值 0.2 / 0.28 后, 单轮概率上限变为多少?',
      answer: '结论:\n1. 当 $\\pi_{\\text{old}} \\approx 0.89$ 时无法触及裁剪上界。\n2. 当 $\\pi_{\\text{old}} = 0.01$ 且采用对称阈值 0.2 时, 单轮概率最多升至 0.012。\n3. 切换为 DAPO 的上界阈值 0.28 后, 单轮概率上限升至 0.0128。\n机制解析:\n- 受概率公理 $\\pi_{\\text{new}} \\le 1$ 约束, 重要性比率满足 $\\rho = \\pi_{\\text{new}}/\\pi_{\\text{old}} \\le 1/\\pi_{\\text{old}}$。\n- 高概率 token ($\\pi_{\\text{old}} \\approx 0.89$): 比率 $\\rho$ 的理论最大值约为 1.12, 始终低于 1.2 的裁剪上界, 因此完全不受上界截断约束, 可直接推向概率 1。\n- 低概率探索 token ($\\pi_{\\text{old}} = 0.01$): 单轮更新被严格限制在 $0.01 \\times 1.2 = 0.012$ 以内, 每轮相对增幅不超过 20%, 必须连续多轮获得正优势才能显著提升概率。这种相对比例裁剪天然偏向既有高概率 token, 容易导致策略分布迅速尖锐化并引发策略熵塌缩。\n- DAPO 的 clip-higher 机制单独将上界放宽至 0.28 ($0.01 \\times 1.28 = 0.0128$), 为低概率正向探索 token 留出更大抬升空间; 同时维持下界 0.2 不变, 防止负优势更新单步将候选概率压近 0 而加剧熵塌缩。\n- 裁剪机制仅在「单批采样复用执行多轮内层更新」时生效: 首轮更新时 $\\pi_{\\text{new}} = \\pi_{\\text{old}}$, 比率 $\\rho$ 恒等于 1, 必然处于裁剪区间内部, 此时截断分支不会触发。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: A > 0 }" @click="A = 1">Â &gt; 0 (好 token, 想推高)</button>
        <button type="button" :class="{ active: A < 0 }" @click="A = -1">Â &lt; 0 (坏 token, 想压低)</button>
        <button type="button" @click="preset(0.2, 0.2)">PPO/GRPO 对称 0.2</button>
        <button type="button" @click="preset(0.2, 0.28)">DAPO clip-higher 0.2/0.28</button>
      </div>
      <LabSlider v-model="eLo" label="ε_low" :min="0.05" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="eHi" label="ε_high" :min="0.05" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="logPi" label="π_old (旧策略给它的概率)" :min="-2" :max="-0.05" :step="0.05" :format="() => piOld.toFixed(3)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 330" role="group" aria-label="clipped surrogate 目标随 ratio 变化">
      <!-- 梯度为 0 的区域 -->
      <rect :x="flat.x" y="20" :width="flat.w" height="290" fill="var(--warn)" opacity="0.14" />
      <text :x="flat.x + flat.w / 2" y="36" text-anchor="middle" class="t" fill="var(--warn)">梯度 = 0</text>
      <line x1="50" x2="622" :y1="sy(0)" :y2="sy(0)" stroke="var(--border-strong)" />
      <line :x1="sx(1)" :x2="sx(1)" y1="20" y2="310" stroke="var(--border-strong)" stroke-dasharray="3 3" />
      <text :x="sx(1)" y="325" text-anchor="middle" class="t">ρ = 1</text>
      <!-- 两个 ε 的标签放在图的上沿: ε 很小时不会和下沿的「ρ = 1」叠在一起 -->
      <text :x="sx(1 - eLo) - 3" y="14" text-anchor="end" class="t">1−ε_low</text>
      <text :x="sx(1 + eHi) + 3" y="14" text-anchor="start" class="t">1+ε_high</text>
      <line v-for="e in [1 - eLo, 1 + eHi]" :key="e" :x1="sx(e)" :x2="sx(e)" y1="20" y2="310" stroke="var(--warn)" stroke-width="1" />
      <!-- 未裁剪的 ρÂ (虚线) 与裁剪后的目标 -->
      <line :x1="sx(0)" :y1="sy(0)" :x2="sx(RMAX)" :y2="sy(RMAX * A)" stroke="var(--text-dim)" stroke-dasharray="4 4" />
      <path :d="objPath" fill="none" stroke="var(--accent)" stroke-width="2.5" />
      <!-- π_new ≤ 1 → ρ ≤ 1/π_old -->
      <g v-if="rhoCap < RMAX">
        <rect :x="sx(rhoCap)" y="20" :width="sx(RMAX) - sx(rhoCap)" height="290" fill="var(--text-dim)" opacity="0.25" />
        <text :x="sx(RMAX) - 4" y="300" text-anchor="end" class="t">ρ &gt; 1/π_old: 不可能 (π_new &gt; 1)</text>
      </g>
      <!-- ★ 手放在 ρ 上 -->
      <circle
        class="draggable" :cx="sx(rho)" :cy="sy(obj(rho))" r="10" :fill="grad ? 'var(--left)' : 'var(--warn)'" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="重要性比率 ρ" :aria-valuenow="rho" aria-valuemin="0" :aria-valuemax="RMAX"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setRho((x - 50) / 260) })"
        @keydown.right.prevent="setRho(rho + 0.05)" @keydown.left.prevent="setRho(rho - 0.05)"
      />
    </svg>

    <template #stats>
      <div class="kv"><span><Tex text="当前 $\rho$, 换算成新概率 $\pi_{\text{new}} = \rho\,\pi_{\text{old}}$" /></span><b>{{ rho.toFixed(2) }} → {{ (rho * piOld).toFixed(4) }}</b></div>
      <div class="kv"><span><Tex text="这个 token 的梯度 $\partial L/\partial\rho$" /></span><b :class="grad ? 'good' : 'bad'">{{ grad ? (A > 0 ? '+1·|Â|' : '−1·|Â|') : '0 (被 clip)' }}</b></div>
      <div class="kv"><span><Tex text="上界内 $\pi$ 这一轮最多涨到 $\pi_{\text{old}}(1+\varepsilon_{\text{high}})$" /></span><b>{{ Math.min(1, piOld * (1 + eHi)).toFixed(4) }}</b></div>
      <div class="kv"><span>上界碰得到吗</span><b :class="rhoCap > 1 + eHi ? 'good' : 'bad'">{{ rhoCap > 1 + eHi ? '碰得到' : '碰不到 (1/π_old 不到 1+ε, 不受约束)' }}</b></div>
      <div class="lab-note">
        <p><Tex text="外层 $\min(\cdot)$ 算子使裁剪仅在「顺着优势优化的有利方向」上生效: $\hat{A} \gt 0$ 时仅截断上界 (防止过度推高), $\hat{A} \lt 0$ 时仅截断下界 (防止过度压低)。" /></p>
        <p><Tex text="若某个 token 因其他参数耦合更新而偏离目标方向 (例如 $\hat{A} \gt 0$ 但 $\rho$ 已跌至 0.5, 正优势 token 概率反而下降), $\min(\cdot)$ 会保留未截断项的完整梯度以将其拉回合理区间。" /></p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

const RMAX = 2.2
const sx = (r) => 50 + r * 260
const sy = (v) => 165 - v * 62

const A = ref(1), eLo = ref(0.2), eHi = ref(0.2), logPi = ref(-1), rho = ref(1.1)
const svgEl = ref(null)
const { start } = useDrag()

const preset = (lo, hi) => { eLo.value = lo; eHi.value = hi }
const piOld = computed(() => 10 ** logPi.value)
const rhoCap = computed(() => 1 / piOld.value)
const setRho = (r) => { rho.value = clamp(Math.round(r * 100) / 100, 0, Math.min(RMAX, rhoCap.value)) }
watch(rhoCap, () => setRho(rho.value))

// ★ PPO clipped surrogate: min(ρÂ, clip(ρ, 1−ε_low, 1+ε_high)·Â)
const obj = (r) => Math.min(r * A.value, clamp(r, 1 - eLo.value, 1 + eHi.value) * A.value)
// 梯度非零 ⇔ min 选中的是未裁剪的那一项
const grad = computed(() => (A.value > 0 ? rho.value <= 1 + eHi.value : rho.value >= 1 - eLo.value))
const flat = computed(() => (A.value > 0
  ? { x: sx(1 + eHi.value), w: sx(RMAX) - sx(1 + eHi.value) }
  : { x: sx(0), w: sx(1 - eLo.value) - sx(0) }))
const objPath = computed(() => [0, 1 - eLo.value, 1 + eHi.value, RMAX]
  .map((r, i) => `${i ? 'L' : 'M'}${sx(r).toFixed(1)},${sy(obj(r)).toFixed(1)}`).join(' '))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
