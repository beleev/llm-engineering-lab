<!--
  DPO loss 曲面实验台 (对应 llm_finetune/methods/dpo.py: DPOLoss)。
  只讲一件事: loss = -log σ(β·Δ), 它对 Δ 的梯度权重是 σ(-β·Δ) ——
  模型已经排对的偏好对几乎不产生梯度, 训练自动把力气花在排错的样本上。
  Δ = (log π_c − log π_r) − (log ref_c − log ref_r), 即 policy 相对 ref 的隐式奖励差。
-->
<template>
  <LabFrame
    title="DPO — 一条 logsigmoid 曲线, 看哪些偏好对还在产生梯度"
    sub="- 模拟什么: DPO (直接偏好优化) 的训练数据是「同一个 prompt 下, 一条更好的回答 chosen 和一条更差的 rejected」。它不训奖励模型, 直接让 policy 相对 ref (冻结的 SFT 起点) 更偏向 chosen、更远离 rejected。每个偏好对的状态可以压缩成一个数 $\Delta$: policy 相对 ref, 给 chosen 的 log 概率比给 rejected 的多涨了多少 (论文里叫隐式奖励差)。$\Delta \gt 0$ 表示这一对已经排对, $\Delta \lt 0$ 表示排反了。单个偏好对的损失为 $\text{loss} = -\log\sigma(\beta\Delta)$: $\Delta$ 越大损失越小且曲线渐趋平坦, 梯度随之自然衰减。
      - 图上是什么: 横轴是 $\Delta$, 左半边淡红色底是排错区。各部分:
        - 上面的紫色曲线: 单个偏好对的 loss 随 $\Delta$ 的变化, 横线标出 $\ln 2 = 0.693$ (训练第 0 步的 loss)。
        - 下面的橙色曲线: 这个偏好对分到的梯度权重 $\sigma(-\beta\Delta)$, 取值 0 到 1。
        - 6 个带编号的大圆点: 一个 batch 里的 6 个偏好对, 绿色在排对区, 红色在排错区。
        - 小橙点: 每个大圆点下方连着一个, 是它当前的梯度权重。
      - 可以做什么: 能动的有:
        - 左右拖任意大圆点 (或聚焦后按方向键): 改这一对的 $\Delta$, 相当于训练把它推着走。
        - 「从 ref 出发」: 把 6 个点都放回 $\Delta = 0$ (policy 刚从 ref 复制出来的状态)。
        - 「训练中途」: 换一组随机的 $\Delta$。
        - 拖 $\beta$: 改曲线的陡峭程度。
      - 应该看到: 点越往右, loss 越小, 梯度权重也越接近 0, 说明已经排对的样本会自动退出训练; 点越往左权重越接近 1, 排错的样本以满额梯度被纠正。右侧读数:
        - 「batch 平均 loss」: 6 个点 loss 的均值。
        - 「偏好准确率」: 有几成的点在 $\Delta \gt 0$。
        - 「排错样本拿走的梯度」: 红点的权重之和占全部权重的比例。它高于红点的条数占比时变绿, 说明梯度在往排错的样本上集中。
        - 「最靠右的样本的权重」: 排得最对的那个点还剩多少梯度, 低于 0.05 变绿。"
    module="llm_finetune/methods/dpo.py"
    run="python -m llm_finetune.run_finetune.dpo.train_dpo"
    :challenge="{
      ask: '按顺序做:\n1. 点「从 ref 出发」: 6 个点的 loss 都是多少? 为什么?\n2. 把一个点拖到 Δ = +8, 再把 β 从 0.1 拖到 1.0: 这个点的梯度权重怎么变?\n3. β 到底在控制什么?',
      answer: '答案:\n1. 「从 ref 出发」时 6 个点的 loss 都是 0.693。\n2. β 从 0.1 拖到 1.0 时, Δ = 8 的点的梯度权重从 0.31 掉到 0.0003。\n3. β 控制的是「Δ 要涨到多大才算够」。\n为什么:\n- 训练第 0 步 policy 就是 ref 的复制, 所以每一对的 $\\Delta = 0$, loss 恒为 $-\\log\\sigma(0) = \\ln 2 \\approx 0.693$, 梯度权重恒为 0.5。跑 DPO 时第一步 loss 不是 0.693, 说明 ref 和 policy 没对齐, 这是最好用的自检。\n- 梯度权重 $\\sigma(-\\beta\\Delta)$ 完全取决于乘积 $\\beta\\Delta$。同样是 $\\Delta = 8$ 的样本:\n  - $\\beta = 0.1$ 时: 乘积只有 0.8, 权重还有 $\\sigma(-0.8) \\approx 0.31$, 模型会继续把它往右推, 离 ref 越来越远。\n  - $\\beta = 1$ 时: 乘积是 8, 权重只剩 $\\sigma(-8) \\approx 0.0003$, 基本停手。\n- 所以 $\\beta$ 大等于贴近 ref、保守; $\\beta$ 小等于允许偏离 ref 很远去拉大差距。它对应 RLHF 目标里 KL 惩罚的系数。\n- 偏离 ref 的代价要盯着看: 偏好排得更准了, 生成的回答反而更差。本仓库 train_dpo 实测 200 步:\n  - 偏好准确率从 0.965 升到 0.996。\n  - 但模型给 chosen 的 $\\log\\pi$ 从 −4.03 掉到 −4.18。\n  - 贪心解码的 EM 从 0.332 掉到 0.137。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="deltas = deltas.map(() => 0)">从 ref 出发 (全部 Δ = 0)</button>
        <button type="button" @click="reseed">训练中途 (换一组)</button>
      </div>
      <LabSlider v-model="beta" label="β" :min="0.05" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 330" role="group" aria-label="DPO loss 与梯度权重曲线">
      <!-- 排错区 (Δ<0) 底色 -->
      <rect :x="sx(-XM)" y="14" :width="sx(0) - sx(-XM)" height="290" fill="var(--danger)" opacity="0.06" />
      <text :x="sx(-XM) + 6" y="28" class="t" fill="var(--danger)">Δ &lt; 0: 排错了 (rejected 涨得更多)</text>
      <text :x="sx(XM) - 6" y="28" class="t" text-anchor="end" fill="var(--left)">Δ &gt; 0: 排对了</text>
      <line :x1="sx(0)" :x2="sx(0)" y1="14" y2="304" stroke="var(--border-strong)" stroke-dasharray="3 3" />

      <!-- 上: loss -->
      <path :d="lossPath" fill="none" stroke="var(--accent)" stroke-width="2" />
      <line :x1="sx(-XM)" :x2="sx(XM)" :y1="ly(Math.LN2)" :y2="ly(Math.LN2)" stroke="var(--border)" />
      <text :x="sx(XM) - 4" :y="ly(Math.LN2) - 4" class="t" text-anchor="end">ln 2 = 0.693</text>
      <text :x="sx(0) + 8" y="46" class="t" fill="var(--accent)">loss = −log σ(βΔ)</text>
      <!-- 下: 梯度权重 -->
      <path :d="wPath" fill="none" stroke="var(--eye)" stroke-width="2" />
      <line :x1="sx(-XM)" :x2="sx(XM)" :y1="wy(0)" :y2="wy(0)" stroke="var(--border)" />
      <text x="6" :y="wy(1) - 4" class="t" fill="var(--eye)">梯度权重 σ(−βΔ)</text>
      <text v-for="t in [-8, -4, 0, 4, 8]" :key="t" :x="sx(t)" y="322" class="t" text-anchor="middle">{{ t }}</text>

      <g v-for="(p, i) in pts" :key="i">
        <line :x1="sx(p.d)" :x2="sx(p.d)" :y1="ly(p.loss)" :y2="wy(p.w)" stroke="var(--text-dim)" stroke-width="0.7" />
        <circle :cx="sx(p.d)" :cy="wy(p.w)" r="5" fill="var(--eye)" />
        <!-- ★ 手放在 Δ 上: 拖动 = 这个偏好对被训练推着走 -->
        <circle
          class="draggable" :cx="sx(p.d)" :cy="ly(p.loss)" r="10"
          :fill="p.d >= 0 ? 'var(--left)' : 'var(--danger)'" stroke="var(--bg-card)" stroke-width="2"
          tabindex="0" role="slider" :aria-label="`偏好对 ${i + 1} 的 Δ`" :aria-valuenow="p.d" :aria-valuemin="-XM" :aria-valuemax="XM"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setD(i, (x - sx(0)) / SC) })"
          @keydown.right.prevent="setD(i, p.d + 0.5)" @keydown.left.prevent="setD(i, p.d - 0.5)"
        />
        <text :x="sx(p.d)" :y="ly(p.loss) + 3.5" text-anchor="middle" class="n">{{ i + 1 }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>batch 平均 loss (6 个点的均值)</span><b>{{ meanLoss.toFixed(3) }}</b></div>
      <div class="kv"><span>偏好准确率 (Δ &gt; 0 的点占几成)</span><b :class="acc === 1 ? 'good' : ''">{{ (acc * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>排错样本拿走的梯度 (红点权重占比)</span><b :class="wrongShare > 1 - acc ? 'good' : ''">{{ (wrongShare * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>最靠右的样本的权重 (还剩多少梯度)</span><b :class="minW < 0.05 ? 'good' : ''">{{ minW.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p><Tex text="梯度 $\partial\,\text{loss}/\partial\Delta = -\beta\cdot\sigma(-\beta\Delta)$: 单个偏好对推动参数更新的权重直接对应下方橙色曲线的高度。" /></p>
        <ul class="pts">
          <li><b>权重接近 0 的样本:</b> 等于自动退出训练。</li>
          <li><b>权重接近 1 的样本 (排得很错):</b> 以满额梯度被纠正。</li>
        </ul>
        <p>这和分类里「只有分错的样本推动决策边界」是同一件事。</p>
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
import { clamp, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const XM = 10, SC = 29, LMAX = 3.2
const sx = (d) => 320 + d * SC
const ly = (l) => 200 - Math.min(l, LMAX) / LMAX * 150     // loss 面板: y∈[50,200]
const wy = (w) => 300 - w * 70                             // 权重面板: y∈[230,300]

const beta = ref(0.3)
const seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

const sample = (s) => { const rand = mulberry32(s * 4099); return range(6).map(() => clamp(Math.round((1.5 + randn(rand) * 4) * 2) / 2, -XM, XM)) }
const deltas = ref(sample(1))
const reseed = () => { seed.value++; deltas.value = sample(seed.value) }
const setD = (i, d) => { deltas.value = deltas.value.map((v, k) => (k === i ? clamp(Math.round(d * 10) / 10, -XM, XM) : v)) }

const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))   // 数值稳定的 log(1+e^z)
const lossOf = (d) => softplus(-beta.value * d)            // ★ −log σ(βΔ) = softplus(−βΔ)
const wOf = (d) => sigmoid(-beta.value * d)                // ★ 梯度权重

const pts = computed(() => deltas.value.map((d) => ({ d, loss: lossOf(d), w: wOf(d) })))
const curve = (f, y) => range(101).map((k) => { const d = -XM + k * 0.2; return `${k ? 'L' : 'M'}${sx(d).toFixed(1)},${y(f(d)).toFixed(1)}` }).join(' ')
const lossPath = computed(() => curve(lossOf, ly))
const wPath = computed(() => curve(wOf, wy))

const meanLoss = computed(() => sum(pts.value.map((p) => p.loss)) / pts.value.length)
const acc = computed(() => pts.value.filter((p) => p.d > 0).length / pts.value.length)
// 排错样本分到的梯度占比。它高于排错样本的条数占比 (1 − acc), 说明梯度在往排错的样本上集中
const wrongShare = computed(() => sum(pts.value.filter((p) => p.d < 0).map((p) => p.w)) / sum(pts.value.map((p) => p.w)))
const minW = computed(() => Math.min(...pts.value.map((p) => p.w)))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.n { font-size: 10px; fill: var(--bg-card); font-weight: 600; pointer-events: none; }
</style>
