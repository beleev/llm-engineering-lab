<!--
  DPO loss 曲面实验台 (对应 llm_finetune/methods/dpo.py: DPOLoss)。
  只讲一件事: loss = -log σ(β·Δ), 它对 Δ 的梯度权重是 σ(-β·Δ) ——
  模型已经排对的偏好对几乎不产生梯度, 训练自动把力气花在排错的样本上。
  Δ = (log π_c − log π_r) − (log ref_c − log ref_r), 即 policy 相对 ref 的隐式奖励差。
-->
<template>
  <LabFrame
    title="DPO — 一条 logsigmoid 曲线, 看哪些偏好对还在产生梯度"
    sub="- 模拟什么: DPO (Direct Preference Optimization, 直接偏好优化) 在包含同一 prompt 下优选回复 chosen 与劣选回复 rejected 的成对偏好数据上训练, 无需显式拟合独立奖励模型, 直接优化策略模型 policy 使其相对冻结参考模型 ref (通常为 SFT 起点) 提升 chosen 的相对对数似然。单个偏好对的当前优化状态可浓缩为标量隐式奖励差 $\Delta$, 表示 policy 相对 ref 在 chosen 上的对数概率增幅超出 rejected 的部分: $\Delta \gt 0$ 代表偏好排序正确, $\Delta \lt 0$ 代表排序颠倒。单样本损失函数定义为 $\text{loss} = -\log\sigma(\beta\Delta)$, 随 $\Delta$ 增大损失单调下降且曲线趋于平缓, 对应样本梯度权重自然衰减。
- 图上是什么: 横轴表示隐式奖励差 $\Delta$, 左侧淡红色背景标示 $\Delta \lt 0$ 的排序错误区域。各视觉元素含义如下:
  - 上方紫色曲线: 单个偏好对损失值随 $\Delta$ 变化的函数曲线, 水平参考线标出初始点损失 $\ln 2 = 0.693$ (对应训练第 0 步的理论损失)。
  - 下方橙色曲线: 该偏好对分配到的梯度权重 $\sigma(-\beta\Delta)$, 取值范围为 0 至 1。
  - 6 个带编号的大圆点: 代表单个 batch 内的 6 组偏好对样本, 落入右侧排对区呈绿色, 落入左侧排错区呈红色。
  - 下方联动小橙点: 垂直对应上方每个样本大圆点, 标示该样本当前获得的梯度权重。
- 可以做什么: 支持以下交互操作:
  - 左右拖动任意带编号大圆点 (或聚焦后使用方向键): 修改该偏好对的 $\Delta$, 模拟训练更新推动样本迁移的过程。
  - 点击「从 ref 出发」按钮: 将全部 6 个样本重置回 $\Delta = 0$ (对应训练起点处 policy 从 ref 精确复制的初始状态)。
  - 点击「训练中途」按钮: 随机生成一组训练中期的样本 $\Delta$ 分布。
  - 拖动 $\beta$ 滑杆: 调节损失曲线与梯度权重曲线的陡峭程度。
- 应该看到: 样本点越靠右, 单点损失越低且梯度权重越趋近于 0, 已充分拉开优势的样本自动淡出参数更新; 样本点越靠左, 梯度权重越接近 1, 排序错误的样本获得优先梯度纠偏。右侧汇总读数含义如下:
  - 「batch 平均 loss」: 当前 6 个偏好对损失值的算术平均。
  - 「偏好准确率」: 满足 $\Delta \gt 0$ 的样本在当前 batch 中的占比。
  - 「排错样本拿走的梯度」: 红色排错样本的梯度权重之和占全批总权重的比例; 当该占比高于排错样本条数占比时显示为绿色, 表明梯度自动向错排困难样本集中。
  - 「最靠右的样本的权重」: 当前裕度最大的正确样本所剩余的梯度权重, 低于 0.05 时显示为绿色。"
    module="llm_finetune/methods/dpo.py"
    run="python -m llm_finetune.run_finetune.dpo.train_dpo"
    :challenge="{
      ask: '请按顺序完成以下操作并回答:\n1. 点击「从 ref 出发」: 此时 6 个样本点的 loss 均为多少? 原因是什么?\n2. 将其中一个样本点拖至 $\\Delta = +8$, 再将温度参数 $\\beta$ 从 0.1 拖至 1.0: 该样本点的梯度权重如何变化?\n3. 超参数 $\\beta$ 在偏好优化中实际控制什么边界?',
      answer: '结论:\n1. 点击「从 ref 出发」时, 全部 6 个样本点的 loss 严格等于 0.693。\n2. 当 $\\beta$ 从 0.1 增至 1.0 时, 位于 $\\Delta = 8$ 处的样本梯度权重从 0.31 降至 0.0003。\n3. 超参数 $\\beta$ 控制隐式奖励差 $\\Delta$ 达到多大裕度后停止继续拉大分差, 即约束策略相对参考分布的允许偏离幅度。\n机制解析:\n- 第 0 步一致性自检: 训练起点处 policy 直接复制自 ref, 因此任意偏好对均满足 $\\Delta = 0$, 初始损失恒为 $-\\log\\sigma(0) = \\ln 2 \\approx 0.693$, 梯度权重恒为 0.5。工程实践中, 若 DPO 第 0 步损失偏离 0.693, 即可直接判定 policy 与 ref 的初始化权重、分词或掩码计算未对齐。\n- 饱和裕度与乘积缩放: 样本梯度权重 $\\sigma(-\\beta\\Delta)$ 完全由无量纲乘积 $\\beta\\Delta$ 决定。对于同一个处于 $\\Delta = 8$ 的已排对样本:\n  - 当 $\\beta = 0.1$ 时, 乘积仅为 0.8, 梯度权重仍高达 $\\sigma(-0.8) \\approx 0.31$, 优化器会继续大幅推高分差并驱使策略远离 ref。\n  - 当 $\\beta = 1$ 时, 乘积增至 8, 梯度权重衰减至 $\\sigma(-8) \\approx 0.0003$, 该样本上的更新基本饱和终止。\n- KL 约束物理意义: 理论推导中 $\\beta$ 直接对应 RLHF 目标函数里的 KL 散度惩罚系数; $\\beta$ 越大策略越紧贴 ref 保持保守更新, $\\beta$ 越小则越纵容策略大幅偏离 ref 以追求更大的偏好裕度。\n- 过度偏离 ref 的退化风险: 单纯提升成对偏好排序裕度并不保证自回归生成质量同步改善。本仓库 train_dpo 训练 200 步实测显示:\n  - 留出集偏好准确率虽从 0.965 升至 0.996。\n  - 模型赋予优选回复 chosen 的对数概率 $\\log\\pi$ 却从 -4.03 降至 -4.18。\n  - 实际贪心解码的完全匹配率 EM 从 0.332 退化至 0.137。',
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
        <p><Tex text="单样本损失对隐式奖励差的梯度为 $\partial\,\text{loss}/\partial\Delta = -\beta\cdot\sigma(-\beta\Delta)$, 单个偏好对驱动参数更新的相对权重直接对应下方橙色曲线的纵坐标高度。" /></p>
        <ul class="pts">
          <li><b>权重接近 0 的样本:</b> 等于自动退出训练。</li>
          <li><b>权重接近 1 的样本 (排得很错):</b> 以满额梯度被纠正。</li>
        </ul>
        <p>这一自适应加权机制与逻辑回归分类中「错分样本主导决策边界更新、高置信度正确样本梯度趋零」完全同构。</p>
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
