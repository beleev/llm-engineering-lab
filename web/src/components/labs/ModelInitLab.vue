<!--
  初始化实验台: 为什么"刚建好的模型"第一步 loss 应该 ≈ ln V, 以及绑权重 + std=1 怎么把它炸到 250。
  对应 llm_models/utils/init.py: std=0.02 时 LLaMA 首步 CE 7.073 (run_models/language_models/llama/README.md 实测), ln 1000 = 6.908。
-->
<template>
  <LabFrame
    title="初始化 — 刚建好的模型, 第一步 loss 该是多少?"
    sub="- 模拟什么: 训练第 1 步时权重均为随机数, 模型尚未学到任何分布规律, 理想状态下应对词表中的 V 个 token 给出近似均匀的概率预测, 此时交叉熵损失 (CE) 恰好等于 $\ln V$。若首步 CE 远大于 $\ln V$, 说明模型在初始阶段就对随机 token 给出了过高置信度, 根源通常在于权重初始化的标准差 $\sigma$。本实验模拟前向的最后一步: 最后一层归一化后的 hidden 向量 h 与输出头 lm_head 各行做点积得到 logits, 再计算 CE。若进一步将 lm_head 与输入 embedding 共享同一权重矩阵 (weight tying), 残差连接会使 h 仍与输入 token 自身的 embedding 高度同向, 导致模型给输入 token 自身打出高达 $\sigma \cdot d$ 的 logit, 而预测目标通常并非输入 token 本身。对应实现见 llm_models/utils/init.py。
- 图上是什么: 一张双对数坐标曲线图:
  - 坐标轴: 横轴为初始化标准差 $\sigma$ (对数刻度), 纵轴为第 1 步的 CE (对数刻度)。
  - 曲线: 各 $\sigma$ 取值下的首步 CE 蒙特卡洛估计值 (随机采样 24 组 logits 取平均)。
  - 虚线: 均匀预测的理论参考线 $\ln V$。
  - 圆点: 当前选定的 $\sigma$ 位置, 支持直接拖动。
- 可以做什么: 按界面控件顺序操作:
  - 「绑权重」: 切换 lm_head 与输入 embedding 是否共享权重矩阵。
  - 两个快捷按钮: 分别跳转至 $\sigma = 0.02$ (本仓库 init_weights 采用的默认值) 与 $\sigma = 1$ (PyTorch nn.Embedding 的默认值)。
  - 「换一组随机权重」: 重新采样随机权重。
  - 三根滑杆: 拖动「嵌入初始化 $\sigma$」「模型宽度 d」「词表大小 V」, 观察首步 CE 曲线的变化。
- 应该看到: 在默认配置 ($\sigma = 1$、$d = 256$、$V = 1000$、开启「绑权重」) 下, 首步 CE 约为 250, 远高于理论参考值 $\ln V = 6.91$; 将 $\sigma$ 调至 0.02 后, 首步 CE 降至约 7.1, 贴近参考线。右侧读数含义如下:
  - 「首步 CE」: 与 $\ln V$ 之差小于 0.5 时视为正常 (与 train 脚本中的断言阈值一致)。
  - 「自己的 logit ≈ $\sigma \cdot d$」: 开启权重绑定时输入 token 自身获得的 logit 幅度。
  - 「其余 logit 的标准差 $\sigma \sqrt{d}$」: 其余候选 token 分数的标准差; 当该值远大于 1 时 softmax 分布已高度尖锐, 即使关闭权重绑定, 首步 CE 仍会远高于 $\ln V$。"
    module="llm_models/utils/init.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '保持 $\\sigma = 1$ 不变, 仅关闭「绑权重」, 首步 CE 会回到 $\\ln V$ 吗? 为什么真实的 LLaMA 模型既不绑定权重、也不乘 $\\sqrt{d}$, 却不会出现首步损失异常偏高的问题?',
      answer: '不会回到 $\\ln V$。\n- 关闭「绑权重」后, 给输入 token 自身打高分的偏置项消失, 但其余候选 logit 的标准差仍为 $\\sigma \\cdot \\sqrt{d} = 1 \\times 16 = 16$。不同 token 的 logit 差距可达十几分, softmax 输出依然高度尖锐, 模型仍会对某个随机 token 给出极高概率, 首步 CE 仍高达数十。\n- 问题的根源在于 logit 整体尺度过大, 权重绑定只是进一步放大了偏差。将 $\\sigma$ 降至 0.02 才能从根本上解决: 此时 logit 标准差降至 $0.02 \\times 16 \\approx 0.3$, 各候选 token 得分接近, softmax 分布近似均匀, 首步 CE 随之回到 $\\ln V$ 附近。\n真实 LLaMA 直接采用 $\\sigma = 0.02$ 初始化 (即 init_weights 的默认值), 从源头消除了 logit 尺度过大与自内积偏置两个问题, 因此既无需针对权重绑定做特殊修正, 也无需额外乘以 $\\sqrt{d}$ 缩放。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: tied }" @click="tied = !tied">绑权重 lm_head = embedding: {{ tied ? '开' : '关' }}</button>
        <button type="button" @click="logSigma = Math.log10(0.02)">σ = 0.02 (本仓库 init_weights 用的值)</button>
        <button type="button" @click="logSigma = 0">σ = 1 (PyTorch nn.Embedding 默认)</button>
        <button type="button" @click="seed++">换一组随机权重</button>
      </div>
      <LabSlider v-model="logSigma" label="嵌入初始化 σ (随机权重的标准差)" :min="-2.5" :max="0.3" :step="0.05" :format="(v) => (10 ** v).toPrecision(2)" />
      <LabSlider v-model="d" label="模型宽度 d" :min="32" :max="512" :step="32" />
      <LabSlider v-model="V" label="词表大小 V" :min="100" :max="2000" :step="100" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="首步交叉熵随初始化标准差的变化曲线, 曲线上的圆点可拖">
      <line :x1="PAD" :x2="W - 8" :y1="yOf(lnV)" :y2="yOf(lnV)" class="ref" />
      <text :x="W - 10" :y="yOf(lnV) - 5" class="lbl" text-anchor="end">ln V = {{ lnV.toFixed(2) }} (均匀瞎猜)</text>
      <polyline :points="curve" class="curve" />
      <!-- ★ 直接拖这个点改 σ: 手放在被解释的那个量上 -->
      <circle
        :cx="xOf(logSigma)" :cy="yOf(ce)" r="8" class="knob draggable" tabindex="0" role="slider"
        aria-label="拖动改变初始化标准差" :aria-valuenow="sigma"
        @pointerdown="drag.start($event, { svg: $event.target.ownerSVGElement, onMove: ({ x }) => (logSigma = clamp(sOf(x), -2.5, 0.3)) })"
        @keydown.left.prevent="logSigma = clamp(logSigma - 0.05, -2.5, 0.3)"
        @keydown.right.prevent="logSigma = clamp(logSigma + 0.05, -2.5, 0.3)"
      />
      <text v-for="t in [-2, -1, 0]" :key="t" :x="xOf(t)" :y="H - 6" class="lbl" text-anchor="middle">σ = {{ 10 ** t }}</text>
      <text :x="4" :y="14" class="lbl">首步 CE (对数轴)</text>
    </svg>

    <template #stats>
      <div class="kv"><span>首步 CE (随机抽 24 组取平均)</span><b :class="ok ? 'good' : 'bad'">{{ ce.toFixed(2) }}</b></div>
      <div class="kv"><span><Tex text="$\ln V$ (均匀瞎猜时的 CE)" /></span><b>{{ lnV.toFixed(2) }}</b></div>
      <div class="kv"><span><Tex text="绑权重时「自己」的 logit $\approx \sigma \cdot d$" /></span><b>{{ tied ? (sigma * d).toFixed(1) : '— (没绑, 这一项不存在)' }}</b></div>
      <div class="kv"><span><Tex text="其余 logit 的标准差 $\sigma \cdot \sqrt{d}$" /></span><b>{{ (sigma * Math.sqrt(d)).toFixed(2) }}</b></div>
      <p class="lab-note">
        <Tex :text="ok ? '✓ 接近 $\\ln V$: 模型从「均匀瞎猜」出发, 之后 loss 下降才说明它学到了东西。'
              : '✗ 远高于 $\\ln V$: 模型一上来就极度自信地猜错。这时「loss 在下降」只是在纠正初始化, 不代表学会了任务。'" />
      </p>
      <div class="lab-note">
        <p><Tex text="简化假设: 最后一层归一化输出的 hidden 向量范数约为 $\sqrt{d}$, 且受残差连接传递影响, 其方向仍与输入 token 的 embedding 保持接近。该近似模型复现了本仓库的实测读数:" /></p>
        <ul class="pts">
          <li><Tex text="当 $\sigma = 1$、$d = 256$、$V = 1000$ 时, 首步 CE 约为 250。" /></li>
          <li><Tex text="当 $\sigma = 0.02$ 时, 首步 CE 约为 7.1。" /></li>
        </ul>
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
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const logSigma = ref(0)
const d = ref(256)
const V = ref(1000)
const tied = ref(true)
const seed = ref(1)
const drag = useDrag()

const sigma = computed(() => 10 ** logSigma.value)
const lnV = computed(() => Math.log(V.value))

// 首步 CE 的蒙特卡洛估计: 目标 token 是随机的 (几乎从不等于输入 token 自己)
const ceAt = (sig, dd, vv, isTied, sd) => {
  const rand = mulberry32(sd * 7919)
  const spread = sig * Math.sqrt(dd) // 其余 logit ~ N(0, σ²·d): |h| ≈ √d, 每个嵌入分量 std = σ
  let total = 0
  const trials = 24
  for (let t = 0; t < trials; t++) {
    const logits = range(vv).map(() => spread * randn(rand))
    if (isTied) logits[0] = sig * dd        // ★ h · e_self = √d · |e_self| ≈ √d · σ√d = σ·d
    const target = logits[1]                // 正确答案是"别的某个 token"
    const m = Math.max(...logits)
    total += m + Math.log(logits.reduce((s, z) => s + Math.exp(z - m), 0)) - target
  }
  return total / trials
}
const ce = computed(() => ceAt(sigma.value, d.value, V.value, tied.value, seed.value))
const ok = computed(() => Math.abs(ce.value - lnV.value) < 0.5) // 与 train 脚本里的断言同一阈值

// ── 画图: x = log10 σ, y = log CE ─────────────────────────────────
const W = 520, H = 260, PAD = 34
const xOf = (ls) => PAD + ((ls + 2.5) / 2.8) * (W - PAD - 12)
const sOf = (x) => ((x - PAD) / (W - PAD - 12)) * 2.8 - 2.5
const yOf = (c) => H - 24 - (Math.log(clamp(c, 3, 1500) / 3) / Math.log(500)) * (H - 48)
const curve = computed(() =>
  range(57).map((i) => {
    const ls = -2.5 + i * 0.05
    return `${xOf(ls)},${yOf(ceAt(10 ** ls, d.value, V.value, tied.value, seed.value))}`
  }).join(' '))
</script>

<style scoped>
.ref { stroke: var(--left); stroke-dasharray: 5 4; stroke-width: 1.5; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.knob { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.knob:focus-visible { outline: none; stroke: var(--warn); stroke-width: 3; }
.lbl { fill: var(--text-dim); font-size: 11px; font-family: "SF Mono", Menlo, monospace; }
</style>
