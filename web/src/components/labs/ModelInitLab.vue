<!--
  初始化实验台: 为什么"刚建好的模型"第一步 loss 应该 ≈ ln V, 以及绑权重 + std=1 怎么把它炸到 250。
  对应 llm_models/utils/init.py: std=0.02 时 LLaMA 首步 CE 7.073 (run_models/language_models/llama/README.md 实测), ln 1000 = 6.908。
-->
<template>
  <LabFrame
    title="初始化 — 刚建好的模型, 第一步 loss 该是多少?"
    sub="- 模拟什么: 训练第 1 步, 权重还是随机数, 模型应该什么都不懂, 对词表里 V 个 token 均匀瞎猜, 这时交叉熵 loss (CE) 恰好等于 $\ln V$。如果第一步 loss 远大于 $\ln V$, 说明模型一上来就在「极度自信地猜错」, 原因几乎都在初始化的尺度 $\sigma$ (随机权重的标准差)。这里模拟的是模型的最后一步: 最后一层 norm 之后的隐藏向量 h 和输出头 lm_head 的每一行做点积得到 logits, 再算 CE。还有一个常见坑: 把 lm_head 和输入 embedding 绑成同一张表 (weight tying, 省参数), 这时 h 几乎还贴着输入 token 自己的 embedding, 模型会给「输入 token 自己」打一个高达 $\sigma \cdot d$ 的分, 而正确答案几乎从来不是它自己。对应 llm_models/utils/init.py。
      - 图上是什么: 横轴是 $\sigma$ (对数刻度), 纵轴是第 1 步的 CE (也是对数刻度)。曲线是每个 $\sigma$ 下的首步 CE, 用蒙特卡洛估计 (随机抽 24 组 logits 取平均); 虚线是 $\ln V$, 均匀瞎猜的参考线; 圆点是当前 $\sigma$, 可以直接拖。
      - 可以做什么: 「绑权重」开关 lm_head 和 embedding 是否共用一张表。两个快捷按钮跳到 $\sigma$ = 0.02 (本仓库 init_weights 用的值) 和 $\sigma$ = 1 (PyTorch nn.Embedding 的默认值)。「换一组随机权重」重抽。拖「嵌入初始化 $\sigma$」、「模型宽度 d」、「词表大小 V」看曲线怎么变。
      - 应该看到: 默认 $\sigma$ = 1、d = 256、V = 1000、绑权重时首步 CE 约 250, 而 $\ln V$ = 6.91, 差了几十倍。拖到 $\sigma$ = 0.02 后 CE 约 7.1, 贴到参考线上。右侧「首步 CE」和 $\ln V$ 相差不到 0.5 时算正常 (和 train 脚本里的断言同一阈值)。「自己的 logit ≈ $\sigma \cdot d$」是绑权重时输入 token 自己拿到的分; 「其余 logit 的标准差 $\sigma \sqrt{d}$」是别的 token 分数的散布, 它远大于 1 时 softmax 已经很尖, 哪怕不绑权重 loss 也会远高于 $\ln V$。"
    module="llm_models/utils/init.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '不改 σ=1, 只把「绑权重」关掉, 首步 CE 会回到 ln V 吗? 为什么真实 LLaMA 不绑权重也不乘 √d 却没这个问题?',
      answer: '不会回到 ln V。\n- 关掉绑权重后, 「给自己打高分」那一项消失, 但其余 logit 的标准差仍然是 $\\sigma \\cdot \\sqrt{d}$ = 1 × 16 = 16。logit 之间动辄差十几分, softmax 依旧极尖, 模型还是在自信地押某个随机 token, CE 还是几十。\n- 根因是 logit 的尺度, 绑权重只是让它更糟。所以把 $\\sigma$ 降到 0.02 才治本: logit 标准差变成 0.02 × 16 ≈ 0.3, 所有 token 的分数都差不多, softmax 近似均匀, CE 回到 $\\ln V$。\n真实 LLaMA 用 $\\sigma$ = 0.02 初始化 (init_weights 的默认值), 两个问题都不存在, 所以它既不需要绑权重时的特别处理, 也不需要乘 √d 来救场。',
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
        <Tex :text="ok ? '✓ 接近 $\\ln V$: 模型从「均匀瞎猜」出发, 之后 loss 下降才说明它真的学到了东西。'
              : '✗ 远高于 $\\ln V$: 模型一上来就极度自信地猜错。这时「loss 在下降」只是在纠正初始化, 不代表学会了任务。'" />
      </p>
      <p class="lab-note"><Tex text="简化模型: 最后一层归一化之后隐藏向量的范数约为 $\sqrt{d}$, 而且方向仍贴着输入 token 的嵌入 (残差连接一路把它带了过来)。它复现了仓库里的实测值: $\sigma=1$, $d=256$, $V=1000$ 时约 250; $\sigma=0.02$ 时约 7.1。" /></p>
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
