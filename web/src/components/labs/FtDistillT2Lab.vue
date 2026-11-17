<!--
  蒸馏的 T² 梯度补偿实验台 (对应 llm_finetune/methods/distill.py: DistillLoss)。
  只讲一件事: 软标签 KL 对 student logits 的梯度是 (1/T)(q^T − p^T), T 大时 q^T − p^T 本身又 ∝ 1/T,
  所以整体按 1/T² 衰减; 不乘 T² 的话, 调高温度等于偷偷把蒸馏项关掉, α 的含义也跟着 T 漂移。
  极限有闭式: T·(q^T − p^T) → [(z_s − z̄_s) − (z_t − z̄_t)] / V, 图上虚线就是它。
-->
<template>
  <LabFrame
    title="蒸馏为什么要乘 T² — 温度一高, 软标签的梯度就悄悄消失"
    sub="- 模拟什么: 知识蒸馏让 student 去学 teacher 的输出分布。为了让 teacher 分布里「第二、第三像的 token」也显出来 (暗知识), 两边的 logits 都先除以温度 $T$ 再做 softmax, $T$ 越大分布越平。总 loss 是 $\alpha$ 倍的硬标签 CE 加 $(1-\alpha)$ 倍的软标签 KL。软标签项对 student logits 的梯度是 $(q^T - p^T)/T$ ($p^T$、$q^T$ 是 teacher 和 student 软化后的分布), 而 $T$ 大时 $q^T - p^T$ 本身也按 $1/T$ 缩小, 整体按 $1/T^2$ 衰减。不补偿的话, 调高温度等于把蒸馏项关掉。Hinton 2015 的做法是给软标签项乘 $T^2$, 本仓库 distill.py 的 DistillLoss 也这样做。
      - 图上是什么: 左右两张图:
        - 左边 6 组成对的柱子: 6 个 token 在温度 $T$ 下的概率, 紫色是 teacher, 绿色是 student, 标着「标签」的那个是硬标签。
        - 右边的曲线: 软标签项的梯度范数随 $T$ 的变化。橙线不补偿, 绿线乘了 $T^2$, 绿色虚线是 $T\to\infty$ 时的极限值。
        - 竖线和圆点: 标出当前 $T$ 在哪。
      - 可以做什么: 按界面顺序:
        - 「未乘 T² / 已乘 T²」: 切换补偿开关 (决定右侧读数按哪条线算)。
        - 「换一组 logits」: 换随机数。
        - 拖「温度 T」: 看竖线扫过两条曲线, 同时左图的分布变平或变尖。
        - 拖「α」: 改硬标签 CE 的权重。
        - 拖「student 离 teacher 多远」: 改 student 的 logits 离 teacher 的差距。
        - 点左图任意 token: 把它设为硬标签。
      - 应该看到: 不补偿时, $T$ 一拖大橙线迅速贴地; 补偿后绿线趋于一个常数。右侧读数:
        - 「‖∇CE‖」: 硬标签项的梯度大小。
        - 「‖∇KD‖ 不补偿」和「‖∇KD‖ × T²」: 软标签项补偿前后的梯度大小。
        - 「软标签占总梯度」: 按当前 $\alpha$ 和开关算出的软标签项在总梯度里的实际占比。它应该接近名义上的 $1-\alpha$, 偏离太远 (红) 说明 $\alpha$ 的含义已经被 $T$ 带跑了。"
    module="llm_finetune/methods/distill.py"
    run="python -m llm_finetune.run_finetune.distill.train_distill"
    :challenge="{
      ask: '默认这组 logits, α 固定 0.5, 按顺序做:\n1. 保持「未乘 T²」, 把 T 从 1 拖到 8: 「软标签占总梯度」从多少掉到多少?\n2. 点「乘 T²」再拖一遍。',
      answer: '答案:\n1. 不补偿时占比从 43.4% 掉到 1.5%。\n2. 乘 $T^2$ 后从 43.4% 升到 50.1%, 之后停在 50% 上下。\n为什么:\n- 不补偿: 软标签梯度范数大约按 $1/T^2$ 掉, $T = 8$ 时只剩 $T = 1$ 时的 2% 左右。「$\\alpha = 0.5$」名义上五五开, 实际上几乎全是硬标签 CE 在训练, 等于在做普通 SFT, 温度白调了。\n- 乘 $T^2$: 软标签梯度随 $T$ 增大趋于一个常数 (图上的虚线)。这个极限有闭式: 两边「去均值 logits 之差」除以词表大小 $V$。高温极限下蒸馏退化成 logits 回归, Hinton 2015 的论文里推导过这一点。\n- 补偿之后 $T$ 只负责「暗知识展开多少」, $\\alpha$ 只负责「软硬各占多少」, 两个超参才互不干扰。本仓库 128 条固定数据上软标签明显比硬标签好 (forward KL 2.293 到 1.860), 前提就是这一项没被温度关掉。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: comp }" @click="comp = !comp">{{ comp ? '✓ 已乘 T²' : '✗ 未乘 T²' }}</button>
        <button type="button" @click="seed++">换一组 logits</button>
      </div>
      <LabSlider v-model="T" label="温度 T" :min="1" :max="10" :step="0.25" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="alpha" label="α (硬标签 CE 的权重)" :min="0" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="dist" label="student 离 teacher 多远" :min="0.2" :max="3" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <svg viewBox="0 0 640 300" role="group" aria-label="软化分布与梯度范数曲线">
      <!-- 左: 温度 T 下的两个分布; 点击选硬标签 -->
      <text x="10" y="16" class="t">softmax(z/T), T = {{ T.toFixed(2) }}</text>
      <g
        v-for="(tk, i) in toks" :key="i" class="tok" role="button" tabindex="0"
        :aria-label="`把 token ${i} 设为硬标签`" :aria-pressed="label === i"
        @click="label = i" @keydown.enter="label = i" @keydown.space.prevent="label = i"
      >
        <rect :x="14 + i * 40" y="24" width="36" height="236" :fill="label === i ? 'var(--accent-soft)' : 'transparent'" />
        <rect :x="17 + i * 40" :y="250 - tk.p * 210" width="14" :height="tk.p * 210" fill="var(--accent)" />
        <rect :x="33 + i * 40" :y="250 - tk.q * 210" width="14" :height="tk.q * 210" fill="var(--left)" />
        <text :x="32 + i * 40" y="274" text-anchor="middle" class="t">{{ label === i ? '标签' : 'tok' + i }}</text>
      </g>

      <!-- 右: ‖∇软标签‖ vs T -->
      <text :x="cx(1)" y="16" class="t">‖∂(软标签项)/∂z_student‖</text>
      <line :x1="cx(1)" :x2="cx(10)" :y1="cy(0)" :y2="cy(0)" stroke="var(--border-strong)" />
      <text v-for="t in [1, 2, 4, 6, 8, 10]" :key="t" :x="cx(t)" y="274" text-anchor="middle" class="t">{{ t }}</text>
      <text :x="cx(5.5)" y="292" text-anchor="middle" class="t">温度 T</text>
      <line :x1="cx(1)" :x2="cx(10)" :y1="cy(limit)" :y2="cy(limit)" stroke="var(--left)" stroke-dasharray="4 4" opacity="0.7" />
      <text :x="cx(10)" :y="cy(limit) + 14" text-anchor="end" class="t" fill="var(--left)">T→∞ 极限 {{ limit.toFixed(3) }}</text>
      <path :d="curve(false)" fill="none" stroke="var(--eye)" stroke-width="2" :opacity="comp ? 0.4 : 1" />
      <path :d="curve(true)" fill="none" stroke="var(--left)" stroke-width="2" :opacity="comp ? 1 : 0.4" />
      <line :x1="cx(T)" :x2="cx(T)" y1="24" :y2="cy(0)" stroke="var(--text-dim)" stroke-dasharray="2 3" />
      <circle :cx="cx(T)" :cy="cy(kdNorm(T, comp))" r="6" :fill="comp ? 'var(--left)' : 'var(--eye)'" stroke="var(--bg-card)" stroke-width="2" />
    </svg>

    <template #stats>
      <div class="kv"><span><Tex text="硬标签项的梯度大小 $\|\nabla\mathrm{CE}\|$" /></span><b>{{ ceNorm.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="软标签项的梯度大小 $\|\nabla\mathrm{KD}\|$, 不补偿" /></span><b>{{ kdNorm(T, false).toFixed(4) }}</b></div>
      <div class="kv"><span><Tex text="软标签项的梯度大小 $\|\nabla\mathrm{KD}\| \times T^2$, 补偿后" /></span><b>{{ kdNorm(T, true).toFixed(4) }}</b></div>
      <div class="kv"><span>软标签占总梯度 (按当前开关算)</span><b :class="share < 0.1 && alpha < 0.9 ? 'bad' : 'good'">{{ (share * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">
        <Tex text="占比 $= (1-\alpha)\cdot\|\nabla\mathrm{KD}\| \,/\, \big(\alpha\cdot\|\nabla\mathrm{CE}\| + (1-\alpha)\cdot\|\nabla\mathrm{KD}\|\big)$, 这个式子在说: 两项各乘自己的权重后, 软标签项占了总梯度的几成。" /><Tex text="$\alpha$" /> = {{ alpha.toFixed(2) }} 名义上给软标签 {{ ((1 - alpha) * 100).toFixed(0) }}%;
        实际占比偏离它多少, 就是 <Tex text="$T^2$" /> 补没补到位的度量。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, softmax, sum } from '@/utils/labmath.js'

const V = 6
const T = ref(2), alpha = ref(0.5), dist = ref(1.5), seed = ref(1), comp = ref(false), label = ref(0)
const cx = (t) => 310 + (t - 1) * 34
const cy = (g) => 250 - Math.min(g / yMax.value, 1.05) * 200

const base = computed(() => {
  const rand = mulberry32(seed.value * 613)
  const zt = range(V).map((i) => (i === 0 ? 4 : randn(rand) * 1.5))     // teacher: tok0 明显最优, 其余有相对排序
  return { zt, noise: range(V).map(() => randn(rand)) }
})
const zt = computed(() => base.value.zt)
const zs = computed(() => zt.value.map((z, i) => z * 0.4 + dist.value * base.value.noise[i]))   // 学生: 还没学好的版本
const norm = (g) => Math.hypot(...g)

// ★ ∂/∂z_s KL(p^T ‖ q^T) = (1/T)·(q^T − p^T); 补偿后再乘 T²
const kdGrad = (t, c) => { const p = softmax(zt.value, t), q = softmax(zs.value, t); return q.map((v, i) => (v - p[i]) / t * (c ? t * t : 1)) }
const kdNorm = (t, c) => norm(kdGrad(t, c))
const ceNorm = computed(() => norm(softmax(zs.value).map((q, i) => q - (i === label.value ? 1 : 0))))   // ∂CE/∂z = q − onehot
// 闭式极限: softmax(z/T) ≈ 1/V + (z − z̄)/(V·T)
const limit = computed(() => {
  const c = (z) => { const m = sum(z) / V; return z.map((x) => x - m) }
  const a = c(zs.value), b = c(zt.value)
  return norm(a.map((x, i) => (x - b[i]) / V))
})
const yMax = computed(() => Math.max(kdNorm(1, true), limit.value, ...range(10).map((k) => kdNorm(k + 1, true))) * 1.05)
const curve = (c) => range(91).map((k) => { const t = 1 + k * 0.1; return `${k ? 'L' : 'M'}${cx(t).toFixed(1)},${cy(kdNorm(t, c)).toFixed(1)}` }).join(' ')

const toks = computed(() => { const p = softmax(zt.value, T.value), q = softmax(zs.value, T.value); return range(V).map((i) => ({ p: p[i], q: q[i] })) })
const share = computed(() => {
  const kd = (1 - alpha.value) * kdNorm(T.value, comp.value), ce = alpha.value * ceNorm.value
  return kd + ce === 0 ? 0 : kd / (kd + ce)
})
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.tok { cursor: pointer; }
.tok:focus-visible { outline: 2px solid var(--accent); }
</style>
