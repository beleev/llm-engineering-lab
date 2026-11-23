<!--
  蒸馏的 T² 梯度补偿实验台 (对应 llm_finetune/methods/distill.py: DistillLoss)。
  只讲一件事: 软标签 KL 对 student logits 的梯度是 (1/T)(q^T − p^T), T 大时 q^T − p^T 本身又 ∝ 1/T,
  所以整体按 1/T² 衰减; 不乘 T² 的话, 调高温度等于偷偷把蒸馏项关掉, α 的含义也跟着 T 漂移。
  极限有闭式: T·(q^T − p^T) → [(z_s − z̄_s) − (z_t − z̄_t)] / V, 图上虚线就是它。
-->
<template>
  <LabFrame
    title="蒸馏为什么要乘 T² — 温度一高, 软标签的梯度就悄悄消失"
    sub="- 模拟什么: 知识蒸馏通过最小化分布散度令 student 拟合 teacher 的输出概率分布。为充分暴露 teacher 在次优候选 token 上的相对偏好结构 (即暗知识), 双方 logits 均先除以温度 $T$ 再执行 softmax 归一化, $T$ 越大输出分布越平滑。混合损失由权重为 $\alpha$ 的硬标签 CE 与权重为 $1-\alpha$ 的软标签 KL 散度相加构成。软标签项对 student logits 的梯度为 $(q^T - p^T)/T$ (其中 $p^T$、$q^T$ 分别为 teacher 与 student 经温度软化后的概率分布); 当 $T$ 增大时, 概率差 $q^T - p^T$ 本身按 $1/T$ 缩小, 导致软标签梯度整体按 $1/T^2$ 衰减。若不加尺度补偿, 升高温度会直接导致蒸馏梯度趋近于零。Hinton 2015 提出为软标签项乘以 $T^2$ 予以补偿, 本仓库 distill.py 中的 DistillLoss 同样采用该设计。
- 图上是什么: 界面由左右两幅联动图表组成:
  - 左侧 6 组成对柱状图: 展示 6 个候选 token 在当前温度 $T$ 下的软化概率, 紫色代表 teacher, 绿色代表 student, 标注「标签」的一组对应真实硬标签类别。
  - 右侧曲线图: 描绘软标签项梯度范数随温度 $T$ 变化的轨迹。橙色曲线表示未乘补偿因子, 绿色实线表示乘以 $T^2$ 补偿后的梯度范数, 绿色水平虚线标示 $T\to\infty$ 时的高温渐近极限。
  - 竖线与圆点: 指示当前选定温度 $T$ 在曲线上的对应位置。
- 可以做什么: 按界面顺序可执行以下交互:
  - 切换「未乘 T² / 已乘 T²」按钮: 控制是否启用 $T^2$ 梯度尺度补偿 (决定右侧汇总读数基于哪条曲线计算)。
  - 点击「换一组 logits」: 重新随机采样一组 teacher 与 student logits。
  - 拖动「温度 T」滑杆: 观察竖线扫过两条梯度范数曲线, 同时对比左图概率分布的平滑程度变化。
  - 拖动「α」滑杆: 调节硬标签 CE 损失的混合权重。
  - 拖动「student 离 teacher 多远」滑杆: 缩放 student 与 teacher logits 之间的初始偏差幅度。
  - 点击左图任意候选 token: 将该位置切换为硬标签目标类别。
- 应该看到: 未启用补偿时, 随温度 $T$ 升高橙色曲线迅速衰减逼近零; 启用 $T^2$ 补偿后绿色曲线平稳收敛至非零常数。右侧读数含义如下:
  - 「‖∇CE‖」: 硬标签交叉熵项对 student logits 的梯度范数。
  - 「‖∇KD‖ 不补偿」与「‖∇KD‖ × T²」: 软标签蒸馏项在补偿前后的梯度范数。
  - 「软标签占总梯度」: 按当前混合系数 $\alpha$ 与补偿开关计算的软标签梯度实际占比; 理想状态下应接近名义权重 $1-\alpha$, 偏离过大时显示为红色, 提示温度参数已扭曲混合权重 $\alpha$ 的实际比例。"
    module="llm_finetune/methods/distill.py"
    run="python -m llm_finetune.run_finetune.distill.train_distill"
    :challenge="{
      ask: '在默认 logits 配置下固定混合权重 $\\alpha = 0.5$, 按顺序执行以下操作:\n1. 保持「未乘 T²」状态, 将温度 T 从 1 拖至 8: 观察「软标签占总梯度」从多少降至多少?\n2. 切换至「乘 T²」补偿状态后重新将 T 从 1 拖至 8, 观察该占比的变化趋势。',
      answer: '结论:\n1. 未乘 $T^2$ 补偿时, 软标签梯度占比从 43.4% 骤降至 1.5%。\n2. 乘以 $T^2$ 补偿后, 该占比从 43.4% 平稳升至 50.1%, 并随温度继续升高稳定在 50% 附近。\n机制解析:\n- 未补偿情形: 软标签项梯度范数按 $1/T^2$ 阶衰减, 在 $T = 8$ 时仅余 $T = 1$ 时的约 2%。此时名义上设定 $\\alpha = 0.5$ 的等权混合, 实际参数更新几乎完全由硬标签 CE 主导, 蒸馏退化为常规 SFT, 高温软化失去实际作用。\n- 乘以 $T^2$ 补偿情形: 软标签梯度范数随 $T$ 增大收敛至非零常数 (对应右图水平虚线), 该渐近极限具有显式解析形式, 等于双方「去均值 logits 之差」除以词表大小 $V$。Hinton 2015 证明了高温极限下带 $T^2$ 补偿的 KL 蒸馏等价于去均值 logits 的均方误差匹配。\n- 超参解耦效应: 引入 $T^2$ 补偿后, 温度 $T$ 专职控制次优候选暗知识的平滑展开程度, 系数 $\\alpha$ 专职控制软硬监督信号的梯度配比, 两个超参数不再相互干扰。在本仓库 128 条固定样本实测中, 引入软标签可将留出集 forward KL 从纯硬标签的 2.293 降至 1.860, 其前提正是蒸馏项梯度未被高温衰减殆尽。',
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
        <Tex text="软标签梯度占比定义为 $(1-\alpha)\cdot\|\nabla\mathrm{KD}\| \,/\, \big(\alpha\cdot\|\nabla\mathrm{CE}\| + (1-\alpha)\cdot\|\nabla\mathrm{KD}\|\big)$, 度量软硬两项分别乘以混合权重后软标签梯度范数在总梯度中的相对比例。" /><Tex text="$\alpha$" /> = {{ alpha.toFixed(2) }} 名义上给软标签 {{ ((1 - alpha) * 100).toFixed(0) }}%;
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
