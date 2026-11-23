<!--
  Bradley–Terry 奖励模型实验台 (对应 llm_finetune/methods/reward_model.py: bradley_terry_loss)。
  只讲一件事: RM 只学"差", P(A≻B) = σ(r_A − r_B); 标注噪声 ε 决定了这个差最多学到多大 ——
  期望 loss 的最小点在 Δ* = ln((1−ε)/ε), 最小值 = 噪声的二元熵 H(ε)。闭式解与图上曲线的最低点一致。
-->
<template>
  <LabFrame
    title="Bradley–Terry — 奖励模型学到的只是两条回答的「分差」"
    sub="- 模拟什么: RLHF 流程的第一步是训练奖励模型 (RM, Reward Model), 将单条回复映射为标量分数 $r$。由于人工标注仅提供「A 优于 B」的成对偏序关系, RM 采用 Bradley–Terry 模型建模偏好概率 $P(A \succ B) = \sigma(r_A - r_B)$, 即预测胜率完全取决于相对分差 $\Delta = r_A - r_B$, 对两个分数同时加上任意常数不会改变胜率。人工标注不可避免存在翻转误差, 本实验台以 $\varepsilon$ 表示标签标反的概率, 展示标注噪声如何约束 RM 可收敛的最优分差。
- 图上是什么: 界面分为上下两幅联动视图:
  - 上方数轴: 展示 RM 对两条候选回复的打分位置。绿色点 A 代表标注偏好项, 红色点 B 代表被拒项, 两点间的粗线段表示分差 $\Delta$ (A 位于右侧时呈绿色, 排序颠倒时呈红色)。
  - 下方紫色曲线: 存在比例为 $\varepsilon$ 的标注翻转噪声时, 成对样本的期望损失随分差 $\Delta$ 变化的函数曲线。
  - 橙色圆点: 当前分差在期望损失曲线上的对应工作点。
  - 绿色虚线: 标示期望损失曲线的极小值位置 $\Delta^*$。
- 可以做什么: 可拖拽数轴圆点或操作按钮与滑杆:
  - 左右拖动点 A 或点 B (或聚焦后使用方向键) 修改各自打分。
  - 点击「两个一起 +2 / −2」将两点同步平移相同常数, 保持相对分差不变。
  - 点击「把 A 放到最优分差」直接将点 A 移至理论最优分差 $\Delta^*$ 处。
  - 拖动「标注噪声 ε」调节标签翻转概率, 取 0 代表无噪声理想标注, 取 0.5 代表完全随机标注。
- 应该看到: 当 $\varepsilon = 0$ 时损失曲线单调递减且无有限极小值, 点 A 右移越远损失越小; 当 $\varepsilon \gt 0$ 时损失曲线呈凸碗形, 唯一极小值位于 $\Delta^* = \ln((1-\varepsilon)/\varepsilon)$。右侧读数含义如下:
  - 「P(A ≻ B)」: RM 当前预测的回复 A 胜率。
  - 「期望 loss」: 当前分差对应的期望交叉熵损失, 逼近理论下限时变为绿色。
  - 「最优分差」: 凸碗形损失曲线谷底对应的分差 $\Delta^*$。
  - 「loss 下限」: 谷底对应的最小可能损失值, 严格等于标注噪声的二元熵 $H(\varepsilon)$, 代表含噪数据集上的理论可优化极限。"
    module="llm_finetune/methods/reward_model.py"
    run="python -m llm_finetune.run_finetune.rm.train_rm"
    :challenge="{
      ask: '请按顺序完成以下三步观察:\n1. 将标注噪声 ε 拖至 0, 持续向右拖动点 A: 期望 loss 是否存在有限最小值?\n2. 将 ε 调至 0.2, 寻找使期望 loss 达到最小的分差, 该最优分差是多少?\n3. 点击「两个一起 +2」: 图中哪些量发生了变化, 哪些保持不变?',
      answer: '结论:\n1. 当 $\\varepsilon = 0$ 时, 期望 loss 不存在有限最小值, 点 A 右移越远损失越低。\n2. 当 $\\varepsilon = 0.2$ 时, 期望 loss 极小值位于 $\\Delta^* = 1.386$。\n3. 点击「两个一起 +2」后, 坐标轴上的绝对打分整体加 2, 而预测胜率 $P$ 与期望 loss 完全不变。\n机制解析:\n- 无噪声情形 ($\\varepsilon = 0$): 损失函数 $\\text{loss} = -\\log\\sigma(\\Delta)$ 随 $\\Delta$ 单调递减, 理论最优分差趋于 $+\\infty$。在无噪声偏好数据上持续训练会驱使 RM 无限拉大分差, 因此 RM 输出的标量分数缺乏固定量纲, 仅具备相对排序意义而不可直接作为绝对收益使用。\n- 含标注噪声情形 ($\\varepsilon &gt; 0$): 期望损失写作 $\\text{loss} = -(1-\\varepsilon)\\log\\sigma(\\Delta) - \\varepsilon\\log\\sigma(-\\Delta)$, 按 $1-\\varepsilon$ 与 $\\varepsilon$ 的标签出现概率对正反方向交叉熵加权求和。当分差过大时, 占比为 $\\varepsilon$ 的反向标注样本会产生线性增长的惩罚, 使曲线呈凸碗形, 极小值点满足 $\\Delta^* = \\ln((1-\\varepsilon)/\\varepsilon)$: 在 $\\varepsilon = 0.2$ 时为 1.386, 在 $0.1$ 时为 2.197, 在 $0.5$ 时退化为 0 (纯随机标签下无法提取任何有效偏好信号)。\n- 极小值点的概率校准含义: 在谷底处有 $\\sigma(\\Delta^*) = 1-\\varepsilon$, 即 RM 预测胜率严格等于人工标注一致率, 说明充分收敛的 Bradley–Terry 奖励模型天然具备概率校准性质。谷底高度等于标签噪声的二元熵 $H(\\varepsilon)$, 构成含噪数据集的理论损失下界, 继续过度训练只会拟合样本级噪声。\n- 平移不变性: 同步加 2 不改变分差 $\\Delta$, 因此胜率 $P$ 与损失值均保持不变。Bradley–Terry 模型对打分整体平移不可辨识, 这也是下游 PPO 与 GRPO 需要对奖励信号引入 baseline 中心化或组内归一化的重要原因。本仓库 train_rm 实测显示, 留出集偏好准确率从 0.549 升至 0.930。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="shift(2)">两个一起 +2</button>
        <button type="button" @click="shift(-2)">两个一起 −2</button>
        <button type="button" :disabled="eps === 0" @click="rA = clamp(rB + dStar, -R, R)">把 A 放到最优分差</button>
      </div>
      <LabSlider v-model="eps" label="标注噪声 ε (标反的概率)" :min="0" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 330" role="group" aria-label="两条回答的奖励分数与期望 loss 曲线">
      <!-- 上: 奖励数轴 -->
      <line :x1="rx(-R)" :x2="rx(R)" y1="60" y2="60" stroke="var(--border-strong)" />
      <text v-for="t in [-6, -3, 0, 3, 6]" :key="'r' + t" :x="rx(t)" y="88" text-anchor="middle" class="t">{{ t }}</text>
      <text x="8" y="20" class="t">RM 分数 r</text>
      <line :x1="rx(rB)" :x2="rx(rA)" y1="60" y2="60" :stroke="delta >= 0 ? 'var(--left)' : 'var(--danger)'" stroke-width="4" />
      <text :x="(rx(rA) + rx(rB)) / 2" y="40" text-anchor="middle" class="t">Δ = {{ delta.toFixed(2) }}</text>
      <g v-for="d in dots" :key="d.id">
        <circle
          class="draggable" :cx="rx(d.v)" cy="60" r="12" :fill="d.color" stroke="var(--bg-card)" stroke-width="2"
          tabindex="0" role="slider" :aria-label="`回答 ${d.id} 的分数`" :aria-valuenow="d.v" :aria-valuemin="-R" :aria-valuemax="R"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => d.set((x - 320) / RS) })"
          @keydown.right.prevent="d.set(d.v + 0.25)" @keydown.left.prevent="d.set(d.v - 0.25)"
        />
        <text :x="rx(d.v)" y="64.5" text-anchor="middle" class="n">{{ d.id }}</text>
      </g>

      <!-- 下: 期望 loss vs Δ -->
      <text x="8" y="120" class="t" fill="var(--accent)">期望 loss(Δ) = −(1−ε)·log σ(Δ) − ε·log σ(−Δ)</text>
      <line :x1="dx(-DM)" :x2="dx(DM)" :y1="ly(0)" :y2="ly(0)" stroke="var(--border-strong)" />
      <text v-for="t in [-6, -3, 0, 3, 6]" :key="'d' + t" :x="dx(t)" y="322" text-anchor="middle" class="t">Δ={{ t }}</text>
      <path :d="lossPath" fill="none" stroke="var(--accent)" stroke-width="2" />
      <g v-if="eps > 0">
        <line :x1="dx(dStar)" :x2="dx(dStar)" :y1="ly(0)" :y2="ly(lossAt(dStar))" stroke="var(--left)" stroke-dasharray="3 3" />
        <circle :cx="dx(dStar)" :cy="ly(lossAt(dStar))" r="4" fill="var(--left)" />
        <text :x="dx(dStar)" :y="ly(lossAt(dStar)) - 8" text-anchor="middle" class="t" fill="var(--left)">Δ* = {{ dStar.toFixed(2) }}</text>
      </g>
      <circle :cx="dx(clamp(delta, -DM, DM))" :cy="ly(lossAt(delta))" r="7" fill="var(--eye)" stroke="var(--bg-card)" stroke-width="2" />
    </svg>

    <template #stats>
      <div class="kv"><span><Tex text="RM 预测的 A 胜率 $P(A \succ B) = \sigma(\Delta)$" /></span><b>{{ sigmoid(delta).toFixed(3) }}</b></div>
      <div class="kv"><span>当前分差下的期望 loss</span><b :class="gap < 0.01 ? 'good' : ''">{{ lossAt(delta).toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="最优分差 $\Delta^* = \ln((1-\varepsilon)/\varepsilon)$ (碗底位置)" /></span><b>{{ eps === 0 ? '+∞' : dStar.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="loss 下限 $= H(\varepsilon)$ (碗底高度)" /></span><b>{{ floor.toFixed(3) }}</b></div>
      <p class="lab-note">
        <Tex text="损失对分差的梯度为 $\partial\,\text{loss}/\partial\Delta = \sigma(\Delta) - (1-\varepsilon)$, 等于模型预测胜率与标注真实一致率之差; 当两者相等时梯度归零, 优化收敛于损失曲面谷底。" />
        噪声越大, 碗底越靠近 0, 学到的分差越扁, 下游 RL 拿到的奖励信号也就越弱。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const R = 7, RS = 42, DM = 8, LTOP = 3
const rx = (r) => 320 + r * RS
const dx = (d) => 320 + d * 36
const ly = (l) => 300 - Math.min(l, LTOP) / LTOP * 165

const rA = ref(1), rB = ref(-0.5), eps = ref(0.1)
const svgEl = ref(null)
const { start } = useDrag()

const snap = (v) => clamp(Math.round(v * 20) / 20, -R, R)
const dots = computed(() => [
  { id: 'B', v: rB.value, color: 'var(--danger)', set: (v) => (rB.value = snap(v)) },
  { id: 'A', v: rA.value, color: 'var(--left)', set: (v) => (rA.value = snap(v)) },
])
// 平移时两个点一起夹住, 保证分差不变
const shift = (s) => { const k = clamp(s, -R - Math.min(rA.value, rB.value), R - Math.max(rA.value, rB.value)); rA.value += k; rB.value += k }

const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))
const delta = computed(() => rA.value - rB.value)
// ★ 标签以 1−ε 的概率说 A 赢, 以 ε 的概率说 B 赢; −log σ(z) = softplus(−z)
const lossAt = (d) => (1 - eps.value) * softplus(-d) + eps.value * softplus(d)
const dStar = computed(() => Math.log((1 - eps.value) / Math.max(eps.value, 1e-12)))
const H = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p))
const floor = computed(() => H(eps.value))
const gap = computed(() => lossAt(delta.value) - floor.value)
const lossPath = computed(() => range(161).map((k) => { const d = -DM + k * 0.1; return `${k ? 'L' : 'M'}${dx(d).toFixed(1)},${ly(lossAt(d)).toFixed(1)}` }).join(' '))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.n { font-size: 11px; fill: var(--bg-card); font-weight: 700; pointer-events: none; }
</style>
