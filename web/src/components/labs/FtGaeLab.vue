<!--
  GAE 实验台: 与 llm_finetune/methods/ppo.py:gae 同一条从后往前的递推。
  只讲一件事: λ 在 TD(0) 和 Monte-Carlo 之间插值 —— λ=0 只信下一步的 V, λ=1 只信真实回报。
  奖励写法同 PPOTrainer: 每个 token 扣 β·KL, verifier 分只落在末 token; 末 token 之后是终止态 (V = 0)。
-->
<template>
  <LabFrame
    title="GAE — 优势估计从 TD(0) 滑到 Monte-Carlo, λ 就是那根滑杆"
    sub="- 模拟什么: PPO 需要为生成回复中的每个位置计算逐 token 优势 $A_t$, 度量当前动作相对状态平均期望的优劣 (为正推高概率, 为负压低概率)。优势计算依赖独立训练的价值网络 critic 对各状态前缀给出的期望回报估值 $V(s_t)$。即时奖励构造与本仓库 PPOTrainer 完全一致: 每个位置扣除系数为 $\beta$ 的逐 token KL 偏离惩罚, 而验证器 verifier 给出的 0/1 终局奖励仅施加于末尾 token。GAE (Generalized Advantage Estimation, 广义优势估计) 通过衰减参数 $\lambda$ 在两类经典估计量之间平滑插值:
  - $\lambda = 0$ 退化为单步时序差分 TD(0), 完全依赖下一步价值估值 $V(s_{t+1})$ 自举。
  - $\lambda = 1$ 退化为 Monte-Carlo 估计, 直接累加后续真实奖励总和而不再用价值网络截断自举 (仅减去当前 $V(s_t)$ 作为方差基线)。
- 图上是什么: 横轴按生成顺序排列一条长度为 6 个 token 的回复序列, 各视觉元素对应如下:
  - 上半区紫色圆点: critic 对各前缀状态给出的价值估值 $V(s_t)$。
  - 下半区柱状图: 当前 GAE 计算得到的逐位置优势 $A_t$ (正值呈绿色, 负值呈红色), 每根柱体上方叠加空心圆与黄色菱形两类参考标记。
  - 空心圆标记: 单步 TD(0) 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
  - 黄色菱形标记: Monte-Carlo 优势估计值 (从步 $t$ 起累加的折扣真实回报减去 $V(s_t)$)。
  - 底部数值行: 逐列列出各位置实际获得的即时奖励 $r_t$。
- 可以做什么: 按界面顺序可执行以下交互:
  - 拖动「GAE λ」滑杆: 观察优势柱顶在空心圆与黄色菱形之间连续插值移动。
  - 拖动「折扣 γ」滑杆: 调整远期奖励的时间折扣因子。
  - 拖动「KL 系数 β」滑杆: 调节逐 token 施加的 KL 偏离惩罚强度。
  - 点击「verifier 全对 / 错」按钮: 将末尾 token 的终局验证奖励切换为 1 或 0。
  - 点击「V ≡ 0」按钮: 将 critic 在所有位置的估值全部置零, 模拟价值网络完全未学到有效特征的退化情形。
  - 点击「换一组 V」按钮: 重新随机生成一组前缀价值估值曲线。
  - 上下拖动任意紫色圆点 (或聚焦后使用方向键): 单独修改指定位置的价值估值 $V(s_t)$。
- 应该看到: 当 $\lambda = 0$ 时全部柱顶与空心圆精确重合, 当 $\lambda = 1$ 时与黄色菱形精确重合。右侧汇总读数含义如下:
  - 前两行: 当前 GAE 序列相对 TD(0) 与相对 MC 估计的最大偏差, 归零时显示为绿色。
  - 「首 token 优势 A_0」: 序列首个生成位置分配到的优势值。
  - 「逐 token 的 A 差多少」: 6 个位置优势值的极差 (最大值减最小值); 极差过小时显示为红色, 表明整条序列各位置分配到几乎相同的优势标量, critic 未能提供有效的逐 token 信用分配 (区分具体哪一步生成导致了最终成败)。"
    module="llm_finetune/methods/ppo.py"
    run="python -m llm_finetune.run_finetune.ppo.train_ppo"
    :challenge="{
      ask: '请按顺序执行以下设置并观察:\n1. 点击「V ≡ 0」模拟冻结且恒输出零的 critic。\n2. 将 GAE 参数 λ 拉至 1、折扣因子 γ 拉至 1。\n此时 6 个位置的优势 $A_t$ 呈现怎样的分布形态? 这一退化结果与 GRPO 为整条回复分配单一序列级优势 $A$ 有何异同?',
      answer: '结论:\n1. 6 个位置的优势 $A_t$ 数值几乎相等, 均近似等于末尾 token 的终局 verifier 得分。\n2. 此时 PPO 与 GRPO 同样退化为整条回复共享近似恒定的序列级优势, 二者的实质区别仅在于方差基线采用全局批均值还是同题组内均值。\n机制解析:\n- 当 $V \\equiv 0$ 时, 单步 TD 误差简化为 $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t) = r_t$, 仅保留各步即时奖励。\n- 在 $\\lambda = 1$ 且 $\\gamma = 1$ 下, GAE 递推式展开为 $A_t = \\sum_{k\\ge 0} r_{t+k}$, 即每个位置的优势等于从该步直至序列末尾的全部奖励累加和。由于可验证 0/1 奖励仅落在最末一个 token 上, 前序各位置累加值均包含同一终局得分, 彼此仅相差若干步微小的逐 token KL 惩罚。\n- 当价值网络未能拟合出准确的前缀估值时, PPO 便会退化至该状态: 同一序列内所有 token 获得近似相等的优势值, 经后续批内白化 (减全批均值并除以批标准差) 后, 等价于带批均值基线的序列级 REINFORCE。本仓库实测中, 将 critic 冻结 (学习率设为 0) 的对照实验留出集 pass@1 为 0.389, 与正常联合训练 critic 的 PPO (0.381) 处于统计波动范围内。\n- 细粒度逐 token 信用分配高度依赖价值网络 $V$ 的拟合精度; 在本仓库 60 步训练、64 维隐藏层的玩具规模下, critic 未能充分收敛, $V(s_0)$ 仅能解释 2.5% 的真实回报方差。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lam" label="GAE λ" :min="0" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="gamma" label="折扣 γ" :min="0.5" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="beta" label="KL 系数 β" :min="0" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
      <div class="row">
        <button type="button" :class="{ active: R === 1 }" @click="R = 1">verifier: 全对 (R=1)</button>
        <button type="button" :class="{ active: R === 0 }" @click="R = 0">verifier: 错 (R=0)</button>
        <button type="button" @click="V = V.map(() => 0)">V ≡ 0 (critic 冻结)</button>
        <button type="button" @click="reseed">换一组 V</button>
      </div>
    </template>

    <svg ref="svgEl" viewBox="0 0 640 350" role="group" aria-label="每个 token 的估值 V 与优势 A">
      <text x="8" y="16" class="t">V(s_t)  拖动</text>
      <line x1="60" :y1="VY(0)" x2="630" :y2="VY(0)" class="axis" />
      <text x="8" y="205" class="t">A_t</text>
      <line x1="60" :y1="AY(0)" x2="630" :y2="AY(0)" class="axis" />
      <g v-for="(c, t) in cols" :key="t">
        <text :x="X(t)" y="344" text-anchor="middle" class="t">o{{ t + 1 }} · r={{ c.r.toFixed(2) }}</text>
        <!-- ★ 拖动 = 改 critic 对这个前缀的估值 -->
        <circle
          class="draggable" :cx="X(t)" :cy="VY(V[t])" r="9" fill="var(--accent)"
          tabindex="0" role="slider" :aria-label="`V(s_${t})`" :aria-valuenow="V[t].toFixed(2)" aria-valuemin="-1.5" aria-valuemax="1.5"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ y }) => setV(t, (VY(0) - y) / SV) })"
          @keydown.up.prevent="setV(t, V[t] + 0.1)" @keydown.down.prevent="setV(t, V[t] - 0.1)"
        />
        <text :x="X(t) + 13" :y="VY(V[t]) + 4" class="v">{{ V[t].toFixed(2) }}</text>
        <rect :x="X(t) - 18" :width="36" :y="Math.min(AY(0), AY(c.A))" :height="Math.abs(AY(c.A) - AY(0))" rx="2"
          :fill="c.A >= 0 ? 'var(--left)' : 'var(--danger)'" opacity="0.75" />
        <circle :cx="X(t)" :cy="AY(c.td)" r="4.5" fill="none" stroke="var(--text)" stroke-width="1.5" />
        <path :d="`M ${X(t)} ${AY(c.mc) - 6} l 6 6 l -6 6 l -6 -6 z`" fill="none" stroke="var(--warn)" stroke-width="1.5" />
        <text :x="X(t)" y="328" text-anchor="middle" class="v">A={{ c.A.toFixed(2) }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span><Tex text="离 TD(0) 多远: $\max_t |A_t - \delta_t|$" /></span><b :class="dTD < 1e-9 ? 'good' : ''">{{ dTD.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="离 MC 多远: $\max_t |G_t - \mathrm{MC}_t|$" /></span><b :class="dMC < 1e-9 ? 'good' : ''">{{ dMC.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="首 token 优势 $A_0$" /></span><b>{{ cols[0].A.toFixed(3) }}</b></div>
      <div class="kv"><span>逐 token 的 A 差多少 (max − min)</span><b :class="spread < 0.05 ? 'bad' : 'good'">{{ spread.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p><Tex text="单步时序差分误差定义为 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, 即当前即时奖励加上折现后下一状态估值与当前状态原估值 $V(s_t)$ 之差; $\delta_t$ 为正表明当前生成的 token 使期望回报超出了 critic 原先的预期。" /></p>
        <p><Tex text="GAE 满足反向递推关系 $A_t = \delta_t + \gamma\lambda A_{t+1}$, 自末尾 token 向前逐位置回传 (终止状态后定义 $V = 0$); 系数 $\lambda$ 控制后续步骤的 TD 误差向前回传的衰减比例, 取 0 时仅保留当前单步误差, 取 1 时无衰减回传全部后续误差。" /></p>
        <p>脚本单元测试验证条件 (随机采样 r 与 V, 折扣因子 γ 取 0.9):</p>
        <ul class="pts">
          <li>λ=0 与 TD(0) 误差 0.0e+00。</li>
          <li>λ=1 与 MC 回报误差 4.8e-07。</li>
        </ul>
        <p>实际训练超参数配置取 γ 为 1、λ 为 0.95、β 为 0.02。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, range } from '@/utils/labmath.js'

const C = 6, SV = 40, SA = 40
const KL = [0.4, 0.1, 0.9, 0.3, 0.6, 0.2]            // 每个 token 的 log π_old − log π_ref (示意值)
const X = (t) => 100 + t * 94
const VY = (v) => 100 - v * SV                       // V 图: 0 在 y=100
const AY = (a) => 255 - clamp(a, -1.9, 1.9) * SA     // A 图: 0 在 y=255, 超出的截断显示

const lam = ref(0.95), gamma = ref(1), beta = ref(0.02), R = ref(1), seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()
const draw = (s) => { const rand = mulberry32(s * 977); return range(C).map((t) => +(0.2 + 0.1 * t + (rand() - 0.5) * 0.6).toFixed(2)) }
const V = ref(draw(1))
const reseed = () => { seed.value++; V.value = draw(seed.value) }
const setV = (t, v) => { V.value = V.value.map((x, k) => (k === t ? clamp(Math.round(v * 20) / 20, -1.5, 1.5) : x)) }

const cols = computed(() => {
  const r = KL.map((k, t) => -beta.value * k + (t === C - 1 ? R.value : 0))   // ★ 分数只落在末 token
  const g = gamma.value, out = Array(C)
  let run = 0, nextV = 0, mc = 0                     // 末 token 之后是终止态: V = 0
  for (let t = C - 1; t >= 0; t--) {
    const td = r[t] + g * nextV - V.value[t]
    run = td + g * lam.value * run                   // ★ GAE 递推, 与 ppo.py:gae 同一行
    mc = r[t] + g * mc
    out[t] = { r: r[t], td, A: run, G: run + V.value[t], mc: mc - V.value[t], MC: mc }
    nextV = V.value[t]
  }
  return out
})
const dTD = computed(() => Math.max(...cols.value.map((c) => Math.abs(c.A - c.td))))
const dMC = computed(() => Math.max(...cols.value.map((c) => Math.abs(c.G - c.MC))))
const spread = computed(() => { const a = cols.value.map((c) => c.A); return Math.max(...a) - Math.min(...a) })
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.v { font-size: 11px; fill: var(--text); font-family: "SF Mono", Menlo, monospace; }
.axis { stroke: var(--border-strong); stroke-dasharray: 3 3; }
</style>
