<!--
  GAE 实验台: 与 llm_finetune/methods/ppo.py:gae 同一条从后往前的递推。
  只讲一件事: λ 在 TD(0) 和 Monte-Carlo 之间插值 —— λ=0 只信下一步的 V, λ=1 只信真实回报。
  奖励写法同 PPOTrainer: 每个 token 扣 β·KL, verifier 分只落在末 token; 末 token 之后是终止态 (V = 0)。
-->
<template>
  <LabFrame
    title="GAE — 优势估计从 TD(0) 滑到 Monte-Carlo, λ 就是那根滑杆"
    sub="- 模拟什么: PPO 要给回复里的每个 token 一个「优势」$A_t$: 这个 token 比平均水平好多少, 正就推高它的概率, 负就压低。算优势要用到 critic 的估值 $V(s_t)$ (critic 是另外训的一个网络, 估「从这个前缀往后还能拿多少分」)。GAE (广义优势估计) 用一个参数 $\lambda$ 在两种算法之间插值: $\lambda = 0$ 是 TD(0), 只信下一步的 $V$; $\lambda = 1$ 是 Monte-Carlo, 只信真实回报, 完全不信 $V$。奖励的写法和 PPOTrainer 一致: 每个 token 扣 $\beta$ 倍的 KL 惩罚 (policy 在这个 token 上比 ref 偏离了多少), verifier 的 0/1 分只落在最后一个 token 上。
      - 图上是什么: 一条 6 个 token 的回复, 横着排。上半部分的紫色圆点是 critic 给每个前缀的估值 $V(s_t)$; 下半部分的柱子是 GAE 算出的优势 $A_t$ (绿正红负), 每根柱子上叠着两个标记: 空心圆是 TD(0) 的结果 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, 黄色菱形是 MC 的结果 (从 t 往后的真实回报总和减 $V(s_t)$)。底部每一列写着这个 token 的即时奖励 $r_t$。
      - 可以做什么: 拖「GAE λ」看柱顶在空心圆和菱形之间滑动; 拖「折扣 γ」改后面的奖励打几折; 拖「KL 系数 β」改每个 token 的 KL 惩罚大小; 「verifier 全对 / 错」把最后一个 token 的分设成 1 或 0; 「V ≡ 0」把 critic 的估值全清零, 模拟一个什么都没学会的 critic; 「换一组 V」换随机估值。上下拖任意紫点 (或聚焦后按方向键) 直接改某个位置的 $V$。
      - 应该看到: $\lambda = 0$ 时柱顶和空心圆重合, $\lambda = 1$ 时和菱形重合。右侧读数: 前两行是柱子离 TD(0) 和离 MC 各有多远, 为 0 时变绿; 「首 token 优势 A_0」是第一个 token 分到的优势; 「逐 token 的 A 差多少」是 6 根柱子的最大值减最小值, 太小 (红) 说明所有 token 分到几乎一样的优势, critic 没在做逐 token 的信用分配 (分辨哪个 token 该为结果负责)。"
    module="llm_finetune/methods/ppo.py"
    run="python -m llm_finetune.run_finetune.ppo.train_ppo"
    :challenge="{
      ask: '先点「V ≡ 0」模拟冻结的 critic, 再把 λ 拉到 1、γ 拉到 1。6 个 token 的 $A_t$ 长什么样? 这和 GRPO 给整条回复一个 $A$ 有什么区别?',
      answer: '答案: 6 个 $A_t$ 几乎一样大, 都约等于最终的 verifier 分; 这时 PPO 和 GRPO 一样是整条回复一个优势, 区别只在 baseline 用批均值还是组均值。\n一步步看:\n- $V \\equiv 0$ 时 $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t) = r_t$, 只剩即时奖励。\n- $\\lambda = 1$、$\\gamma = 1$ 时 $A_t = \\sum_{k\\ge 0} r_{t+k}$, 这个式子在说: 每个 token 的优势就是它之后所有奖励的总和。verifier 的分只在最后一个 token 上, 所以每个位置都约等于那个分, 只差后面几步的小 KL 惩罚。\n- 这就是 critic 学不会时的 PPO: 所有 token 分到几乎一样的 $A$, 再做批内白化 (减批均值、除批标准差), 等价于 REINFORCE 减批均值。脚本里 critic 冻结 (lr=0) 的对照 pass@1 0.389, 和正常 PPO 的 0.381 在噪声内。\n- 逐 token 的信用分配要靠一个能估准的 $V$。本仓库 60 步、64 维的 critic 没学到: $V(s_0)$ 只解释 2.5% 的回报方差。',
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
        <p><Tex text="$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, 这个式子在说: 走一步拿到的奖励加上下一步的估值, 比原来估的多了多少, 就是这一步的「惊喜」。" /></p>
        <p><Tex text="$A_t = \delta_t + \gamma\lambda A_{t+1}$, 从最后一个 token 往前递推, 末 token 之后 $V = 0$。$\lambda$ 决定后面几步的惊喜要带回来多少: 0 只看这一步, 1 全带回来。" /></p>
        <p>脚本的单测: 随机 r / V、γ=0.9, λ=0 与 TD(0) 误差 0.0e+00, λ=1 与 MC 回报误差 4.8e-07。训练用 γ=1、λ=0.95、β=0.02。</p>
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
