<!--
  GAE 实验台: 与 llm_finetune/methods/ppo.py:gae 同一条从后往前的递推。
  只讲一件事: λ 在 TD(0) 和 Monte-Carlo 之间插值 —— λ=0 只信下一步的 V, λ=1 只信真实回报。
  奖励写法同 PPOTrainer: 每个 token 扣 β·KL, verifier 分只落在末 token; 末 token 之后是终止态 (V = 0)。
-->
<template>
  <LabFrame
    title="GAE: 从 TD(0) 滑到 Monte-Carlo"
    sub="一条 6 个 token 的回复。上图的点是 critic 给每个前缀的估值 $V(s_t)$, 可以上下拖。
      下图的柱是 GAE 算出的优势 $A_t$:
      - 圆点是 TD(0) 的 $\delta_t$, 菱形是 MC 回报减 $V(s_t)$。
      拖 λ, 看柱子从圆点滑到菱形。"
    module="llm_finetune/methods/ppo.py"
    run="python -m llm_finetune.run_finetune.ppo.train_ppo"
    :challenge="{
      ask: '先点「V ≡ 0」模拟冻结的 critic, 再把 λ 拉到 1、γ 拉到 1。6 个 token 的 $A_t$ 长什么样? 这和 GRPO 给整条回复一个 $A$ 有什么区别?',
      answer: '$V \\equiv 0$ 时 $\\delta_t = r_t$, λ=1、γ=1 下 $A_t = \\sum_{k\\ge 0} r_{t+k}$: 每个 token 都约等于最终的 verifier 分, 只差后面几步的小 KL 惩罚。\n这就是 critic 学不会时的 PPO: 所有 token 分到几乎一样的 $A$, 再做批内白化, 等价于 REINFORCE 减批均值。\n脚本里 critic 冻结 (lr=0) 的对照 pass@1 0.389, 和正常 PPO 的 0.381 在噪声内。\n逐 token 的信用分配要靠一个能估准值的 $V$。60 步、64 维的 critic 没学到: $V(s_0)$ 只解释 2.5% 的回报方差。',
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
      <div class="kv"><span><Tex text="$\max_t |A_t - \delta_t|$ (离 TD(0))" /></span><b :class="dTD < 1e-9 ? 'good' : ''">{{ dTD.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="$\max_t |G_t - \mathrm{MC}_t|$ (离 MC)" /></span><b :class="dMC < 1e-9 ? 'good' : ''">{{ dMC.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="首 token 优势 $A_0$" /></span><b>{{ cols[0].A.toFixed(3) }}</b></div>
      <div class="kv"><span>逐 token 的 A 差多少 (max − min)</span><b :class="spread < 0.05 ? 'bad' : 'good'">{{ spread.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p><Tex text="$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, $A_t = \delta_t + \gamma\lambda A_{t+1}$, 末 token 之后 $V = 0$。" /></p>
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
