<!--
  KTO 实验台。公式与 llm_finetune/methods/kto.py:KTOLoss 相同 (β=0.5, z0 = 0 —— 本例 z0 全程被 clamp 成 0)。
  只讲一件事: 单条 👍 / 👎 也能训, 但好坏不均时, 两类样本的总推力 λ·n 必须拉平。
  曲线和推力是前端实时算的; 右侧准确率 / log π / EM 来自 python -m llm_finetune.run_finetune.kto.train_kto 的输出表。
-->
<template>
  <LabFrame
    title="KTO: 好坏 1:9 时, λ 为什么必须跟着调"
    sub="上面 10 个格子是一个 batch 的样本比例, 绿 = 👍, 红 = 👎。
      下图是单条样本对隐式奖励 $r = \log\pi/\pi_{\text{ref}}$ 的推力 $|\partial v/\partial r|$, 拖竖线改 $r$。
      条形是整批的总推力: 好样本往上推 $r$, 坏样本往下压。"
    module="llm_finetune/methods/kto.py"
    run="python -m llm_finetune.run_finetune.kto.train_kto"
    :challenge="{
      ask: '选 1:9、关掉「按比例调 λ」。坏样本的总推力是好样本的几倍? 模型会怎样应对? 再打开开关看 $\\lambda_U$ 变成多少。',
      answer: '$\\lambda_D = \\lambda_U = 1$ 时, 同一个 $r$ 上每条样本推力一样, 9 条坏样本的总推力就是 1 条好样本的 9 倍。\n模型最省力的办法是把所有回复一起往下压。实测 log π(chosen) 从 −4.03 掉到 −30.02, 准确率 0.492, EM 0.000。偏好方向丢了, 生成全崩。\n按 $\\lambda_D n_D \\approx \\lambda_U n_U$ 把 $\\lambda_U$ 调成 1/9 后, 两边推力拉平: 准确率 0.988, EM 0.254。\n这份 1:9 的数据 DPO 一对也凑不出, 因为每个 prompt 只出现一条回复。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: ratio === 1 }" @click="ratio = 1">好:坏 = 1:1</button>
        <button type="button" :class="{ active: ratio === 9 }" @click="ratio = 9">好:坏 = 1:9</button>
        <button type="button" :class="{ active: tune }" :aria-pressed="tune" @click="tune = !tune">按比例调 λ_U: {{ tune ? '开' : '关' }}</button>
      </div>
    </template>

    <div class="cells batch" style="grid-template-columns: repeat(10, 26px);">
      <span v-for="i in 10" :key="i" class="cell" :class="i <= nD ? 'ok' : 'bad'">{{ i <= nD ? '👍' : '👎' }}</span>
    </div>

    <svg ref="svgEl" viewBox="0 0 560 170" role="group" aria-label="单条样本的推力曲线">
      <line x1="30" y1="140" x2="550" y2="140" class="axis" />
      <line :x1="RX(0)" y1="20" :x2="RX(0)" y2="140" class="axis" />
      <text :x="RX(0) + 4" y="156" class="t">r = z0 = 0</text>
      <path :d="curve('d')" fill="none" stroke="var(--left)" stroke-width="2" />
      <path :d="curve('u')" fill="none" stroke="var(--danger)" stroke-width="2" />
      <text x="34" y="30" class="t" style="fill: var(--left)">好: λ_D·β·σ(1−σ)</text>
      <text x="34" y="46" class="t" style="fill: var(--danger)">坏: λ_U·β·σ(1−σ)  (λ_U = {{ lamU.toFixed(3) }})</text>
      <!-- ★ 拖竖线 = 改隐式奖励 r -->
      <line :x1="RX(r)" y1="16" :x2="RX(r)" y2="140" stroke="var(--accent)" stroke-width="2" />
      <rect
        class="draggable" :x="RX(r) - 8" y="8" width="16" height="136" fill="transparent"
        tabindex="0" role="slider" aria-label="隐式奖励 r" :aria-valuenow="r.toFixed(1)" aria-valuemin="-10" aria-valuemax="10"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setR((x - 290) / 26) })"
        @keydown.right.prevent="setR(r + 0.5)" @keydown.left.prevent="setR(r - 0.5)"
      />
      <text :x="RX(r) + 5" y="20" class="t">r = {{ r.toFixed(1) }}</text>
    </svg>

    <div class="push">
      <div v-for="b in pushes" :key="b.id" class="prow">
        <span class="pl">{{ b.name }}</span>
        <i class="pbar" :style="{ width: (b.v / pmax) * 100 + '%', background: b.color }" />
        <span class="mono pv">{{ b.v.toFixed(3) }}</span>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>坏 : 好 总推力</span><b :class="Math.abs(pushes[1].v / pushes[0].v - 1) < 0.05 ? 'good' : 'bad'">{{ (pushes[1].v / pushes[0].v).toFixed(2) }} : 1</b></div>
      <div class="kv"><span>留出集偏好准确率 (SFT 0.965)</span><b :class="row.acc > 0.965 ? 'good' : 'bad'">{{ row.acc.toFixed(3) }}</b></div>
      <div class="kv"><span>log π(chosen) (SFT −4.03)</span><b :class="row.lpc > -4.03 ? 'good' : row.lpc < -10 ? 'bad' : ''">{{ row.lpc.toFixed(2) }}</b></div>
      <div class="kv"><span>贪心 EM (SFT 0.332)</span><b :class="row.em < 0.1 ? 'bad' : ''">{{ row.em.toFixed(3) }}</b></div>
      <p class="lab-note">
        对照 DPO (成对数据): 准确率 0.996, log π(chosen) −4.25, EM 0.121。同样的前向预算, KTO 每步有标签的回复只有 64 条, DPO 是 128 条。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const BETA = 0.5, LAM_D = 1
// train_kto.py 输出表的三行 KTO (1:1 时调不调 λ_U 都是 1, 同一行)
const MEASURED = {
  '1-1': { acc: 0.992, lpc: -3.73, em: 0.324 },
  '9-1': { acc: 0.988, lpc: -4.12, em: 0.254 },
  '9-0': { acc: 0.492, lpc: -30.02, em: 0.0 },
}
const ratio = ref(9), tune = ref(false), r = ref(0)
const svgEl = ref(null)
const { start } = useDrag()
const setR = (v) => { r.value = clamp(Math.round(v * 2) / 2, -10, 10) }

const nD = computed(() => (ratio.value === 1 ? 5 : 1))
const nU = computed(() => 10 - nD.value)
const lamU = computed(() => (tune.value ? (LAM_D * nD.value) / nU.value : 1))   // ★ λ_D·n_D ≈ λ_U·n_U
const row = computed(() => MEASURED[ratio.value === 1 ? '1-1' : tune.value ? '9-1' : '9-0'])

const sig = (x) => 1 / (1 + Math.exp(-x))
// |∂v/∂r|: 好 v = λ_D σ(β(r − z0)), 坏 v = λ_U σ(β(z0 − r)); 两者导数的大小形状一样, 只差 λ
const grad = (lam, x) => { const s = sig(BETA * x); return lam * BETA * s * (1 - s) }
const RX = (x) => 290 + x * 26
const GY = (g) => 140 - g * 800
const curve = (k) => range(81).map((i) => {
  const x = -10 + i * 0.25
  return `${i ? 'L' : 'M'} ${RX(x).toFixed(1)} ${GY(grad(k === 'd' ? LAM_D : lamU.value, x)).toFixed(1)}`
}).join(' ')
const pushes = computed(() => [
  { id: 'd', name: `好样本 × ${nD.value}`, v: nD.value * grad(LAM_D, r.value), color: 'var(--left)' },
  { id: 'u', name: `坏样本 × ${nU.value}`, v: nU.value * grad(lamU.value, r.value), color: 'var(--danger)' },
])
const pmax = computed(() => Math.max(...pushes.value.map((b) => b.v), 1e-9))
</script>

<style scoped>
.batch { margin-bottom: 12px; }
.batch .cell { width: 26px; height: 26px; font-size: 13px; }
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.axis { stroke: var(--border-strong); stroke-dasharray: 3 3; }
.push { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.prow { display: grid; grid-template-columns: 90px 1fr 56px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); }
.pbar { display: block; height: 12px; border-radius: 2px; }
.pv { text-align: right; color: var(--text); }
</style>
