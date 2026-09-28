<!--
  Beam search 实验台 (对应 llm_infer/m24_beam_search/beam.py:beam_search)。
  一件事: beam 找的是"最可能"的序列, 而最可能的往往更重复、更短 —— 长度惩罚只改 finished 候选之间怎么比。
-->
<template>
  <LabFrame
    title="Beam search — 更可能, 不等于更好"
    sub="- 上图: TinyLM 上 10 个 prompt × 20 token, beam 宽度 $w$ 对序列 log 概率和重复率的影响 (Python 实测)。点柱子或拖滑杆改 $w$。
      - 下图: 分数是负数累加, 多一个 token 只会更低。拖 $\alpha$ 的把手, 按 $\text{score}/\text{len}^\alpha$ 比较 1 个 token 就收尾的候选和 15 个 token 的候选, 看谁赢。"
    module="llm_infer/m24"
    run="python -m llm_infer.m24_beam_search.demo"
    :challenge="{
      ask: '把 α 放在 0.5, 短候选还赢吗? 那 α 至少要多大, 长候选才能翻盘?',
      answer: '$\\alpha=0.5$ 时短候选仍赢: $-3/1 = -3$ 比 $-35/\\sqrt{15} \\approx -9.0$ 高。\n翻盘点: $3 = 35/15^\\alpha$, 即 $\\alpha^* = \\ln(35/3)/\\ln 15 \\approx 0.91$。\n这和 demo 的实测一致: EOS logit +3 后, $\\alpha=0$ 和 $0.5$ 的平均长度都是 1.3 token, $\\alpha=1$ 才到 15.7 (greedy 11.4)。\nHF 的 length_penalty 是除以 $\\text{len}^\\alpha$, $\\alpha \\gt 0$ 反而鼓励长序列。它只影响已结束的候选之间怎么比。',
    }"
  >
    <template #controls>
      <LabSlider v-model="wi" label="beam 宽度 w" :min="0" :max="5" :format="(i) => W[i]" />
      <LabSlider v-model="alpha" label="长度惩罚 α" :min="0" :max="1.5" :step="0.05" :format="(v) => v.toFixed(2)" />
      <div class="row"><button v-for="a in [0, 0.5, 1]" :key="a" type="button" :class="{ active: alpha === a }" @click="alpha = a">α = {{ a }}</button></div>
    </template>

    <svg ref="svg" viewBox="0 0 560 330" role="group" aria-label="beam 宽度与 log 概率和重复率; 长度惩罚下两条候选的分数">
      <text x="4" y="12" class="cap">① 柱 = 重复率 rep-2 (左轴), 线 = logP/token (右轴)</text>
      <g v-for="(w, i) in W" :key="w" class="grp" tabindex="0" role="button" :aria-label="`w=${w}`" @click="wi = i" @keydown.enter="wi = i">
        <rect :x="bx(i) - 30" y="18" width="60" height="140" class="hit" :class="{ sel: i === wi }" />
        <rect :x="bx(i) - 16" :y="ry(REP[i])" width="32" :height="158 - ry(REP[i])" class="rep" />
        <text :x="bx(i)" :y="ry(REP[i]) + 12" class="tick in">{{ REP[i].toFixed(2) }}</text>
        <text :x="bx(i)" y="170" class="tick">w={{ w }}{{ LOSE[i] ? ` · 输 ${LOSE[i]}` : '' }}</text>
      </g>
      <polyline :points="LOGP.map((v, i) => `${bx(i)},${ly(v)}`).join(' ')" class="lp" />
      <circle v-for="(v, i) in LOGP" :key="`c${i}`" :cx="bx(i)" :cy="ly(v)" :r="i === wi ? 4.5 : 3" class="lpd" />
      <text x="552" :y="ly(-2.2) + 3" class="tick end">−2.2</text>
      <text x="552" :y="ly(-2.7) + 3" class="tick end">−2.7</text>

      <text x="4" y="196" class="cap">② 归一化分数 score / len^α (越高越优先)</text>
      <line x1="40" x2="540" :y1="sy(0)" :y2="sy(0)" class="grid" />
      <polyline :points="curve(SHORT)" class="cand short" />
      <polyline :points="curve(LONG)" class="cand long" />
      <line :x1="ax(aStar)" :x2="ax(aStar)" y1="204" y2="306" class="star" />
      <text :x="ax(aStar) + 4" y="212" class="tick start">α* ≈ {{ aStar.toFixed(2) }}</text>
      <text v-for="a in [0, 0.5, 1, 1.5]" :key="a" :x="ax(a)" y="320" class="tick">{{ a }}</text>
      <text x="44" :y="sy(score(SHORT, 0)) - 4" class="tick start short-t">1 token 就收尾 (logP −3)</text>
      <text x="44" :y="sy(score(LONG, 0)) + 12" class="tick start long-t">15 token (logP −35)</text>
      <line :x1="ax(alpha)" :x2="ax(alpha)" y1="204" y2="306" class="cursor" />
      <rect :x="ax(alpha) - 8" y="204" width="16" height="102" class="grab draggable" tabindex="0" role="slider" aria-label="拖动改变 α"
        :aria-valuenow="alpha" @pointerdown="start($event, { svg, onMove })"
        @keydown.right.prevent="alpha = step(0.05)" @keydown.left.prevent="alpha = step(-0.05)" />
    </svg>

    <template #stats>
      <div class="kv"><span>w = {{ W[wi] }}: logP/token</span><b :class="wi > 0 ? 'good' : ''">{{ LOGP[wi].toFixed(3) }}</b></div>
      <div class="kv"><span>重复率 rep-2 (greedy 0.453)</span><b :class="REP[wi] > 0.5 ? 'bad' : ''">{{ REP[wi].toFixed(3) }}</b></div>
      <div class="kv"><span>输给 greedy 的 prompt (共 10)</span><b :class="LOSE[wi] ? 'bad' : 'good'">{{ LOSE[wi] }}</b></div>
      <div class="kv"><span>采样 T=1: logP/token / rep-2</span><b>−4.303 / 0.003</b></div>
      <div class="kv"><span>② α = {{ alpha.toFixed(2) }} 时胜出</span><b :class="win === 'long' ? 'good' : 'bad'">{{ win === 'long' ? '15 token 的候选' : '1 token 就收尾' }}</b></div>
      <div class="kv"><span>实测平均长度 α = 0 / 0.5 / 1</span><b>1.3 / 1.3 / 15.7</b></div>
      <div class="lab-note">
        <p>① 与实测长度来自 demo 的 [1] [3] 段。② 的两条候选取 README 自测题里的量级 (−3 与 −35), 分数按公式现算。</p>
        <p>多样性 (demo [2]): 同一 prompt 的 8 条 beam 候选平均只差 3.5/20 个位置, 8 条采样差 19.7。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

// 数据: python -m llm_infer.m24_beam_search.demo 的 [1] 段 (w=1 与 greedy 逐 token 相同)
const W = [1, 2, 4, 8, 16, 32]
const LOGP = [-2.62, -2.445, -2.391, -2.332, -2.298, -2.255]
const REP = [0.453, 0.5, 0.447, 0.5, 0.463, 0.563]
const LOSE = [0, 2, 1, 1, 0, 0]
const SHORT = { lp: -3, len: 1 }, LONG = { lp: -35, len: 15 }

const wi = ref(3), alpha = ref(0)
const svg = ref(null)
const { start } = useDrag()

const bx = (i) => 60 + i * 88
const ry = (v) => 158 - v * 220
const ly = (v) => 30 + ((-2.2 - v) / 0.5) * 120
// ★ HF 式长度惩罚: 只在比较已结束的候选时除以 len^α
const score = (c, a) => c.lp / c.len ** a
const AS = Array.from({ length: 31 }, (_, k) => k * 0.05)
const ax = (a) => 40 + (a / 1.5) * 500
const sy = (s) => 206 + (-s / 36) * 100
const curve = (c) => AS.map((a) => `${ax(a)},${sy(score(c, a))}`).join(' ')
const aStar = Math.log(LONG.lp / SHORT.lp) / Math.log(LONG.len / SHORT.len)
const win = computed(() => (score(LONG, alpha.value) > score(SHORT, alpha.value) ? 'long' : 'short'))
const step = (d) => Math.round(clamp(alpha.value + d, 0, 1.5) * 20) / 20
const onMove = ({ x }) => { alpha.value = Math.round(clamp(((x - 40) / 500) * 1.5, 0, 1.5) * 20) / 20 }
</script>

<style scoped>
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 10px; fill: var(--text-muted); }
.grid { stroke: var(--border-strong); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.tick.end { text-anchor: end; }
.tick.start { text-anchor: start; }
.tick.in { fill: var(--text); }
.hit { fill: transparent; cursor: pointer; }
.hit.sel { fill: var(--accent-soft); }
.grp { outline: none; }
.grp:focus-visible .hit { stroke: var(--warn); }
.rep { fill: var(--danger); opacity: 0.45; }
.lp { fill: none; stroke: var(--accent); stroke-width: 2; }
.lpd { fill: var(--accent); }
.cand { fill: none; stroke-width: 2; }
.cand.short, .short-t { stroke: var(--danger); fill: none; }
.short-t { fill: var(--danger); stroke: none; }
.cand.long { stroke: var(--left); }
.long-t { fill: var(--left); }
.star { stroke: var(--text-muted); stroke-dasharray: 2 3; }
.cursor { stroke: var(--warn); stroke-dasharray: 3 3; }
.grab { fill: transparent; cursor: ew-resize; }
</style>
