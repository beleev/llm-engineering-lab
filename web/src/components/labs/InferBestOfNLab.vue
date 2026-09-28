<!--
  推理时计算实验台 · 并行采样 (对应 llm_infer/m23_test_time_compute/tts.py:best_of_n / majority_vote / prm_beam_search)。
  一件事: 多采样之后"靠什么挑"决定上限 —— 投票只能消随机错, 外部判分器才能越过模型自己的众数。
-->
<template>
  <LabFrame
    title="Test-time compute — 采 N 条之后靠什么挑"
    sub="200 道 4 步算术题, 32.5% 带一个陷阱步 (模型最想写的是看错运算符的答案)。
      - 上图: 每题采 N 条 (token = 4N), 四种挑法的正确率。拖竖线或点圆点改 N, 点图例隐藏一条线。
      - 下图: 同样的 token 数, PRM 逐步 beam 和 best-of-N 比。点一组柱子看数字。"
    module="llm_infer/m23"
    run="python -m llm_infer.m23_test_time_compute.demo"
    :challenge="{
      ask: '先猜: N 从 16 加到 64, 多数投票在陷阱题上会变好还是变坏? 再把下面的 p 调到 0.75、K 调到 4, 完美判分器要几条才能到 95%?',
      answer: '变坏: 0.254 → 0.108。\n陷阱题上最终答案的众数就是那个误解 (T=1 时 0.56 > 0.34)。N 越大, 样本众数越稳定地等于分布众数, 投票越稳定地选错。\n- 多数投票: 只用模型自己的分布, 消掉的是随机错。\n- best-of-N + ORM: 外部判分器能从少数派里认出正确答案, N=64 到 0.970。\n完美判分器下 BoN = pass@N = $1-(1-p^K)^N$。$0.75^4 \\approx 0.316$, $1 - 0.684^8 \\approx 0.952$, 所以要 8 条。',
    }"
  >
    <template #controls>
      <LabSlider v-model="ni" label="采样条数 N" :min="0" :max="6" :format="(i) => NS[i]" />
      <div class="row">
        <button v-for="s in SERIES" :key="s.key" type="button" :class="{ active: shown[s.key] }" :aria-pressed="shown[s.key]" @click="shown[s.key] = !shown[s.key]">
          <span class="sw" :style="{ background: s.c }" />{{ s.name }}
        </button>
      </div>
      <LabSlider v-model="p" label="单步正确率 p (公式)" :min="0.5" :max="0.95" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="K" label="链长 K (公式)" :min="1" :max="8" />
    </template>

    <svg ref="svg" viewBox="0 0 560 350" role="group" aria-label="正确率随 N 的曲线, 以及同 token 下 PRM beam 与 best-of-N 的对比">
      <text x="4" y="12" class="cap">① 正确率 vs N (横轴按 log₂ N)</text>
      <line v-for="v in [0, 0.25, 0.5, 0.75, 1]" :key="v" x1="44" x2="550" :y1="cy(v)" :y2="cy(v)" class="grid" />
      <text v-for="v in [0, 0.5, 1]" :key="`t${v}`" x="38" :y="cy(v) + 3" class="tick end">{{ v }}</text>
      <template v-for="s in SERIES" :key="s.key">
        <g v-if="shown[s.key]">
          <polyline :points="s.data.map((v, i) => `${cx(i)},${cy(v)}`).join(' ')" class="line" :class="s.key" />
          <g v-for="(v, i) in s.data" :key="i" class="pt" tabindex="0" role="button" :aria-label="`${s.name} N=${NS[i]}: ${v}`" @click="ni = i" @keydown.enter="ni = i">
            <circle :cx="cx(i)" :cy="cy(v)" r="9" class="hit" />
            <circle :cx="cx(i)" :cy="cy(v)" :r="i === ni ? 4.5 : 3" class="dot" :class="s.key" />
          </g>
        </g>
      </template>
      <text v-for="(n, i) in NS" :key="`n${i}`" :x="cx(i)" y="186" class="tick">{{ n }}</text>
      <line :x1="cx(ni)" :x2="cx(ni)" y1="18" y2="176" class="cursor" />
      <rect :x="cx(ni) - 8" y="18" width="16" height="158" class="grab draggable" tabindex="0" role="slider" aria-label="拖动改变 N"
        :aria-valuenow="NS[ni]" @pointerdown="start($event, { svg, onMove })"
        @keydown.right.prevent="ni = Math.min(6, ni + 1)" @keydown.left.prevent="ni = Math.max(0, ni - 1)" />

      <text x="4" y="212" class="cap">② 同 token 预算: PRM beam (width×expand) vs best-of-N (+ORM)</text>
      <g v-for="(r, i) in BEAM" :key="r.cfg" class="grp" tabindex="0" role="button" :aria-label="`预算 ${r.tok} token`" @click="bi = i" @keydown.enter="bi = i">
        <rect :x="gx(i) - 6" y="218" width="104" height="128" class="hit" :class="{ sel: i === bi }" />
        <rect :x="gx(i)" :y="by(r.beam)" width="40" :height="330 - by(r.beam)" class="bar beam" />
        <rect :x="gx(i) + 46" :y="by(r.bon)" width="40" :height="330 - by(r.bon)" class="bar bon" />
        <text :x="gx(i) + 20" :y="by(r.beam) - 3" class="tick">{{ r.beam.toFixed(2) }}</text>
        <text :x="gx(i) + 66" :y="by(r.bon) - 3" class="tick">{{ r.bon.toFixed(2) }}</text>
        <text :x="gx(i) + 43" y="342" class="tick">{{ r.tok }} tok</text>
      </g>
      <line :x1="gx(3) - 2" :x2="gx(3) + 42" :y1="by(0.97)" :y2="by(0.97)" class="ideal" />
      <text :x="gx(3) + 44" :y="by(0.97) + 3" class="tick start">σ=0</text>
    </svg>

    <template #stats>
      <div class="kv"><span>N = {{ NS[ni] }} · token</span><b>{{ NS[ni] * 4 }}</b></div>
      <div class="kv"><span>pass@N (完美判分器上限)</span><b>{{ PASS[ni].toFixed(3) }}</b></div>
      <div class="kv"><span>best-of-N + ORM</span><b :class="BON[ni] > 0.9 ? 'good' : ''">{{ BON[ni].toFixed(3) }}</b></div>
      <div class="kv"><span>多数投票 (全部 / 陷阱题)</span><b :class="VOTE_TRAP[ni] < VOTE_TRAP[4] ? 'bad' : ''">{{ VOTE[ni].toFixed(3) }} / {{ VOTE_TRAP[ni].toFixed(3) }}</b></div>
      <div class="kv"><span>② {{ BEAM[bi].cfg }} vs best-of-{{ BEAM[bi].n }}</span><b :class="BEAM[bi].beam > BEAM[bi].bon ? 'good' : 'bad'">{{ BEAM[bi].beam.toFixed(3) }} vs {{ BEAM[bi].bon.toFixed(3) }}</b></div>
      <div class="kv"><span>公式: 单条 <Tex text="$q = p^K$" /></span><b>{{ q.toFixed(3) }}</b></div>
      <div class="kv"><span>公式: <Tex text="$1-(1-q)^N$" /> / 到 95% 要几条</span><b>{{ passN.toFixed(3) }} / {{ n95 }}</b></div>
      <div class="lab-note">
        <p>① ② 的数字来自 demo 的 [1] [2] 段输出; 最后两行按公式在前端现算。</p>
        <p>② 小预算时 PRM beam 赢 (28 token: 0.865 vs 0.747), 52 token 起打平。PRM 无噪声时 4×4 到 0.970 (绿线), 差距来自判分噪声 σ=0.5。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

// 数据: python -m llm_infer.m23_test_time_compute.demo 的 [1] 段 (每题 64 条切成互不重叠的 N 条一组)
const NS = [1, 2, 4, 8, 16, 32, 64]
const PASS = [0.275, 0.471, 0.701, 0.876, 0.973, 1.0, 1.0]
const BON = [0.275, 0.439, 0.628, 0.775, 0.865, 0.94, 0.97]
const VOTE = [0.275, 0.277, 0.388, 0.569, 0.706, 0.725, 0.71]
const VOTE_TRAP = [0.151, 0.154, 0.169, 0.246, 0.254, 0.2, 0.108]
// 同一 demo 的 [2] 段: PRM beam 的 token = expand + (K−1)·width·expand, BoN 取同 token 数的 N
const BEAM = [
  { cfg: '1×2', tok: 8, beam: 0.575, n: 2, bon: 0.442 },
  { cfg: '2×2', tok: 14, beam: 0.68, n: 4, bon: 0.623 },
  { cfg: '2×4', tok: 28, beam: 0.865, n: 7, bon: 0.747 },
  { cfg: '4×4', tok: 52, beam: 0.83, n: 13, bon: 0.849 },
  { cfg: '4×8', tok: 104, beam: 0.92, n: 26, bon: 0.932 },
]
const SERIES = [
  { key: 'pass', name: 'pass@N', data: PASS, c: 'var(--text-muted)' },
  { key: 'bon', name: 'BoN + ORM', data: BON, c: 'var(--left)' },
  { key: 'vote', name: '多数投票', data: VOTE, c: 'var(--accent)' },
  { key: 'trap', name: '投票 · 陷阱题', data: VOTE_TRAP, c: 'var(--danger)' },
]

const ni = ref(4), bi = ref(2), p = ref(0.75), K = ref(4)
const shown = reactive({ pass: true, bon: true, vote: true, trap: true })
const svg = ref(null)
const { start } = useDrag()

const cx = (i) => 60 + i * 80
const cy = (v) => 176 - v * 150
const gx = (i) => 14 + i * 110
const by = (v) => 330 - v * 100
const onMove = ({ x }) => { ni.value = clamp(Math.round((x - 60) / 80), 0, 6) }

// ★ 完美判分器下 best-of-N 就是 pass@N: 只要 N 条里有一条对
const q = computed(() => p.value ** K.value)
const passN = computed(() => 1 - (1 - q.value) ** NS[ni.value])
const n95 = computed(() => (q.value >= 0.95 ? 1 : Math.ceil(Math.log(0.05) / Math.log(1 - q.value))))
</script>

<style scoped>
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 10px; fill: var(--text-muted); }
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); text-anchor: middle; }
.tick.end { text-anchor: end; }
.tick.start { text-anchor: start; fill: var(--left); }
polyline.line { fill: none; stroke-width: 2; }
.pass { stroke: var(--text-muted); fill: var(--text-muted); }
.bon { stroke: var(--left); fill: var(--left); }
.vote { stroke: var(--accent); fill: var(--accent); }
.trap { stroke: var(--danger); fill: var(--danger); }
.line.pass { stroke-dasharray: 4 3; }
.pt { cursor: pointer; outline: none; }
.pt:focus-visible .dot { stroke: var(--warn); stroke-width: 3; }
.hit { fill: transparent; cursor: pointer; }
.hit.sel { fill: var(--accent-soft); }
.grp { outline: none; }
.grp:focus-visible .hit { stroke: var(--warn); }
.cursor { stroke: var(--warn); stroke-dasharray: 3 3; }
.grab { fill: transparent; cursor: ew-resize; }
.bar.beam { fill: var(--eye); }
.bar.bon { fill: var(--left); opacity: 0.7; }
.ideal { stroke: var(--left); stroke-width: 2; }
.sw { display: inline-block; width: 10px; height: 10px; margin-right: 4px; border-radius: 2px; vertical-align: -1px; }
</style>
