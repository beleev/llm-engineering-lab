<!--
  推理时计算实验台 · 并行采样 (对应 llm_infer/m23_test_time_compute/tts.py:best_of_n / majority_vote / prm_beam_search)。
  一件事: 多采样之后"靠什么挑"决定上限 —— 投票只能消随机错, 外部判分器才能越过模型自己的众数。
  图 ① 的四条线和图 ② 是 demo 实测的静态数据; 只有公式那条线跟着 p、K 现算。
-->
<template>
  <LabFrame
    title="Test-time compute — 采 N 条之后靠什么挑, 决定能涨到多高"
    sub="- 模拟什么: 模型不换, 推理时多花 token 换正确率。任务是 200 道 4 步算术题, 玩具「模型」每一步有 6 个候选: 1 个正确, 4 个粗心错 (随机的, 换个样本就可能对), 1 个「看错运算符」的误解。32.5% 的题带一个陷阱步, 在那一步上误解的概率 (0.56) 比正确答案 (0.34) 还高。并行采样的做法: 每题采 N 条 (花 4N 个 token), 再靠某种办法从 N 条里挑一条交卷。挑法不同, 上限就不同。
      - 图上是什么: 图 ① 横轴是采样条数 N (1 到 64, 按 2 的幂排), 纵轴是 200 道题的正确率, 里面有五条线; 图 ② 在它下面:
        - 灰色虚线 pass@N: N 条里只要有一条对就算对, 是「挑得完美」时的上限。
        - 绿线 BoN + ORM: best-of-N, 用 ORM (outcome reward model, 只看最终答案打分的外部判分器, 这里是真值加 σ = 0.5 的噪声) 挑分最高的一条。
        - 紫线多数投票: 取 N 条最终答案里出现最多的那个, 不用判分器。
        - 红线: 多数投票只在陷阱题上的正确率。
        - 粉色虚线「公式」: $1-(1-p^K)^N$, 完美判分器下 best-of-N 的理论值, 跟着 p、K 滑杆变。
        - 图 ②: 比的是按步搜索和并行采样。每组柱子的 token 预算相同, 橙色柱是 PRM beam (每一步采 expand 个候选, 让 PRM, 也就是给每一步打分的判分器, 留下最好的 width 条), 绿色柱是花同样多 token 的 best-of-N + ORM; 绿色短线是 PRM 无噪声时 4×4 的成绩 0.970。
      - 可以做什么: 四种操作:
        - 拖「采样条数 N」、拖图 ① 的黄色把手或点线上的圆点, 改 N。
        - 点图例按钮隐藏或显示一条线。
        - 拖「单步正确率 p」和「链长 K」只改「公式」那条线 (四条实测线是 demo 跑出来的, 不会动)。
        - 点图 ② 的一组柱子, 右侧显示它的数字。
      - 应该看到: N 越大, pass@N 和 BoN + ORM 一路上涨, 多数投票在 0.71 附近饱和, 陷阱题上的投票过了 N = 16 反而往下掉。右侧读数:
        - 第一行: 当前 N 下 BoN + ORM 和 pass@N 的正确率。
        - 第二行: 多数投票在全部题和陷阱题上的正确率, 陷阱题过了拐点变红。
        - 第三行: 公式在当前 p、K、N 下的值, 以及要几条才能让它到 95%。
        - 第四行: 图 ② 选中那组里 PRM beam 对 best-of-N 的成绩, PRM beam 赢是绿色。"
    module="llm_infer/m23"
    run="python -m llm_infer.m23_test_time_compute.demo"
    :challenge="{
      ask: '先猜: N 从 16 加到 64, 多数投票在陷阱题上会变好还是变坏?',
      answer: '变坏: 陷阱题上的投票正确率从 0.254 (N = 16) 掉到 0.108 (N = 64)。\n原因在陷阱步:\n1. 那一步上模型最想写的是「看错运算符」的误解 (概率 0.56), 正确答案只有 0.34。\n2. 所以陷阱题的最终答案里, 出现最多的那个 (众数) 就是错的。\n3. 多数投票取的就是众数, N 越大, 样本里的众数越稳定地等于分布的众数, 投票就越稳定地选错。\nN 小的时候反而偶尔会因为运气选到正确答案。\n- 多数投票: 只用模型自己的分布, 能消掉的只有粗心错 (随机错)。\n- best-of-N + ORM: 外部判分器不管哪个答案出现得多, 只看对不对, 能从少数派里认出正确答案, N = 64 时到 0.970。\n判分器完美时 best-of-N 就等于 pass@N $= 1-(1-p^K)^N$: 单步正确率 $p = 0.75$、链长 $K = 4$ 时, 单条正确率 $0.75^4 \\approx 0.316$, 采 8 条就有 $1 - 0.684^8 \\approx 0.952$, 所以右侧「到 95% 要几条」显示 8。',
    }"
  >
    <template #controls>
      <LabSlider v-model="ni" label="采样条数 N" :min="0" :max="6" :format="(i) => NS[i]" />
      <div class="row">
        <button v-for="s in series" :key="s.key" type="button" :class="{ active: shown[s.key] }" :aria-pressed="shown[s.key]" @click="shown[s.key] = !shown[s.key]">
          <span class="sw" :style="{ background: s.c }" />{{ s.name }}
        </button>
      </div>
      <LabSlider v-model="p" label="单步正确率 p (公式)" :min="0.5" :max="0.95" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="K" label="链长 K (公式)" :min="1" :max="8" />
    </template>

    <svg ref="svg" viewBox="0 0 560 192" role="group" aria-label="正确率随 N 的曲线: 四条实测线和一条公式线">
      <text x="4" y="12" class="cap">① 正确率 vs N (横轴按 log₂ N)。实测的四条不随 p、K 变化</text>
      <line v-for="v in [0, 0.25, 0.5, 0.75, 1]" :key="v" x1="44" x2="550" :y1="cy(v)" :y2="cy(v)" class="grid" />
      <text v-for="v in [0, 0.5, 1]" :key="`t${v}`" x="38" :y="cy(v) + 3" class="tick end">{{ v }}</text>
      <template v-for="s in series" :key="s.key">
        <g v-if="shown[s.key]">
          <polyline :points="s.data.map((v, i) => `${cx(i)},${cy(v)}`).join(' ')" class="line" :class="s.key" />
          <g v-for="(v, i) in s.data" :key="i" class="pt" @click="ni = i">
            <circle :cx="cx(i)" :cy="cy(v)" r="9" class="hit" />
            <circle :cx="cx(i)" :cy="cy(v)" :r="i === ni ? 4.5 : 3" class="dot" :class="s.key" />
          </g>
        </g>
      </template>
      <text v-for="(n, i) in NS" :key="`n${i}`" :x="cx(i)" y="186" class="tick">{{ n }}</text>
      <g class="draggable knob" tabindex="0" role="slider" aria-label="拖动改变 N" aria-valuemin="1" aria-valuemax="64" :aria-valuenow="NS[ni]"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.right.prevent="ni = Math.min(6, ni + 1)" @keydown.left.prevent="ni = Math.max(0, ni - 1)">
        <rect :x="cx(ni) - 8" y="18" width="16" height="158" class="grab" />
        <line :x1="cx(ni)" :x2="cx(ni)" y1="18" y2="176" class="cursor" />
        <rect :x="cx(ni) - 5" y="16" width="10" height="14" rx="3" class="handle" />
      </g>
    </svg>

    <p class="blk">② 实测, 不随滑杆变化: 同样的 token 预算下, PRM beam (width×expand) 对 best-of-N (+ORM); 点一组柱子看数字</p>
    <svg viewBox="0 0 560 142" role="group" aria-label="同 token 下 PRM beam 与 best-of-N 的对比">
      <g v-for="(r, i) in BEAM" :key="r.cfg" class="grp" tabindex="0" role="button" :aria-label="`预算 ${r.tok} token`" :aria-pressed="i === bi" @click="bi = i" @keydown.enter="bi = i">
        <rect :x="gx(i) - 6" y="4" width="104" height="136" class="hit" :class="{ sel: i === bi }" />
        <rect :x="gx(i)" :y="by(r.beam)" width="40" :height="by(0) - by(r.beam)" class="bar beam" />
        <rect :x="gx(i) + 46" :y="by(r.bon)" width="40" :height="by(0) - by(r.bon)" class="bar bon" />
        <text :x="gx(i) + 20" :y="by(r.beam) - 3" class="tick">{{ r.beam.toFixed(2) }}</text>
        <text :x="gx(i) + 66" :y="by(r.bon) - 3" class="tick">{{ r.bon.toFixed(2) }}</text>
        <text :x="gx(i) + 43" y="136" class="tick">{{ r.tok }} tok</text>
      </g>
      <line :x1="gx(3) - 2" :x2="gx(3) + 42" :y1="by(0.97)" :y2="by(0.97)" class="ideal" />
      <text :x="gx(3) + 44" :y="by(0.97) + 3" class="tick start">σ=0</text>
    </svg>
    <p class="legend"><span class="sw beam" /> PRM beam <span class="sw bon" /> best-of-N + ORM</p>

    <template #stats>
      <div class="kv"><span>N = {{ NS[ni] }}: best-of-N + ORM / pass@N 的正确率</span><b :class="BON[ni] > 0.9 ? 'good' : ''">{{ BON[ni].toFixed(3) }} / {{ PASS[ni].toFixed(3) }}</b></div>
      <div class="kv"><span>多数投票的正确率 (全部题 / 只看陷阱题)</span><b :class="ni > PEAK && VOTE_TRAP[ni] < VOTE_TRAP[PEAK] ? 'bad' : ''">{{ VOTE[ni].toFixed(3) }} / {{ VOTE_TRAP[ni].toFixed(3) }}</b></div>
      <div class="kv"><span>公式 <Tex text="$1-(1-p^K)^N$" /> 的值 / 要到 95% 需要几条</span><b>{{ passN.toFixed(3) }} / {{ n95 }}</b></div>
      <div class="kv"><span>② PRM beam {{ BEAM[bi].cfg }} 对 best-of-{{ BEAM[bi].n }} ({{ BEAM[bi].tok }} token)</span><b :class="BEAM[bi].beam > BEAM[bi].bon ? 'good' : 'bad'">{{ BEAM[bi].beam.toFixed(3) }} 对 {{ BEAM[bi].bon.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>图 ① 的四条实测线和图 ② 的数字来自 demo 的 [1] [2] 段输出。「公式」那条按单条正确率 <Tex :text="`$q = p^K = ${q.toFixed(3)}$`" /> 在前端现算。</p>
        <p>图 ② 的读法:</p>
        <ul class="pts">
          <li>token 预算小时 PRM beam 赢 (28 token: 0.865 对 0.747): 错的一步当场被剪掉, 省下的 token 花在别的候选上。</li>
          <li>52 token 起打平甚至略输。</li>
          <li>PRM 判分无噪声时 4×4 能到 0.970 (绿线), 差距全来自判分噪声 σ = 0.5: 候选一多, 总有错步被噪声抬高。</li>
        </ul>
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
import { argmax, clamp } from '@/utils/labmath.js'

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
const PEAK = argmax(VOTE_TRAP)   // 陷阱题上投票的拐点 (N=16): 过了它, N 再大反而变差

const ni = ref(4), bi = ref(2), p = ref(0.75), K = ref(4)
const shown = reactive({ pass: true, bon: true, vote: true, trap: true, form: true })
const svg = ref(null)
const { start } = useDrag()

const cx = (i) => 60 + i * 80
const cy = (v) => 176 - v * 150
const gx = (i) => 14 + i * 110
const by = (v) => 124 - v * 100
const onMove = ({ x }) => { ni.value = clamp(Math.round((x - 60) / 80), 0, 6) }

// ★ 完美判分器下 best-of-N 就是 pass@N: 只要 N 条里有一条对
const q = computed(() => p.value ** K.value)
const passN = computed(() => 1 - (1 - q.value) ** NS[ni.value])
const n95 = computed(() => (q.value >= 0.95 ? 1 : Math.ceil(Math.log(0.05) / Math.log(1 - q.value))))
// 公式曲线和实测线画在同一张图里: 拖 p、K 时只有它动
const series = computed(() => [...SERIES, { key: 'form', name: '公式 (跟 p、K 变)', data: NS.map((n) => 1 - (1 - q.value) ** n), c: 'var(--right)' }])
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
.form { stroke: var(--right); fill: var(--right); }
.line.pass { stroke-dasharray: 4 3; }
.line.form { stroke-dasharray: 7 3; }
.pt { cursor: pointer; }
.hit { fill: transparent; cursor: pointer; }
.hit.sel { fill: var(--accent-soft); }
.grp { outline: none; }
.grp:focus-visible .hit { stroke: var(--warn); }
.cursor { stroke: var(--warn); stroke-dasharray: 3 3; }
.grab { fill: transparent; }
.knob { cursor: ew-resize; outline: none; }
.handle { fill: var(--warn); stroke: var(--bg-card); }
.knob:focus-visible .handle { stroke: var(--text); stroke-width: 2; }
.blk { font-size: 12px; color: var(--text-muted); margin: 14px 0 4px; }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
.sw.beam { background: var(--eye); }
.sw.bon { background: var(--left); opacity: 0.7; }
.bar.beam { fill: var(--eye); }
.bar.bon { fill: var(--left); opacity: 0.7; }
.ideal { stroke: var(--left); stroke-width: 2; }
.sw { display: inline-block; width: 10px; height: 10px; margin-right: 4px; border-radius: 2px; vertical-align: -1px; }
</style>
