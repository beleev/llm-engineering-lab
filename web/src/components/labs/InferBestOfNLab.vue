<!--
  推理时计算实验台 · 并行采样 (对应 llm_infer/m23_test_time_compute/tts.py:best_of_n / majority_vote / prm_beam_search)。
  一件事: 多采样之后"靠什么挑"决定上限 —— 投票只能消随机错, 外部判分器才能越过模型自己的众数。
  图 ① 的四条线和图 ② 是 demo 实测的静态数据; 只有公式那条线跟着 p、K 现算。
-->
<template>
  <LabFrame
    title="Test-time compute — 采 N 条之后靠什么挑, 决定能涨到多高"
    sub="- 模拟什么: 固定模型权重不变, 在推理阶段通过增加采样 token 预算换取更高的解题准确率。评测基准为 200 道 4 步算术题, 合成推理模型在每一步面对 6 个候选结果: 1 个正确值、4 个随机粗心错误 (各次采样相互独立, 重采可能恢复正确) 以及 1 个源于看错运算符的系统性误解。其中 32.5% 的题目包含一个陷阱步, 模型在该步生成系统性误解的概率 (0.56) 高于生成正确答案的概率 (0.34)。并行采样的通用流程是对每题独立采样 N 条解答 (消耗 4N 个 token), 再按特定聚合准则从中选出一条作为最终输出; 聚合机制决定了准确率随采样规模扩展的上限。
- 图上是什么: 上方图 ① 横轴为采样条数 N (取值 1 至 64, 按 2 的幂次递增), 纵轴为 200 道题的总体或子集准确率, 共绘制五条曲线; 下方为图 ②:
  - 灰色虚线 pass@N: 只要 N 条采样中至少有一条正确即计为通过, 代表拥有理想无噪判分器时的准确率理论上界。
  - 绿线 BoN + ORM: 即 best-of-N 配合结果奖励模型 (此处以真值叠加标准差 $\sigma = 0.5$ 的高斯噪声模拟带噪外部判分器), 选取评分最高的一条候选作答。
  - 紫线多数投票: 无需外部判分器, 直接选取 N 条采样最终答案里出现频次最高的众数。
  - 红线: 单独统计多数投票在含陷阱步题目子集上的准确率。
  - 粉色虚线「公式」: 理想判分器下 best-of-N 的解析理论曲线 $1-(1-p^K)^N$, 随单步正确率 p 与链长 K 滑杆实时联动。
  - 图 ②: 在相同总 token 预算下对比逐步树搜索与句级并行采样。橙色柱为 PRM beam (每步采样 expand 个单步候选, 由过程奖励模型 PRM 筛选保留得分最高的前 width 条分支), 绿色柱为消耗同等 token 预算的 best-of-N + ORM; 绿色短横线标出无噪声理想 PRM 在 4×4 配置下达到的准确率 0.970。
- 可以做什么: 支持四类交互操作:
  - 拖动「采样条数 N」滑杆、拖动图 ① 中的黄色游标或点击曲线节点以改变采样规模 N。
  - 点击图例按钮单独隐藏或显示某条曲线。
  - 拖动「单步正确率 p」与「链长 K」滑杆: 调整粉色「公式」理论曲线的形态 (其余四条实测曲线来自脚本离线评测数据, 保持固定)。
  - 点击图 ② 中的任意一组对比柱, 在右侧面板查看该预算档位下的详细准确率读数。
- 应该看到: 随着采样规模 N 增大, pass@N 与 BoN + ORM 持续上升, 而多数投票总体准确率在 0.71 附近趋于饱和, 其在陷阱题子集上的准确率在超过 $N = 16$ 后反而调头下降。右侧四行读数含义如下:
  - 第一行: 当前 N 下 BoN + ORM 的实测准确率与 pass@N 上界。
  - 第二行: 多数投票在全集与陷阱题子集上的准确率, 当陷阱题进入逆向退化区间时标红提示。
  - 第三行: 解析公式在当前 p、K、N 参数下的理论准确率, 以及达到 95% 准确率所需的最小采样条数。
  - 第四行: 图 ② 当前选中预算组中 PRM beam 与 best-of-N 的准确率对比, PRM beam 占优时显示为绿色。"
    module="llm_infer/m23"
    run="python -m llm_infer.m23_test_time_compute.demo"
    :challenge="{
      ask: '当采样条数 N 从 16 增至 64 时, 多数投票在含陷阱步的题目子集上准确率会上升还是下降?',
      answer: '准确率反而下降: 陷阱题子集上的多数投票准确率从 $N = 16$ 时的 0.254 跌至 $N = 64$ 时的 0.108。\n根本原因在于陷阱步的分布偏置:\n1. 在陷阱步上, 模型生成系统性误解的概率 (0.56) 高于生成正确答案的概率 (0.34)。\n2. 由此导致陷阱题最终答案分布的众数本身就是错误答案。\n3. 多数投票本质上是用有限样本众数逼近真实分布众数; 随着采样规模 N 增大, 大数定律使样本众数越来越稳定地收敛于该错误众数, 从而导致投票结果更加确定地选错。\n采样数较小时, 样本波动反而偶发使正确答案成为局部多数。两类聚合机制的能力边界对比如下:\n- 多数投票: 完全依赖策略模型自身的输出分布, 仅能通过统计平均消除零均值的随机粗心错误, 无法纠正模型分布占优的系统性偏差。\n- best-of-N + ORM: 借助独立于生成分布的外部验证信号评判正误, 无视候选答案的出现频次, 能够从低频少数派样本中识别出正确答案, 在 $N = 64$ 时总体准确率可达 0.970。\n当外部判分器理想无噪时, best-of-N 准确率即等于理论上限 $\\text{pass@}N = 1-(1-p^K)^N$: 取单步正确率 $p = 0.75$、推理链长 $K = 4$, 则单条完整链路全对的概率为 $0.75^4 \\approx 0.316$, 独立采样 8 条即可达到 $1 - 0.684^8 \\approx 0.952$, 因此右侧面板「到 95% 要几条」显示为 8。',
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
