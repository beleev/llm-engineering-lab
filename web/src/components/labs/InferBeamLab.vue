<!--
  Beam search 实验台 (对应 llm_infer/m24_beam_search/beam.py:beam_search)。
  一件事: beam 找的是"最可能"的序列, 而最可能的往往更重复、更短 —— 长度惩罚只改 finished 候选之间怎么比。
-->
<template>
  <LabFrame
    title="Beam search — 更可能的序列, 不等于更好的序列"
    sub="- 模拟什么: beam search (束搜索) 是一类确定性近似解码策略: 贪心解码 (greedy) 每步仅保留单条最高概率路径, 而 beam search 在每步保留累计对数概率 (序列各 token 对数概率之和, 取值为负, 越接近 0 代表联合概率越高) 排名前 $w$ 的候选前缀, $w$ 称为束宽 (beam 宽度), 最终输出得分最高的一条序列。本实验检验「最大化序列联合概率」是否等同于「更高的生成质量」。其中图 ① 来自 Python 脚本实测 (基于随机权重 TinyLM, 对 10 个 prompt 各生成 20 个 token); 图 ② 为前端实时计算的序列长度惩罚数值算例。
- 图上是什么: 视图包含上下两张图表:
  - 图 ①: 横轴为束宽 $w$ (取值 1 至 32, 其中 $w = 1$ 退化为贪心解码)。红色柱表示二元重复率 rep-2 (输出中重复出现的相邻双 token 组合占比, 对应左轴, 数值越高复读越严重); 紫色折线表示每 token 平均对数概率 logP/token (对应右轴, 越高代表序列联合概率越大); 柱底标注的「输 2」等字样表示在 10 个测试 prompt 中有几个样本上束搜索的最终得分反而低于贪心解码。
  - 图 ②: 对比两条已终止候选序列的归一化得分。红线代表第 1 步即输出结束符 EOS 提前收尾的极短候选, 累计对数概率为 −3; 绿线代表长度为 15 个 token 的长候选, 累计对数概率为 −35。横轴为长度惩罚指数 $\alpha$, 纵轴为长度归一化得分 $\text{score}/\text{len}^\alpha$, 得分更高者胜出; 竖向虚线 $\alpha^*$ 标出两条曲线的交点临界值。
- 可以做什么: 支持两组交互调节:
  - 拖动「beam 宽度 w」滑杆或直接点击图 ① 中的柱条, 查看对应束宽 $w$ 下的各项实测指标。
  - 拖动「长度惩罚 α」滑杆、点击预设的 α 取 0 / 0.5 / 1 快捷按钮, 或拖动图 ② 中的黄色游标, 观察不同 $\alpha$ 下两条候选序列的胜负反转。
- 应该看到: 两张图表与右侧读数呈现如下规律:
  - 图 ①: 随着束宽 $w$ 增大, logP/token 稳步上升 (搜到的序列概率更高), 但重复率 rep-2 不降反升, 在 $w = 32$ 时达到 0.563 (而贪心解码为 0.453), 说明更高联合概率的序列反而更容易陷入局部循环复读。
  - 图 ②: 当 $\alpha = 0$ 时极短候选占优 (−3 高于 −35), 只有将 $\alpha$ 拖过临界点 0.91 后长候选才能实现反超。
  - 右侧四项读数: 前三项展示当前束宽 $w$ 下的实测 logP/token、rep-2 以及总分负于贪心解码的 prompt 数量; 末项显示当前 $\alpha$ 下图 ② 中胜出的候选序列。"
    module="llm_infer/m24"
    run="python -m llm_infer.m24_beam_search.demo"
    :challenge="{
      ask: '在图 ② 中, 长度惩罚系数 $\\alpha$ 至少需要调至多大, 长度为 15 个 token 的长候选序列才能实现得分反超?',
      answer: '约为 0.91。\n长候选反超的临界条件是两条序列的长度归一化得分相等, 即 $-3/1^\\alpha = -35/15^\\alpha$, 整理得 $15^\\alpha = 35/3$, 解得临界值 $\\alpha^* = \\ln(35/3)/\\ln 15 \\approx 0.91$。\n- 当 $\\alpha = 0$ 时: 不做长度归一化, 直接比较累计总分, $-3$ 高于 $-35$, 长度为 1 个 token 的短候选胜出。\n- 当 $\\alpha = 0.5$ 时: 短候选得分为 $-3/1 = -3$, 长候选得分为 $-35/\\sqrt{15} \\approx -9.0$, 仍是短候选胜出。\n- 当 $\\alpha = 1$ 时: 退化为比较每 token 平均对数概率, 短候选为 $-3$, 长候选为 $-35/15 \\approx -2.3$, 长候选反超胜出。\n这一计算与脚本实测结果完全吻合: 将 EOS 的 logit 加上 3 使模型具备主动终止倾向后, $\\alpha = 0$ 与 $\\alpha = 0.5$ 下的平均生成长度均仅为 1.3 个 token, 直至 $\\alpha = 1$ 时才恢复至 15.7 个 token (贪心解码基线为 11.4 个 token)。\nHugging Face 生成配置中的 length_penalty 参数正是此处的指数 $\\alpha$: 将累计对数概率除以 $\\text{len}^\\alpha$, 因对数概率为负值, 取 $\\alpha \\gt 0$ 会缩小长序列负分的绝对值从而鼓励长输出。该系数仅在比较已终止的完整候选序列时生效, 不改变束搜索中间步的扩展排序。',
    }"
  >
    <template #controls>
      <LabSlider v-model="wi" label="beam 宽度 w" :min="0" :max="5" :format="(i) => W[i]" />
      <LabSlider v-model="alpha" label="长度惩罚 α" :min="0" :max="1.5" :step="0.05" :format="(v) => v.toFixed(2)" />
      <div class="row"><button v-for="a in [0, 0.5, 1]" :key="a" type="button" :class="{ active: alpha === a }" @click="alpha = a">α = {{ a }}</button></div>
    </template>

    <svg ref="svg" viewBox="0 0 560 330" role="group" aria-label="beam 宽度与 log 概率和重复率; 长度惩罚下两条候选的分数">
      <text x="4" y="12" class="cap">① 实测, 不随 α 变化: 柱 = 重复率 rep-2 (左轴), 线 = logP/token (右轴)</text>
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

      <text x="4" y="196" class="cap">② 前端现算, 不随 w 变化: 归一化分数 score / len^α (越高越优先)</text>
      <line x1="40" x2="540" :y1="sy(0)" :y2="sy(0)" class="grid" />
      <polyline :points="curve(SHORT)" class="cand short" />
      <polyline :points="curve(LONG)" class="cand long" />
      <line :x1="ax(aStar)" :x2="ax(aStar)" y1="204" y2="306" class="star" />
      <text :x="ax(aStar) + 4" y="212" class="tick start">α* ≈ {{ aStar.toFixed(2) }}</text>
      <text v-for="a in [0, 0.5, 1, 1.5]" :key="a" :x="ax(a)" y="320" class="tick">{{ a }}</text>
      <text x="44" :y="sy(score(SHORT, 0)) - 4" class="tick start short-t">1 token 就收尾 (logP −3)</text>
      <text x="44" :y="sy(score(LONG, 0)) + 12" class="tick start long-t">15 token (logP −35)</text>
      <g class="draggable knob" tabindex="0" role="slider" aria-label="拖动改变 α" aria-valuemin="0" aria-valuemax="1.5" :aria-valuenow="alpha"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.right.prevent="alpha = step(0.05)" @keydown.left.prevent="alpha = step(-0.05)">
        <rect :x="ax(alpha) - 8" y="204" width="16" height="102" class="grab" />
        <line :x1="ax(alpha)" :x2="ax(alpha)" y1="204" y2="306" class="cursor" />
        <rect :x="ax(alpha) - 5" y="292" width="10" height="14" rx="3" class="handle" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>w = {{ W[wi] }} 时每 token 平均 log 概率 (greedy −2.620)</span><b :class="LOGP[wi] > LOGP[0] ? 'good' : ''">{{ LOGP[wi].toFixed(3) }}</b></div>
      <div class="kv"><span>重复率 rep-2 (greedy 0.453)</span><b :class="REP[wi] > 0.5 ? 'bad' : ''">{{ REP[wi].toFixed(3) }}</b></div>
      <div class="kv"><span>总分输给 greedy 的 prompt 数 (共 10 个)</span><b :class="LOSE[wi] ? 'bad' : 'good'">{{ LOSE[wi] }}</b></div>
      <div class="kv"><span>② α = {{ alpha.toFixed(2) }} 时胜出的候选</span><b :class="win === 'long' ? 'good' : 'bad'">{{ win === 'long' ? '15 token 的候选' : '1 token 就收尾' }}</b></div>
      <div class="lab-note">
        <p>对照 (demo 实测):</p>
        <ul class="pts">
          <li>采样 T=1 的 logP/token 是 −4.303, rep-2 是 0.003, 采样走的是低概率区域, 几乎不复读。</li>
          <li>给 EOS 的 logit 加 3 让模型学会停之后, α = 0 / 0.5 / 1 的平均输出长度是 1.3 / 1.3 / 15.7 个 token。</li>
        </ul>
        <p>图 ① 指标及上述平均生成长度实测值均取自本节 demo 脚本的 [1] 与 [3] 段输出; 图 ② 的两条候选总分沿用本章自测题的数量级 (−3 与 −35), 曲线数值由前端按公式实时计算。</p>
        <p>候选多样性对比 (来自 demo [2] 段实测): 针对同一 prompt 搜索出的 8 条 beam 候选序列平均仅有 3.5/20 个位置存在差异, 而独立采样生成的 8 条序列平均有 19.7/20 个位置不同; 束搜索保留的候选彼此高度同质, 若需多样化生成应采用随机采样策略。</p>
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
.grab { fill: transparent; }
.knob { cursor: ew-resize; outline: none; }
.handle { fill: var(--warn); stroke: var(--bg-card); }
.knob:focus-visible .handle { stroke: var(--text); stroke-width: 2; }
</style>
