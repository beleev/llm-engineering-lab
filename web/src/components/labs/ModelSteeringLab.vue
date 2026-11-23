<!--
  激活引导实验台 (对应 llm_models/run_models/foundation/steering/demo.py: mean_diff / steer)。
  只讲一件事: 探针在混杂数据上读出的方向, 不一定是下游真正在读的方向; 沿它加向量改不了行为, 加得越长副作用越大。
  玩具: 残差流画成二维平面, 横轴是情绪方向 (下游用它选形容词), 纵轴是话题方向 (下游用它选名词)。
  探针 = 混杂数据上的线性判别 (两类均值差除以各轴方差), 与 Python 里的逻辑回归是同一类线性读出。
-->
<template>
  <LabFrame
    title="激活引导 — 能读出的方向, 不一定是加进去能改变行为的方向"
    sub="- 模拟什么: 往某一层的残差流里加一个向量, 让模型的行为朝某个方向偏, 这叫激活引导。这里把残差流画成一个平面: 横轴是情绪方向, 下游层读它决定说正面词还是负面词; 纵轴是话题方向, 下游层读它决定说运动还是食物。线性探针是在激活上训练的一个线性分类器, 这里用它在混杂数据 (正面总配运动、负面总配食物) 上读情绪标签。
- 图上是什么: 实心点是当前数据里的提示激活, 绿色是正面, 粉色是负面, 圆点是运动话题, 方块是食物话题; 淡色点是另一种搭配。虚线箭头是探针学到的方向。实线箭头从负面提示的中心出发, 是你加的引导向量, 空心点是负面提示加上向量之后的位置。
- 可以做什么: 三个按钮把引导向量对准情绪方向、话题方向或探针方向。滑杆调系数, 向量长度等于系数乘情绪方向两类均值之差的长度。直接拖箭头的端点可以任意改方向和长度 (选中端点后也可以用方向键)。数据按钮在混杂数据和反着配的数据之间切换, 看探针还对不对。
- 应该看到: 混杂数据上探针准确率是 1.00, 换到反着配的数据就掉到 0 附近, 因为它主要读的是话题。引导向量对准情绪方向时, 负面提示的正面词概率随系数上升; 对准话题方向或探针方向时, 正面词概率几乎不动, 改的是名词。右侧的副作用读数随向量长度的平方增长。最下面的表是 Python 实测, 不随左侧变化。"
    module="llm_models/run_models/foundation/steering/demo.py"
    run="python -m llm_models.run_models.foundation.steering.demo"
    :challenge="{
      ask: '探针在混杂数据上的准确率是 1.00。先预测: 点「对准探针方向」, 把系数拉到 2, 负面提示的正面词概率会变成多少? 再点「反着配的数据」, 探针准确率是多少?',
      answer: '答案: 正面词概率只从约 0.15 升到 0.25 左右, 而对准情绪方向、同样系数 2 时接近 1.00; 反着配的数据上探针准确率是 0。\n原因:\n- 混杂数据里情绪和话题完全相关, 探针用哪个方向都能拿满分。话题方向上两类拉得更开 (均值差 3, 情绪只有 2), 类内又散得更窄, 线性判别按 \'均值差除以类内方差\' 加权, 所以探针几乎完全对准了话题 (横向分量不到 0.1)。\n- 下游层只读横轴决定形容词, 沿探针方向加向量时只有很小的横向分量起作用, 长度几乎都花在了话题上, 改的是名词。\n- 反着配以后, 正面变成了食物话题, 探针按话题判就全错了。Python demo 里逻辑回归探针在反着配的数据上准确率 0.000, 沿它加 2 倍长度时正面词概率仍是 0.10。\n要找能改变行为的方向, 对照样本只能差目标属性 (CAA 的做法), 找到以后还要用干预去验证。'
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="d in DIRS" :key="d.id" type="button" :class="{ active: dirId === d.id }" @click="aim(d.id)">对准{{ d.label }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: data === 'conf' }" @click="data = 'conf'">混杂数据 (正面配运动)</button>
        <button type="button" :class="{ active: data === 'flip' }" @click="data = 'flip'">反着配的数据</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="coef" label="引导系数 c" :min="0" :max="4" :step="0.25" />
    </template>

    <svg ref="svg" viewBox="0 0 320 320" class="plane" role="group" aria-label="残差流平面: 情绪方向与话题方向, 可拖动引导向量">
      <line x1="0" y1="160" x2="320" y2="160" class="axis" /><line x1="160" y1="0" x2="160" y2="320" class="axis" />
      <text x="314" y="152" class="lbl" text-anchor="end">情绪方向 (下游选形容词) →</text>
      <text x="166" y="12" class="lbl">↑ 话题方向 (运动)</text>
      <g v-for="(c, k) in clusters" :key="k" :opacity="c.shown ? 1 : 0.18">
        <template v-for="(p, i) in c.pts" :key="i">
          <circle v-if="c.sport" :cx="X(p[0])" :cy="Y(p[1])" r="3" :class="c.pos ? 'pos' : 'neg'" />
          <rect v-else :x="X(p[0]) - 2.5" :y="Y(p[1]) - 2.5" width="5" height="5" :class="c.pos ? 'pos' : 'neg'" />
        </template>
      </g>
      <circle v-for="(p, i) in negPts" :key="'g' + i" :cx="X(p[0] + v[0])" :cy="Y(p[1] + v[1])" r="3" class="ghost" />
      <line x1="160" y1="160" :x2="X(probe[0] * 2.5)" :y2="Y(probe[1] * 2.5)" class="probe" />
      <text :x="X(probe[0] * 2.5) + 4" :y="Y(probe[1] * 2.5)" class="lbl probe-t">探针方向</text>
      <line :x1="X(-1)" :y1="Y(0)" :x2="X(-1 + v[0])" :y2="Y(v[1])" class="arrow" />
      <circle :cx="X(-1 + v[0])" :cy="Y(v[1])" r="8" class="tip draggable" tabindex="0" role="slider" aria-label="引导向量端点"
              :aria-valuetext="`横向 ${v[0].toFixed(1)}, 纵向 ${v[1].toFixed(1)}`" @pointerdown="drag" @keydown="nudge" />
    </svg>

    <template #stats>
      <div class="kv"><span>探针在当前数据上的准确率</span><b :class="probeAcc > 0.9 ? 'good' : 'bad'">{{ probeAcc.toFixed(2) }}</b></div>
      <div class="kv"><span>负面提示说正面词的概率</span><b :class="posRate > 0.5 ? 'good' : ''">{{ posRate.toFixed(2) }}</b></div>
      <div class="kv"><span>名词是运动类的概率 (不引导 0.50)</span><b>{{ sportRate.toFixed(2) }}</b></div>
      <div class="kv"><span>副作用: 无关任务 loss 增加 (玩具, ∝ 长度²)</span><b :class="side > 0.3 ? 'bad' : 'good'">+{{ side.toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>Python 实测, 不随左侧变化 (负面提示的正面词概率; 最后一列是 d_mood 的副作用):</p>
        <table class="tbl mono">
          <tr><th>系数</th><th>d_mood</th><th>d_topic</th><th>探针</th><th>计数 loss</th></tr>
          <tr v-for="r in MEASURED" :key="r[0]"><td v-for="(x, j) in r" :key="j">{{ x }}</td></tr>
        </table>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { mulberry32, randn, clamp } from '@/utils/labmath.js'

const MEASURED = [['0', '0.10', '0.10', '0.10', '+0.000'], ['1', '0.13', '0.10', '0.10', '+0.043'], ['2', '0.69', '0.11', '0.10', '+0.484'],
  ['4', '0.78', '0.13', '0.15', '+1.156'], ['8', '0.82', '0.05', '0.21', '+1.514']]
const MOOD = 2, TOPIC = 3                                          // 两类均值之差: 情绪 2, 话题 3 (话题拉得更开)
const SD = [0.8, 0.3]                                              // 类内标准差: 情绪方向上散得更开
const X = (x) => 160 + x * 40, Y = (y) => 160 - y * 40
const seed = ref(1), data = ref('conf'), dirId = ref('mood'), v = ref([MOOD, 0])
const svg = ref(null)

const clusters = computed(() => {
  const r = mulberry32(seed.value * 131)
  return [[1, 1], [1, 0], [0, 1], [0, 0]].map(([pos, sport]) => ({
    pos, sport,
    shown: data.value === 'conf' ? pos === sport : pos !== sport, // 混杂: 正面配运动、负面配食物
    pts: Array.from({ length: 16 }, () => [(pos ? 1 : -1) * MOOD / 2 + SD[0] * randn(r), (sport ? 1 : -1) * TOPIC / 2 + SD[1] * randn(r)]),
  }))
})
const negPts = computed(() => clusters.value.filter((c) => !c.pos).flatMap((c) => c.pts))
// ★ 探针只在混杂数据上训练: 两类均值差除以各轴方差 (对角协方差的线性判别)
const probe = computed(() => {
  const conf = clusters.value.filter((c) => c.pos === c.sport)
  const mean = (pts, k) => pts.reduce((s, p) => s + p[k], 0) / pts.length
  const varOf = (k) => conf.flatMap((c) => c.pts.map((p) => (p[k] - mean(c.pts, k)) ** 2)).reduce((a, b) => a + b, 0) / 32
  const w = [0, 1].map((k) => (mean(conf[0].pts, k) - mean(conf[1].pts, k)) / varOf(k))
  const n = Math.hypot(...w)
  return w.map((x) => x / n)
})
const probeAcc = computed(() => {
  const shown = clusters.value.filter((c) => c.shown)
  const score = (p) => p[0] * probe.value[0] + p[1] * probe.value[1]
  return shown.reduce((s, c) => s + c.pts.filter((p) => (score(p) > 0) === !!c.pos).length, 0) / 32
})
const sig = (z) => 1 / (1 + Math.exp(-z))
const posRate = computed(() => negPts.value.reduce((s, p) => s + sig(2.5 * (p[0] + v.value[0])), 0) / negPts.value.length)   // 下游只读横轴
const sportRate = computed(() => negPts.value.reduce((s, p) => s + sig(2 * (p[1] + v.value[1])), 0) / negPts.value.length)
const side = computed(() => 0.06 * (v.value[0] ** 2 + v.value[1] ** 2))

const DIRS = [{ id: 'mood', label: '情绪方向 d_mood' }, { id: 'topic', label: '话题方向 d_topic' }, { id: 'probe', label: '探针方向' }]
const unitOf = (id) => (id === 'mood' ? [1, 0] : id === 'topic' ? [0, 1] : probe.value)
const coef = computed({
  get: () => Math.round((Math.hypot(...v.value) / MOOD) * 4) / 4,
  set: (c) => { const u = Math.hypot(...v.value) > 1e-6 ? v.value.map((x) => x / Math.hypot(...v.value)) : unitOf(dirId.value); v.value = u.map((x) => x * c * MOOD) },
})
function aim(id) { dirId.value = id; const c = coef.value || 1; v.value = unitOf(id).map((x) => x * c * MOOD) }
const { start } = useDrag()
function drag(e) {
  dirId.value = null
  start(e, { svg: svg.value, onMove: ({ x, y }) => { v.value = [clamp((x - 160) / 40 + 1, -3, 5), clamp((160 - y) / 40, -3.8, 3.8)] } })
}
function nudge(e) {
  const d = { ArrowRight: [0.25, 0], ArrowLeft: [-0.25, 0], ArrowUp: [0, 0.25], ArrowDown: [0, -0.25] }[e.key]
  if (d) { e.preventDefault(); dirId.value = null; v.value = [v.value[0] + d[0], v.value[1] + d[1]] }
}
</script>

<style scoped>
.plane { width: 100%; max-width: 360px; display: block; }
.axis { stroke: var(--border-strong); stroke-width: 1; }
.lbl { font-size: 9px; fill: var(--text-dim); }
.pos { fill: var(--left); }
.neg { fill: var(--right); }
.ghost { fill: none; stroke: var(--right); stroke-width: 1; opacity: 0.7; }
.probe { stroke: var(--eye); stroke-width: 2; stroke-dasharray: 5 3; }
.probe-t { fill: var(--eye); }
.arrow { stroke: var(--accent); stroke-width: 2.5; }
.tip { fill: var(--accent); stroke: var(--bg); stroke-width: 2; cursor: grab; }
.tip:focus { outline: none; stroke: var(--text); }
.tbl { border-collapse: collapse; font-size: 11px; margin-top: 4px; }
.tbl td, .tbl th { padding: 1px 6px; text-align: right; color: var(--text-muted); font-weight: 400; }
.tbl th { color: var(--text-dim); }
</style>
