<!--
  LLM-as-judge 的位置偏差 (对应 llm_train/m21_llm_eval:rule_judge / judge_swap)。
  只讲一件事: 裁判给第一个位置加分, 单次判就偏向 A; 交换顺序各判一次、两次一致才算, 偏差抵消, 代价是平局。
  在前端按 rule_judge 的同一公式现模拟 2000 对 (质量 ~ N(0,1), 裁判 = 质量差 + bias + noise·N(0,1))。
  随机数与 numpy 不同, 数字会和 demo 差一点; demo 第 [4] 段 (bias 0.8, noise 0.5) 的实测写在读数区。
-->
<template>
  <LabFrame
    title="LLM 裁判的位置偏差 — 先出现的回答占便宜吗? 交换顺序能治吗?"
    sub="- 模拟什么: 开放式写作与多轮对话等缺乏唯一标准答案的任务, 通常借助大模型担任裁判 (LLM-as-judge), 将成对回答 A 与 B 一并输入并判定孰优孰劣。真实的 LLM 裁判普遍存在位置偏差, 即倾向于偏爱排在前面的回答。本实验台使用线性阈值模型模拟该现象: 裁判打分等于两回答真实质量差加上先手偏置分 bias 再叠加随机噪声, 打分大于 0 时判定排在首位者胜出。前端基于该公式实时模拟 2000 对回答, 并在散点图中绘制其中 400 对。交换判是消除位置偏差的标准流程:
  1. 将 A 置于首位评判一次。
  2. 将 B 置于首位再评判一次。
  3. 仅当两次评判均选出同一回答时才判其获胜, 否则记为平局。
- 图上是什么: 图中每个散点代表一对候选回答, 横轴为真实质量差 $q_A - q_B$ (右侧表示 A 真实质量更优, 左侧表示 B 更优), 纵坐标仅作随机抖动以便展开散点。
  - 散点颜色编码评判结果: 绿点表示判定正确, 红点表示判定错误, 灰点表示交换判下记为平局的样本对。
  - 中央实线标示真实质量差为 0 的无偏分界线。
  - 黄色虚线标示单次判的实际决策边界: 由于裁判对首个位置 (A) 额外加上了 bias 分, 决策边界由 0 左移至 −bias; 落在黄色虚线与 0 实线之间的样本对原本是 B 质量更优, 但在单次判下会被误判为 A 胜出。
- 可以做什么: 控件从上到下依次为:
  - 点击「单次判」或「交换判 (两次一致才算)」切换评判协议。
  - 拖动「先手加分 bias」滑杆, 或直接左右拖动图中黄色虚线, 调节裁判的位置偏好强度。
  - 拖动「裁判噪声」滑杆调节随机评判误差的波动幅度。
  - 点击「换一组数据」重新抽样生成一批候选回答对。
- 应该看到: 单次判模式下红点密集分布在黄色虚线与 0 实线之间的带状区域; 切换至交换判后, 该区域的红点整体转化为灰色平局点。右侧读数含义如下:
  - 「选第一个 (A) 的比例」: 当 bias 较大时会显著高于 50%, 而末行「真实 A 更好的比例」始终接近一半。
  - 分组准确率: 分别统计真实更优回答位于 A 位与位于 B 位时的判对率。单次判下 B 更优组的准确率显著偏低, 两组差距超过 0.1 时标红; 交换判下两组准确率基本一致 (差距小于 0.05 时显示绿色)。
  - 「平局比例」: 交换判付出的覆盖率代价; 当 bias 取 0.8 时约四成样本对无法分出胜负, 比例超过 0.3 时标红预警。"
    module="llm_train/m21"
    run="python -m llm_train.m21_llm_eval.demo"
    :challenge="{
      ask: '在单次判模式下, 误判红点集中在分界线的哪一侧? 切换至交换判后, 这批红点转化成了什么状态?',
      answer: '结论:\n- 单次判模式下, 红点集中在黄色虚线与 0 实线之间 (即 0 的左侧区域)。\n- 切换至交换判后, 这批红点整体转化为灰色的平局样本。\n机制解析:\n- 单次判: 裁判打分等于真实质量差加上先手偏置 bias 与随机噪声, 因此只要 A 的真实质量落后于 B 的幅度不超过 bias, 裁判仍会判定首位的 A 胜出。横轴位于 −bias 与 0 之间的样本对原本是 B 质量更优, 但因真实差距小于 bias 而被系统性误判给先出现的 A, 形成图中的红色带状区域; 调大 bias 时该红区宽度随之扩大。其结果是「好的在 A」分组准确率接近 1, 而「好的在 B」分组准确率跌至 0.6 左右 (m21 实测分别为 0.994 与 0.593), 「选第一个的比例」高达 0.700, 尽管真实 A 更优的比例仅为 0.498。此外当噪声较大时, 0 线两侧还会出现少量零星红点, 源于与位置无关的随机评判波动。\n- 交换判: 让 A 与 B 轮流处于首位各评判一次, 双方各获得一次先手加分从而相互抵消。仅当两次评判选出同一回答时才计为胜出: 真实质量差距显著的样本对在两次评判中结论一致并被准确判对; 原先落入红区的样本对在 A 居首时选 A、B 居首时选 B, 因两次结果矛盾而被归为平局, 红点由此转为灰点, 两组分组准确率同步恢复至 0.99 左右 (实测分别为 0.993 与 0.991)。\n- 交换判的工程代价:\n  - 裁判推理调用次数翻倍。\n  - 当 bias 为 0.8 时约有 40% 的样本对无法分出胜负 (实测平局率为 0.407), 主要集中在双方真实质量接近、位置偏置足以掩盖真实差距的区间。实际评测通常将平局记为双方各得 0.5 分, 或引入多个裁判投票裁决。\n需要注意, 位置交换仅能消除位置偏差; 真实 LLM 裁判还普遍存在长度偏差 (偏好更长的回答) 与自我偏好 (偏好同源模型生成的文本), 这两类偏差与摆放顺序无关, 无法通过交换位置消除。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'single' }" @click="mode = 'single'">单次判</button>
        <button type="button" :class="{ active: mode === 'swap' }" @click="mode = 'swap'">交换判 (两次一致才算)</button>
        <button type="button" @click="seed++">换一组数据</button>
      </div>
      <LabSlider v-model="bias" label="先手加分 bias" :min="0" :max="1.5" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="noise" label="裁判噪声" :min="0.1" :max="1.5" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="每对回答的真实质量差与裁判结果, 分界虚线可拖">
      <line :x1="px(0)" :x2="px(0)" :y1="8" :y2="H - 20" class="zero" />
      <g v-for="t in [-3, -2, -1, 0, 1, 2, 3]" :key="t">
        <text :x="px(t)" :y="H - 6" class="tick" text-anchor="middle">{{ t }}</text>
      </g>
      <circle v-for="(d, i) in shown" :key="i" :cx="px(d.diff)" :cy="d.y" r="2.6" :class="d.cls" />
      <g class="draggable focus-inset" tabindex="0" role="slider" aria-label="先手加分 bias" :aria-valuenow="bias"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (bias = clamp(Math.round(-pxInv(x) * 20) / 20, 0, 1.5)) })"
        @keydown.left.prevent.stop="bias = clamp(+(bias + 0.05).toFixed(2), 0, 1.5)" @keydown.right.prevent.stop="bias = clamp(+(bias - 0.05).toFixed(2), 0, 1.5)">
        <line :x1="px(-bias)" :x2="px(-bias)" :y1="8" :y2="H - 20" class="bound" />
        <rect :x="px(-bias) - 8" y="4" width="16" height="16" rx="4" class="handle" />
        <text :x="px(-bias) - 12" y="16" class="hl" text-anchor="end">−bias</text>
      </g>
    </svg>

    <template #stats>
      <template v-if="mode === 'single'">
        <div class="kv"><span>选第一个 (A) 的比例 (没有偏差时应接近下面「真实 A 更好」的比例)</span><b :class="r.pickA > 0.6 ? 'bad' : 'good'">{{ r.pickA.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 A 位时判对的比例</span><b>{{ r.accA1.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 B 位时判对的比例</span><b :class="r.accA1 - r.accB1 > 0.1 ? 'bad' : 'good'">{{ r.accB1.toFixed(3) }}</b></div>
      </template>
      <template v-else>
        <div class="kv"><span>平局比例 (两次判不一致, 给不出胜负)</span><b :class="r.tie > 0.3 ? 'bad' : ''">{{ r.tie.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 A 位时判对的比例 (只算分出胜负的)</span><b>{{ r.accA2.toFixed(3) }}</b></div>
        <div class="kv"><span>准确率: 好的回答在 B 位时判对的比例 (只算分出胜负的)</span><b :class="Math.abs(r.accA2 - r.accB2) < 0.05 ? 'good' : 'bad'">{{ r.accB2.toFixed(3) }}</b></div>
      </template>
      <div class="kv"><span>真实 A 更好的比例 (数据本身, 和裁判无关)</span><b>{{ r.aBetter.toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>★ 前端随机数生成器与 Python numpy 不同, 实时模拟读数会与离线脚本存在细微差异。m21 固定实测基准 (bias 取 0.8, noise 取 0.5, 不随滑杆变化):</p>
        <ul class="pts">
          <li><b>单次判:</b> 选 A 的比例 0.700, 准确率 好的在 A 0.994 / 在 B 0.593。</li>
          <li><b>交换判:</b> 准确率 0.993 / 0.991, 平局 0.407。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const N = 2000, SHOW = 400
const mode = ref('single'), bias = ref(0.8), noise = ref(0.5), seed = ref(1), svg = ref(null)
const { start } = useDrag()

// 质量和噪声只随 seed 变; 拖 bias / noise 时点不乱跳
const base = computed(() => {
  const rand = mulberry32(seed.value * 7919 + 21)
  return range(N).map(() => ({ qa: randn(rand), qb: randn(rand), e1: randn(rand), e2: randn(rand), jit: rand() }))
})
// rule_judge: 质量差 + bias + 噪声 > 0 → 选第一个
const judge = (qFirst, qSecond, e) => qFirst - qSecond + bias.value + noise.value * e > 0

const sim = computed(() => base.value.map((d) => {
  const pickA = judge(d.qa, d.qb, d.e1)       // A 在前
  const pickB = judge(d.qb, d.qa, d.e2)       // ★ 交换: B 在前
  const v = pickA && !pickB ? 1 : !pickA && pickB ? -1 : 0
  return { ...d, aBetter: d.qa > d.qb, pickA, v }
}))
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0)
const r = computed(() => {
  const s = sim.value, A = s.filter((d) => d.aBetter), B = s.filter((d) => !d.aBetter)
  const dec = (xs) => xs.filter((d) => d.v !== 0)
  return {
    aBetter: A.length / N,
    pickA: mean(s.map((d) => +d.pickA)),
    accA1: mean(A.map((d) => +d.pickA)), accB1: mean(B.map((d) => +!d.pickA)),
    tie: mean(s.map((d) => +(d.v === 0))),
    accA2: mean(dec(A).map((d) => +(d.v === 1))), accB2: mean(dec(B).map((d) => +(d.v === -1))),
  }
})

const W = 560, H = 200, X0 = 14
const px = (v) => X0 + ((clamp(v, -3.5, 3.5) + 3.5) / 7) * (W - 2 * X0)
const pxInv = (x) => ((x - X0) / (W - 2 * X0)) * 7 - 3.5
const shown = computed(() => sim.value.slice(0, SHOW).map((d) => {
  const correct = mode.value === 'single' ? d.pickA === d.aBetter : d.v === (d.aBetter ? 1 : -1)
  const cls = mode.value === 'swap' && d.v === 0 ? 'tie' : correct ? 'ok' : 'bad'
  return { diff: d.qa - d.qb, y: 24 + d.jit * (H - 52), cls }
}))
</script>

<style scoped>
.zero { stroke: var(--border-strong); }
.bound { stroke: var(--warn); stroke-width: 2; stroke-dasharray: 5 4; }
.handle { fill: var(--warn); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.hl { font-size: 10px; fill: var(--warn); font-family: "SF Mono", Menlo, monospace; }
.ok { fill: var(--left); }
.bad { fill: var(--danger); }
.tie { fill: var(--text-dim); opacity: 0.6; }
</style>
