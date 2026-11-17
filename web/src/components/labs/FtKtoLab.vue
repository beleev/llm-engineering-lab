<!--
  KTO 实验台。公式与 llm_finetune/methods/kto.py:KTOLoss 相同 (β=0.5, z0 = 0 —— 本例 z0 全程被 clamp 成 0)。
  只讲一件事: 单条 👍 / 👎 也能训, 但好坏不均时, 两类样本的总推力 λ·n 必须拉平。
  曲线和推力是前端实时算的; 右侧表格是 python -m llm_finetune.run_finetune.kto.train_kto 的输出, 不随控件变化。
-->
<template>
  <LabFrame
    title="KTO — 单条 👍/👎 也能训, 但好坏 1:9 时 λ 必须跟着调"
    sub="- 模拟什么: KTO 不需要成对的偏好, 每条回复只要一个「好」或「坏」的标签就能训。它给每条回复算一个隐式奖励 $r = \log(\pi/\pi_{\text{ref}})$ (policy 比 ref 给这条回复的 log 概率多了多少); 好样本的目标是把 $r$ 推到参考点 $z_0$ 之上, 坏样本是把 $r$ 压到 $z_0$ 之下, 本例 $z_0 = 0$。推力是一条 $\sigma$ 曲线的斜率, 离 $z_0$ 太远的样本斜率趋近 0, 就不再推。好坏两类各有一个权重 $\lambda_D$ (好) 和 $\lambda_U$ (坏); 一个 batch 里好坏数量不均时, 两类的总推力「权重乘条数」必须拉平, 否则多的那一类会把所有回复一起拖走。
      - 图上是什么: 从上到下:
        - 顶部 10 个格子: 一个 batch 里的样本比例, 绿 👍 是好样本, 红 👎 是坏样本。
        - 中间的曲线: 单条样本在不同隐式奖励 $r$ 处的推力大小 $|\partial v/\partial r| = \lambda\beta\sigma(1-\sigma)$。绿线是好样本 ($\lambda_D = 1$), 红线是坏样本 ($\lambda_U$ 由滑杆给), $\beta = 0.5$ 和 kto.py 一致。
        - 紫色竖线: 当前的 $r$。
        - 下面两根横条: 整批的总推力, 好样本条数乘单条推力, 坏样本条数乘单条推力。
      - 可以做什么: 按界面顺序:
        - 比例按钮: 「好:坏 = 1:1 / 1:9」一键设比例。
        - 拖「好样本条数 n_D」: 自己定比例。
        - 拖「坏样本权重 λ_U」: 改红线的高度。
        - 拖紫色竖线 (或聚焦后按方向键): 改 $r$, 看在不同位置两类的推力各是多少。
      - 应该看到: 1:9 且 $\lambda_U = 1$ 时, 坏样本那根横条是好样本的 9 倍。右侧读数下面的表格是 train_kto.py 在留出集上的实测, 不随左侧变化。读数和表格各列:
        - 「好样本 λ_D·n_D」和「坏样本 λ_U·n_U」: 两类的总权重。
        - 「两者之比」: 落在论文建议的 1 到 1.33 之间变绿。
        - 「整批净推力」: 显示现在是偏向推高、偏向压低还是大致拉平。
        - 表格「准确率」: 偏好准确率。
        - 表格「log π(chosen)」: 模型给正确回复的 log 概率 (越负越不会写)。
        - 表格「贪心 EM」: 贪心解码的完全匹配率。"
    module="llm_finetune/methods/kto.py"
    run="python -m llm_finetune.run_finetune.kto.train_kto"
    :challenge="{
      ask: '点「好:坏 = 1:9」, $\\lambda_U$ 先不动:\n1. 坏样本的总推力是好样本的几倍?\n2. 模型会怎样应对?\n3. 再拖 $\\lambda_U$, 找到两边推力相等的位置。',
      answer: '答案:\n1. 坏样本的总推力是好样本的 9 倍。\n2. 模型会把所有回复一起往下压。\n3. $\\lambda_U$ 拖到 0.11 (约 1/9) 时两边拉平。\n为什么:\n- 推力的式子 $\\lambda\\beta\\sigma(1-\\sigma)$ 在说: 同一个 $r$ 上, 每条样本的推力只差一个 $\\lambda$。$\\lambda_D = \\lambda_U = 1$ 时每条一样, 9 条坏样本的总推力就是 1 条好样本的 9 倍。\n- 坏样本的目标是把 $r = \\log(\\pi/\\pi_{\\text{ref}})$ 压低, 最省力的办法是让 policy 对什么回复都不给概率, 好样本那 1 条拦不住。实测「KTO 1:9, 不调 λ」偏好方向丢了, 贪心解码一条也答不对:\n  - log π(chosen) 从 −4.03 掉到 −30.02。\n  - 准确率 0.492。\n  - 贪心 EM 0.000。\n- 拉平的条件是 $\\lambda_D n_D \\approx \\lambda_U n_U$, 代入 $1\\times 1 = \\lambda_U\\times 9$ 得 $\\lambda_U \\approx 0.11$。实测这一档: 准确率 0.988, EM 0.254。论文建议把 $\\lambda_D n_D / (\\lambda_U n_U)$ 控制在 1 到 4/3 之间。\n- 这份 1:9 的数据 DPO 一对也凑不出, 因为每个 prompt 只出现一条回复; KTO 用单条标签就能训。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: nD === 5 }" @click="nD = 5">好:坏 = 1:1</button>
        <button type="button" :class="{ active: nD === 1 }" @click="nD = 1">好:坏 = 1:9</button>
      </div>
      <LabSlider v-model="nD" label="好样本条数 n_D" :min="1" :max="5" unit=" / 10" />
      <LabSlider v-model="lamU" label="坏样本权重 λ_U" :min="0.05" :max="1.2" :step="0.01" :format="(v) => v.toFixed(2)" />
    </template>

    <div class="cells batch" style="grid-template-columns: repeat(10, 26px);">
      <span v-for="i in 10" :key="i" class="cell" :class="i <= nD ? 'ok' : 'bad'">{{ i <= nD ? '👍' : '👎' }}</span>
    </div>

    <svg ref="svgEl" viewBox="0 0 560 170" role="group" aria-label="单条样本的推力曲线">
      <line x1="30" y1="140" x2="550" y2="140" class="axis" />
      <line :x1="RX(0)" y1="20" :x2="RX(0)" y2="140" class="axis" />
      <text :x="RX(0) + 4" y="156" class="t">r = z0 = 0</text>
      <path :d="curve(LAM_D)" fill="none" stroke="var(--left)" stroke-width="2" />
      <path :d="curve(lamU)" fill="none" stroke="var(--danger)" stroke-width="2" />
      <text x="34" y="30" class="t" style="fill: var(--left)">好: λ_D·β·σ(1−σ)  (λ_D = 1)</text>
      <text x="34" y="46" class="t" style="fill: var(--danger)">坏: λ_U·β·σ(1−σ)  (λ_U = {{ lamU.toFixed(2) }})</text>
      <!-- 拖竖线 = 改隐式奖励 r -->
      <line :x1="RX(r)" y1="16" :x2="RX(r)" y2="140" stroke="var(--accent)" stroke-width="2" />
      <rect
        class="draggable focus-inset" :x="RX(r) - 8" y="8" width="16" height="136" fill="transparent"
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
      <div class="kv"><span>好样本总权重 λ_D·n_D</span><b>{{ (LAM_D * nD).toFixed(2) }}</b></div>
      <div class="kv"><span>坏样本总权重 λ_U·n_U</span><b>{{ (lamU * nU).toFixed(2) }}</b></div>
      <div class="kv"><span>两者之比 (论文建议 1 到 1.33)</span><b :class="inBand ? 'good' : 'bad'">{{ balance.toFixed(2) }}</b></div>
      <div class="kv"><span>整批净推力</span><b :class="inBand ? 'good' : 'bad'">{{ balance > 4 / 3 ? '偏向推高' : balance < 0.99 ? '偏向压低' : '大致拉平' }}</b></div>
      <p class="tcap">train_kto.py 留出集 (实测, 不随左侧变化)</p>
      <table class="cmp mono">
        <thead><tr><th>配置</th><th>偏好准确率</th><th>log π(chosen)</th><th>贪心 EM</th></tr></thead>
        <tbody>
          <tr v-for="m in MEASURED" :key="m.name"><td>{{ m.name }}</td><td>{{ m.acc }}</td><td>{{ m.lpc }}</td><td>{{ m.em }}</td></tr>
        </tbody>
      </table>
      <p class="lab-note">
        代价: 同样的前向预算下, KTO 每步有标签的回复只有 64 条, DPO 是 128 条, 因为 KTO 每步有一半前向花在估参考点 z0 用的错配样本上 (本例 z0 被 clamp 成 0)。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const BETA = 0.5, LAM_D = 1
// train_kto.py 输出表, 原样抄录
const MEASURED = [
  { name: 'SFT 起点', acc: '0.965', lpc: '−4.03', em: '0.332' },
  { name: 'DPO 成对', acc: '0.996', lpc: '−4.25', em: '0.121' },
  { name: 'KTO 1:1', acc: '0.992', lpc: '−3.73', em: '0.324' },
  { name: 'KTO 1:9, λ_U = 1/9', acc: '0.988', lpc: '−4.12', em: '0.254' },
  { name: 'KTO 1:9, 不调 λ', acc: '0.492', lpc: '−30.02', em: '0.000' },
]
const nD = ref(5), lamU = ref(1), r = ref(0)
const svgEl = ref(null)
const { start } = useDrag()
const setR = (v) => { r.value = clamp(Math.round(v * 2) / 2, -10, 10) }

const nU = computed(() => 10 - nD.value)
// ★ 要拉平的量是 λ·n, 不是 λ: λ_D·n_D / (λ_U·n_U)
const balance = computed(() => (LAM_D * nD.value) / (lamU.value * nU.value))
const inBand = computed(() => balance.value >= 0.99 && balance.value <= 4 / 3)   // 0.99: 滑杆步长 0.01 凑不出精确的 1/9

const sig = (x) => 1 / (1 + Math.exp(-x))
// |∂v/∂r|: 好 v = λ_D σ(β(r − z0)), 坏 v = λ_U σ(β(z0 − r)); 两者导数的大小形状一样, 只差 λ
const grad = (lam, x) => { const s = sig(BETA * x); return lam * BETA * s * (1 - s) }
const RX = (x) => 290 + x * 26
const GY = (g) => 140 - g * 800
const curve = (lam) => range(81).map((i) => {
  const x = -10 + i * 0.25
  return `${i ? 'L' : 'M'} ${RX(x).toFixed(1)} ${GY(grad(lam, x)).toFixed(1)}`
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
.tcap { font-size: 11px; color: var(--text); margin-top: 4px; }
.cmp { width: 100%; border-collapse: collapse; font-size: 10px; }
.cmp th, .cmp td { border: 1px solid var(--border); padding: 3px 4px; text-align: left; color: var(--text-muted); }
.cmp th { color: var(--text); background: var(--bg-elev); }
</style>
