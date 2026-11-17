<!--
  Agent 评测: pass@k vs pass^k (对应 llm_agent/m13_evals)。
  只讲一件事: 同一个单次成功率 p, "k 次里至少成一次"和"k 次全成"随 k 走向相反的两端。
-->
<template>
  <LabFrame
    title="pass@k 与 pass^k — 「能做到」和「每次都做到」是两个指标"
    sub="- 模拟什么: 评测一个有随机性的 agent。同一个任务跑一次成功的概率记作 $p$ (单次成功率)。给它 $k$ 次机会, 可以问两个不同的问题 (两条曲线都从 k=1 时的 p 出发, 却往相反的方向走):
        - pass@k: 「k 次里至少成功一次」的概率, 说的是能力上限 (有验证器、可以挑最好的一次时看它)。
        - pass^k: 「k 次全部成功」的概率, 说的是可靠性 (每次都要一次做对、没人复核时看它)。
      - 图上是什么: 上面是曲线图, 下面是抽样格子:
        - 坐标轴: 横轴是尝试次数 k (1 到 16), 纵轴是概率。
        - 两条曲线: 绿线是 pass@k = 1 − (1 − p)^k, 随 k 上升; 红线是 pass^k = p^k, 随 k 下降。
        - 虚线和圆点: 竖直虚线是当前选的 k, 线上两个圆点是当前的两个值。左端 k=1 处的大圆点是 p 的拖拽手柄。
        - 下面两块格子: 按 p 实际抽样出来的 40 个任务。每一行一个任务, 每一格一次尝试, 绿格成功、红格失败; 行尾的 @ 亮起表示这一行至少成功一次 (算进 pass@k), ^ 亮起表示这一行全部成功 (算进 pass^k)。
      - 可以做什么: 三种操作:
        - 改 p: 拖「单次成功率 p」滑杆, 或直接拖图上左端的圆点。
        - 改 k: 拖「尝试次数 k」滑杆, 或点曲线图任意位置。
        - 「换一组抽样」: 重新抽 40 个任务的随机数; 拖 p 时格子只会单调地翻面, 不会乱跳。
      - 应该看到: 现象和右侧读数:
        - p = 0.9、k = 1: 两个指标都是 90%。
        - 把 k 拖到 8: pass@8 接近 100%, pass^8 只剩 43%。
        - 右侧前两项: 公式算出的理论值, 颜色和 p 比, 高于 p 绿、低于 p 红。
        - 「40 个任务实际数出来的 @ / ^ 比例」是下面格子里数出来的两个比例 (有 @ 的行数 / 40, 有 ^ 的行数 / 40), 和理论值接近但有抽样误差。
        - 「要 pass^k ≥ 90%, 单次 p 至少」是反过来算: 想让 k 次全对的概率不低于 90%, 单次成功率至少要多高, k=8 时是 98.7%。"
    module="llm_agent/m13"
    run="python -m llm_agent.m13_evals.demo"
    :challenge="{
      ask: 'p = 0.9 听起来很可靠。默认 k = 1, 两个指标都是 90%。先猜: 把 k 拖到 8, pass@k 和 pass^k 各变成多少? 一个每天被调用 8 次的客服 agent, 用户感受到的是哪一个?',
      answer: 'pass@8 $= 1 - 0.1^8 \\approx 100\\%$, pass^8 $= 0.9^8 \\approx 43\\%$。\n- pass@k 回答的是「多给几次机会, 它做不做得到」: 8 次里只要有一次成功就算, 8 次全失败的概率是 0.1 的 8 次方, 几乎为 0。它适合有验证器、可以重试挑最优的场景, 比如写代码然后跑测试, 错了再来。\n- pass^k 回答的是「每一次都做对的概率」: 8 次每次 90%, 连乘下来只剩 43%。动作不可撤销、没人复核的 agent (退款、发邮件、改数据库), 用户感受到的就是它: 一天用 8 次, 大约一半的日子里至少会遇到一次失败。\n- 要把 pass^8 提到 90%, 单次 $p$ 得到 98.7% (把 0.9 开 8 次方根)。k 越大, 这个门槛越接近 100%。',
    }"
  >
    <template #controls>
      <LabSlider v-model="p" label="单次成功率 p" :min="0.05" :max="0.99" :step="0.01" :format="(v) => (v * 100).toFixed(0) + '%'" />
      <LabSlider v-model="k" label="尝试次数 k" :min="1" :max="KMAX" />
      <div class="row"><button type="button" @click="seed++">换一组抽样</button></div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="pass@k 与 pass^k 曲线, 圆点可拖动改 p" @click="pickK">
      <g class="grid">
        <line v-for="y in [0, 0.25, 0.5, 0.75, 1]" :key="y" :x1="X0" :x2="W - 10" :y1="py(y)" :y2="py(y)" />
        <text v-for="y in [0, 0.5, 1]" :key="'t' + y" :x="X0 - 6" :y="py(y) + 4" text-anchor="end">{{ y * 100 }}%</text>
        <text v-for="i in [1, 4, 8, 12, 16]" :key="'k' + i" :x="px(i)" :y="H - 6" text-anchor="middle">k={{ i }}</text>
      </g>
      <line class="cursor" :x1="px(k)" :x2="px(k)" :y1="py(1)" :y2="py(0)" />
      <polyline class="at" :points="curve((i) => 1 - (1 - p) ** i)" />
      <polyline class="hat" :points="curve((i) => p ** i)" />
      <circle class="dot at" :cx="px(k)" :cy="py(passAt)" r="4" />
      <circle class="dot hat" :cx="px(k)" :cy="py(passHat)" r="4" />
      <text class="lbl at" :x="W - 12" :y="py(1) + 14" text-anchor="end">pass@k = 1 − (1 − p)ᵏ</text>
      <text class="lbl hat" :x="W - 12" :y="py(0) - 8" text-anchor="end">pass^k = pᵏ</text>
      <circle
        class="handle draggable" :cx="px(1)" :cy="py(p)" r="8" tabindex="0" role="slider" aria-label="单次成功率 p"
        :aria-valuenow="p" aria-valuemin="0.05" aria-valuemax="0.99"
        @pointerdown="start($event, { svg, onMove: ({ y }) => (p = clamp(Math.round((1 - (y - Y0) / (H - Y0 - YB)) * 100) / 100, 0.05, 0.99)) })"
        @click.stop @keydown.up.prevent="p = clamp(+(p + 0.01).toFixed(2), 0.05, 0.99)" @keydown.down.prevent="p = clamp(+(p - 0.01).toFixed(2), 0.05, 0.99)"
      />
    </svg>

    <div class="trial-wrap">
      <div v-for="half in [0, 1]" :key="half" class="cells trials" :style="{ gridTemplateColumns: `repeat(${k}, 12px) 16px 16px` }">
        <template v-for="(row, t) in trials.slice(half * TASKS / 2, (half + 1) * TASKS / 2)" :key="t">
          <span v-for="(ok, a) in row" :key="a" class="cell" :class="ok ? 'ok' : 'bad'" />
          <span class="tag-at mono" :class="{ yes: row.some(Boolean) }">@</span>
          <span class="tag-hat mono" :class="{ yes: row.every(Boolean) }">^</span>
        </template>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>pass@{{ k }} ({{ k }} 次里至少成功一次)</span><b :class="vsP(passAt)">{{ pct(passAt) }}</b></div>
      <div class="kv"><span>pass^{{ k }} ({{ k }} 次全部成功)</span><b :class="vsP(passHat)">{{ pct(passHat) }}</b></div>
      <div class="kv"><span>{{ TASKS }} 个任务实际数出来的 @ / ^ 比例</span><b>{{ pct(emp.at) }} / {{ pct(emp.hat) }}</b></div>
      <div class="kv"><span>要 pass^{{ k }} ≥ 90%, 单次 p 至少</span><b>{{ pct(0.9 ** (1 / k)) }}</b></div>
      <p class="lab-note">前两项的颜色是和单次成功率 p 比: 高于 p 变绿, 低于 p 变红。k = 1 时两个指标都等于 p。</p>
      <p class="lab-note">这里的「成功」只看结果。真实评测里只看最终答案还不够: 轨迹检查 (trajectory check) 会另外检查过程, 比如有没有调用不该调的工具、有没有跳过必须的确认步骤。答案对、过程违规, 同样算失败。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, range } from '@/utils/labmath.js'

const KMAX = 16, TASKS = 40
const W = 560, H = 230, X0 = 44, Y0 = 12, YB = 24
const p = ref(0.9), k = ref(1), seed = ref(1) // k = 1 是对照: 两个指标重合, 拖 k 才分开
const svg = ref(null)
const { start } = useDrag()

const px = (i) => X0 + ((i - 1) / (KMAX - 1)) * (W - X0 - 16)
const py = (v) => Y0 + (1 - v) * (H - Y0 - YB)
const curve = (f) => range(KMAX).map((i) => `${px(i + 1)},${py(f(i + 1))}`).join(' ')
const pct = (v) => (v * 100).toFixed(1) + '%'

// ★ 全部的数学: 独立同分布的 k 次尝试
const passAt = computed(() => 1 - (1 - p.value) ** k.value)
const passHat = computed(() => p.value ** k.value)
const vsP = (v) => (v > p.value + 1e-9 ? 'good' : v < p.value - 1e-9 ? 'bad' : '')

// 抽样: 每个格子先抽一个固定的 u, 成功 = u < p。拖 p 时格子只会单调地翻面, 不会乱跳
const us = computed(() => { const rand = mulberry32(seed.value * 7919); return range(TASKS).map(() => range(KMAX).map(() => rand())) })
const trials = computed(() => us.value.map((row) => row.slice(0, k.value).map((u) => u < p.value)))
const emp = computed(() => ({
  at: trials.value.filter((r) => r.some(Boolean)).length / TASKS,
  hat: trials.value.filter((r) => r.every(Boolean)).length / TASKS,
}))

const pickK = (e) => {
  const pt = svg.value.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY
  const x = pt.matrixTransform(svg.value.getScreenCTM().inverse()).x
  k.value = clamp(Math.round(((x - X0) / (W - X0 - 16)) * (KMAX - 1)) + 1, 1, KMAX)
}
</script>

<style scoped>
svg { width: 100%; cursor: crosshair; }
.grid line { stroke: var(--border); stroke-width: 1; }
.grid text { font-size: 10px; fill: var(--text-dim); }
polyline { fill: none; stroke-width: 2; }
.at { stroke: var(--left); } .hat { stroke: var(--danger); }
.dot.at { fill: var(--left); } .dot.hat { fill: var(--danger); }
.lbl { font-size: 11px; stroke: none; } .lbl.at { fill: var(--left); } .lbl.hat { fill: var(--danger); }
.cursor { stroke: var(--accent); stroke-dasharray: 3 3; }
.handle { fill: var(--accent); stroke: var(--bg-card); stroke-width: 2; }
.trial-wrap { display: flex; flex-wrap: wrap; gap: 8px 28px; margin-top: 12px; }
.trials { gap: 2px; grid-auto-rows: 10px; align-content: start; }
.trials .cell { min-width: 0; width: 12px; height: 10px; border-radius: 2px; }
.tag-at, .tag-hat { font-size: 9px; line-height: 10px; color: var(--text-dim); opacity: 0.3; padding-left: 4px; }
.tag-at.yes { color: var(--left); opacity: 1; } .tag-hat.yes { color: var(--danger); opacity: 1; }
</style>
