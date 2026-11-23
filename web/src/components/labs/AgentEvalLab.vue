<!--
  Agent 评测: pass@k vs pass^k (对应 llm_agent/m13_evals)。
  只讲一件事: 同一个单次成功率 p, "k 次里至少成一次"和"k 次全成"随 k 走向相反的两端。
-->
<template>
  <LabFrame
    title="pass@k 与 pass^k — 「能做到」和「每次都做到」是两个指标"
    sub="- 模拟什么: 具有采样随机性的 Agent 在多次运行下的两类核心评测指标。设同一任务单次独立运行成功的概率为 $p$, 赋予系统 $k$ 次尝试机会时, 评测可回答两个截然不同的工程问题 (两条曲线均从 $k=1$ 时的概率 $p$ 出发, 随 $k$ 增大向相反方向发散):
  - pass@k: 衡量「$k$ 次尝试中至少成功一次」的概率, 反映系统的能力上限 (适用于配备自动验证器、可从多次采样中择优采纳的场景)。
  - pass^k: 衡量「连续 $k$ 次尝试全部成功」的概率, 反映系统的执行可靠性 (适用于无人工复核、每次线上调用均须一次做对的生产场景)。
- 图上是什么: 上方为理论概率曲线图, 下方为蒙特卡洛抽样矩阵:
  - 坐标轴与双曲线: 横轴为尝试次数 $k$ (范围 1 至 16), 纵轴为成功概率; 绿色上升曲线代表 pass@k ($1 - (1 - p)^k$), 红色下降曲线代表 pass^k ($p^k$)。
  - 游标虚线与交互圆点: 竖直虚线标示当前选定的 $k$ 值, 与两条曲线的交点圆点对应当前两项理论读数; 左端 $k=1$ 处的大圆点可直接上下拖动以调节 $p$。
  - 底部抽样矩阵: 按当前 $p$ 对 40 个独立任务各模拟 $k$ 次尝试。每行代表一个任务, 每格代表一次尝试 (绿格成功、红格失败); 行尾「@」亮起表示该行至少成功一次 (计入 pass@k), 「^」亮起表示该行全部成功 (计入 pass^k)。
- 可以做什么: 提供三项交互控制:
  - 调节单次成功率: 拖动「单次成功率 p」滑杆, 或直接上下拖拽曲线图左端 $k=1$ 处的手柄圆点。
  - 调节尝试次数: 拖动「尝试次数 k」滑杆, 或直接点击曲线图内任意横坐标位置。
  - 点击「换一组抽样」: 重新生成 40 个任务的底层均匀随机数; 在固定随机种子下拖动 $p$ 时, 抽样矩阵中的格子仅按阈值单调翻转而不会无序闪烁。
- 应该看到: 观察双曲线的发散规律与右侧读数:
  - 当 $p = 0.9$、$k = 1$ 时: 两项指标均等于 90%。
  - 将 $k$ 拖至 8: pass@8 趋近于 100%, 而 pass^8 跌至仅剩 43%。
  - 右侧前两项理论值: 颜色相对单次成功率 $p$ 编码, 高于 $p$ 标绿、低于 $p$ 标红。
  - 「40 个任务实际数出来的 @ / ^ 比例」: 统计下方矩阵中点亮「@」与「^」的行数占 40 的比例, 围绕理论值上下波动并直观体现有限样本的抽样误差。
  - 「要 pass^k ≥ 90%, 单次 p 至少」: 反向求解当要求连续 $k$ 次全对的概率不低于 90% 时所需的最低单次成功率, 在 $k=8$ 时高达 98.7%。"
    module="llm_agent/m13"
    run="python -m llm_agent.m13_evals.demo"
    :challenge="{
      ask: '单次成功率 $p = 0.9$ 看似已经相当可靠: 在默认 $k = 1$ 时, 两项指标均为 90%。请先估算: 将尝试次数 $k$ 调至 8 后, pass@k 与 pass^k 分别变为多少? 对于一个每天被线上调用 8 次的生产客服 Agent, 业务方实际承受的体验对应哪一项指标?',
      answer: '当 $k = 8$ 时, pass@8 为 $1 - 0.1^8 \\approx 100\\%$, 而 pass^8 仅为 $0.9^8 \\approx 43\\%$; 每天被调用 8 次的线上客服 Agent 实际对应的是 pass^8。\n- pass@k 衡量试错上限: 只要 8 次采样中有任意 1 次通过即可采纳, 连续 8 次全部失败的概率为 $0.1^8$, 几乎为 0; 该指标适用于拥有单元测试或编译器等确定性验证器、允许反复生成择优的离线编程等场景。\n- pass^k 衡量连续无故障率: 8 次独立请求每次成功率为 90%, 连乘后全天零失误的概率仅剩 43%。对于执行退款、发送外部邮件或修改生产数据库等不可逆动作且无人工复核的客服 Agent, 用户直接感知的正是 pass^k——每天调用 8 次, 超过一半的工作日里至少会发生一次线上事故。\n- 可靠性门槛的指数级攀升: 若要求全天 8 次调用全部成功的概率 pass^8 达到 90%, 单次成功率 $p$ 必须提升至 $0.9^{1/8} \\approx 98.7\\%$ (即对 0.9 开 8 次方根); 调用频次 $k$ 越大, 对单次 $p$ 的要求越逼近 100%。',
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
      <p class="lab-note">右侧前两项理论读数的颜色均以单次成功率 $p$ 为基准: 高于 $p$ 时标绿, 低于 $p$ 时标红; 当 $k = 1$ 时两项指标均恰好等于 $p$。</p>
      <p class="lab-note">本实验台将每次尝试简化为二元成败事件; 在真实 Agent 评测中仅核对最终结果并不充分, 还需配合轨迹检查 (trajectory check) 审计中间执行过程 (例如是否调用了未授权工具、是否跳过了强制确认环节), 即使最终答案正确但过程违规也应判定为失败。</p>
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
