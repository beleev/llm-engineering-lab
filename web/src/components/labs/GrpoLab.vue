<!--
  GRPO 入门实验台 (对应 llm_finetune/methods/grpo.py 的 group_advantages)。
  只讲一件事: 优势 Â 是组内相对位置。全组奖励一起涨, Â 不动; 只动一条, 全组的 Â 都变。
  每个 token 分到多少梯度、Dr.GRPO / DAPO 改了什么, 由 FtGrpoAdvLab 讲。
  std 用无偏估计 (除以 G−1), 与 torch.std 默认值一致。
-->
<template>
  <LabFrame
    title="GRPO — 同一道题采 G 条回复互相比较, 顶替 PPO 的 critic"
    sub="- 模拟什么: 在强化学习微调中, 策略模型针对同一 prompt 采样 $G$ 条候选回复, 每条回复由规则判分器或奖励模型给出标量奖励 $r$ (取值范围 0 至 1)。要将绝对奖励转化为策略梯度信号, 需引入代表当前题目平均得分水平的基线 (baseline): 得分高于基线的回复增大生成概率, 低于基线的压低概率。本实验聚焦从奖励到优势的组内转换过程; 至于优势如何分配至各个 token, 详见「GRPO 进阶」一章的实验台。基线估计主要有两种路线:
  - PPO: 额外维护并训练一个与策略模型同规模的 critic 价值网络来拟合基线。
  - GRPO: 直接取当前组内 $G$ 条回复的奖励均值 $\mu$ 作为基线, 并除以组内标准差 $\sigma$, 计算每条回复的归一化优势 $\hat{A} = (r - \mu)/\sigma$。
- 图上是什么: 视图由上下两张联动柱状图组成:
  - 上图: 每根柱代表一条采样回复的奖励 $r$, 柱顶横条可上下拖动; 黄色虚线标出当前组内均值 $\mu$。
  - 下图: 对应的组内归一化优势 $\hat{A}$。绿色向上代表正优势 (增大该回复概率), 红色向下代表负优势 (压低该回复概率), 柱端数字显示具体的 $\hat{A}$ 数值。
- 可以做什么: 支持以下交互调节:
  - 拖动任意柱顶横条: 单独修改该条回复的奖励分值 (聚焦后按键盘上下方向键每次增减 0.05)。
  - 点击「重新采样」: 随机刷新整组奖励分值。
  - 点击「全组 +0.1 / −0.1」: 将组内所有回复的奖励同步上调或下调 0.1。
  - 点击「全组同分」: 将所有回复的奖励统一设为 0.5。
  - 拖动「组大小 G」(范围 4 至 16): 改变单组采样条数, 已有柱的分值保持不变, 仅增减尾部样本。
- 应该看到: 单独拖动某一条回复的奖励时, 全组所有回复的优势柱都会随之重算, 因为均值 $\mu$ 与标准差 $\sigma$ 由全组共同决定; 点击「全组 +0.1」时优势柱纹丝不动; 点击「全组同分」时所有优势全部归 0。右侧读数与对比表含义如下:
  - 「组均值 μ」与「组标准差 σ」: 当前采样组的统计量。
  - 「max Â / min Â」: 组内最高分与最低分回复对应的优势值。
  - 「被推高 / 被压低的回复」: 获得正优势与负优势的回复条数。
  - 下方对比表: 并列对比 PPO 与 GRPO 的模型驻留开销, GRPO 省去了 critic 网络, 训练期显存中少驻留一份大模型。"
    module="llm_finetune/methods/grpo.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '依次操作并观察以下现象:\n1. 点击「全组 +0.1」将所有回复的奖励同步上调, 下方的优势柱如何变化?\n2. 仅拖动其中一条回复的奖励, 组内共有几条回复的优势随之改变?\n3. 点击「全组同分」后, 各条回复的优势变为多少?',
      answer: '答案:\n1. 全组同步 +0.1 时, 所有优势保持不变。\n2. 仅拖动一条奖励时, 全组 $G$ 条回复的优势全部随之改变。\n3. 全组同分时, 所有回复的优势均变为 0。\n原因解析:\n- 平移不变性: 在 $\\hat{A} = (r - \\mu)/\\sigma$ 中, 所有奖励同步增加 0.1 会使组均值 $\\mu$ 同步增加 0.1, 差值 $r - \\mu$ 与标准差 $\\sigma$ 均保持不变。因此奖励模型的整体打分偏松或偏紧不会影响梯度方向与幅值, 决定更新信号的仅是组内相对排序与分差。\n- 组内耦合性: 由于 $\\mu$ 与 $\\sigma$ 由全组样本联合统计, 修改单条回复的分数会同时改变基线与缩放尺度, 进而联动改变全组每条回复的优势值。\n- 零方差退化: 当组内全部同分时, $r - \\mu$ 恒为 0, 所得优势 $\\hat{A}$ 全为 0, 整组采样对策略梯度无任何贡献 (代码实现中在分母加上 1e-4 以防除零溢出)。本仓库训练脚本实测显示, 在 60 步 GRPO 训练中平均有 32.1% 的 prompt 出现全对或全错的同分退化, 空占 batch 算力却不提供梯度信号。\n省去 critic 网络的代价在于基线完全依赖当次采样的 $G$ 条回复, 一旦组内无法拉开分差便无法产生有效学习信号。DAPO 提出的动态采样机制正是针对这一退化, 直接过滤全对与全错的 prompt 并补充采样至填满有效 batch。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="reseed">重新采样</button>
        <button type="button" :disabled="hi + 0.1 > 1" @click="shift(0.1)">全组 +0.1</button>
        <button type="button" :disabled="lo - 0.1 < 0" @click="shift(-0.1)">全组 −0.1</button>
        <button type="button" @click="flat">全组同分</button>
      </div>
      <LabSlider v-model="G" label="组大小 G" :min="4" :max="16" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 300" role="group" aria-label="一组回复的奖励与优势">
      <text x="40" y="12" class="t">每条回复的奖励 r (0–1), 拖柱顶横条可改</text>
      <line x1="40" y1="130" x2="630" y2="130" class="axis" />
      <line x1="40" :y1="yR(st.mu)" x2="630" :y2="yR(st.mu)" class="mean" />
      <text x="630" :y="yR(st.mu) - 3" text-anchor="end" class="t warn">组均值 μ = {{ st.mu.toFixed(3) }}</text>
      <text x="40" y="162" class="t">优势 Â = (r − μ) / σ: 绿色推高, 红色压低</text>
      <line x1="40" y1="230" x2="630" y2="230" class="axis" />
      <g v-for="(o, i) in rows" :key="i">
        <rect :x="o.x" :y="yR(o.r)" :width="bw" :height="130 - yR(o.r)" fill="var(--eye)" opacity="0.7" />
        <!-- ★ 拖柱顶 = 改这条回复的奖励 -->
        <rect
          class="draggable" :x="o.x - 2" :y="yR(o.r) - 4" :width="bw + 4" height="8" rx="2" fill="var(--text)"
          tabindex="0" role="slider" :aria-label="`回复 ${i + 1} 的奖励`" :aria-valuenow="o.r.toFixed(2)" aria-valuemin="0" aria-valuemax="1"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ y }) => setR(i, (130 - y) / 110) })"
          @keydown.up.prevent="setR(i, o.r + 0.05)" @keydown.down.prevent="setR(i, o.r - 0.05)"
        />
        <rect :x="o.x" :y="o.a >= 0 ? 230 - o.h : 230" :width="bw" :height="o.h" :fill="o.a >= 0 ? 'var(--left)' : 'var(--danger)'" />
        <text :x="o.x + bw / 2" :y="o.a >= 0 ? 224 - o.h : 242 + o.h" text-anchor="middle" class="t">{{ o.a.toFixed(1) }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span><Tex text="组均值 $\mu$ (当基准用, 顶替 critic)" /></span><b>{{ st.mu.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="组标准差 $\sigma$ (0 = 全组同分, 没梯度)" /></span><b :class="{ bad: st.sd === 0 }">{{ st.sd.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="$\max\hat{A}$ / $\min\hat{A}$ (最好 / 最差那条的优势)" /></span><b>{{ st.max.toFixed(2) }} / {{ st.min.toFixed(2) }}</b></div>
      <div class="kv"><span>被推高 / 被压低的回复条数</span><b :class="{ bad: st.sd === 0 }">{{ st.up }} / {{ st.down }}</b></div>
      <table class="cmp">
        <thead>
          <tr><th></th><th>PPO</th><th>GRPO</th></tr>
        </thead>
        <tbody>
          <tr><td>需要 critic</td><td>是</td><td>否</td></tr>
          <tr><td>同时驻留模型</td><td>4 个 (policy/ref/RM/critic)</td><td>2-3 个</td></tr>
          <tr><td>baseline 来源</td><td>value 网络预测</td><td>组内均值</td></tr>
          <tr><td>奖励来源</td><td>reward model</td><td>规则验证 (RLVR) 或 RM</td></tr>
        </tbody>
      </table>
      <div class="lab-note">
        <p v-if="st.sd === 0"><Tex text="当组内奖励全部相同时, 优势 $\hat{A}$ 全为 0, 该组 $G$ 次采样对策略梯度无任何贡献, 仅消耗 batch 采样与前向开销。" /></p>
        <p v-else>获得正优势的回复在更新中被推高生成概率, 负优势回复则被压低概率。以组内相对均值的高低直接构造强化学习监督信号, 既无需人工偏好标注也无需额外训练 critic 网络, 构成了 DeepSeek-R1 强化学习阶段的核心算法骨架。</p>
        <p><Tex text="完整训练目标还包含一项 $\beta\cdot\mathrm{KL}$ 正则惩罚: KL 散度度量当前策略偏离参考模型 (通常取训练初始权重) 的分布距离, 用于约束策略更新幅度, 系数 $\beta$ 越大策略越保守。由于该项独立于组内优势计算之外, 本实验台未将其计入柱状图。" /></p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, range, sum } from '@/utils/labmath.js'

const GMAX = 16, AMAX = 4, X0 = 40, W = 590
const G = ref(8)
const seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

// 一次抽满 16 条, G 只决定用前几条: 拖 G 时已有的柱子不会跳
const sample = (s) => {
  const rand = mulberry32(s * 977)
  return range(GMAX).map(() => clamp(0.45 + (rand() - 0.5) * 0.7, 0, 1))
}
const all = ref(sample(1))
const rewards = computed(() => all.value.slice(0, G.value))
const hi = computed(() => Math.max(...rewards.value))
const lo = computed(() => Math.min(...rewards.value))

const reseed = () => { seed.value++; all.value = sample(seed.value) }
const setR = (i, r) => { all.value = all.value.map((v, k) => (k === i ? clamp(Math.round(r * 100) / 100, 0, 1) : v)) }
const shift = (d) => { all.value = all.value.map((v) => clamp(v + d, 0, 1)) }
const flat = () => { all.value = all.value.map(() => 0.5) }

const st = computed(() => {
  const rs = rewards.value, mu = sum(rs) / rs.length
  const sd = Math.sqrt(sum(rs.map((r) => (r - mu) ** 2)) / (rs.length - 1))
  const adv = rs.map((r) => (r - mu) / (sd + 1e-4))   // ★ 与 group_advantages 同一个式子
  return {
    mu, sd, adv, max: Math.max(...adv), min: Math.min(...adv),
    up: adv.filter((a) => a > 1e-9).length, down: adv.filter((a) => a < -1e-9).length,
  }
})

const slot = computed(() => W / G.value)
const bw = computed(() => slot.value * 0.6)
const yR = (r) => 130 - r * 110
// 优势柱用固定刻度 (±4 占 60px): G=16 时单条最大优势是 15/√16 = 3.75
const rows = computed(() => rewards.value.map((r, i) => {
  const a = st.value.adv[i]
  return { r, a, x: X0 + i * slot.value + slot.value * 0.2, h: (Math.min(Math.abs(a), AMAX) / AMAX) * 60 }
}))
</script>

<style scoped>
.t { font-size: 10px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.t.warn { fill: var(--warn); }
.axis { stroke: var(--border-strong); stroke-width: 1; }
.mean { stroke: var(--warn); stroke-width: 1; stroke-dasharray: 4 3; }
.draggable:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.cmp { width: 100%; border-collapse: collapse; font-size: 11px; }
.cmp th, .cmp td { border: 1px solid var(--border); padding: 4px 6px; text-align: left; color: var(--text-muted); }
.cmp th { color: var(--text); background: var(--bg-elev); }
</style>
