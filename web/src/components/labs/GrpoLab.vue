<!--
  GRPO 入门实验台 (对应 llm_finetune/methods/grpo.py 的 group_advantages)。
  只讲一件事: 优势 Â 是组内相对位置。全组奖励一起涨, Â 不动; 只动一条, 全组的 Â 都变。
  每个 token 分到多少梯度、Dr.GRPO / DAPO 改了什么, 由 FtGrpoAdvLab 讲。
  std 用无偏估计 (除以 G−1), 与 torch.std 默认值一致。
-->
<template>
  <LabFrame
    title="GRPO 入门 — 组内排名替代 critic"
    sub="同一个 prompt 采 $G$ 条回复, 奖励减组均值、除组标准差就是优势 $\hat{A}$, 不需要任何 value 网络。
      上图是奖励, 拖柱顶的横条改某一条的分数; 下图是算出来的优势。
      这里只看「奖励 → 优势」这一步。每个 token 分到多少梯度, 见「GRPO 进阶」一章的实验台。"
    module="llm_finetune/methods/grpo.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '先猜: 点「全组 +0.1」把每条奖励一起抬高, 下面的优势柱怎么变? 再只拖一条奖励, 有几条回复的优势跟着变? 最后点「全组同分」。',
      answer: '全组 +0.1: 优势一根都不动。$r - \\mu$ 把整体平移消掉了, 判分器宽松还是严格不影响梯度。\n只拖一条: 全组的优势都变, 因为 $\\mu$ 和 $\\sigma$ 是全组共用的。一条回复好不好, 要看同组的其他回复。\n全组同分: $\\hat{A}$ 全为 0, 这组采样没有梯度。训练脚本实测 grpo 有 32.1% 的 prompt 属于这种情况。\n这就是省掉 critic 的代价: baseline 只来自这 $G$ 条采样, 组里分不出高低就学不到东西。',
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
      <text x="40" y="12" class="t">奖励 r (0–1)</text>
      <line x1="40" y1="130" x2="630" y2="130" class="axis" />
      <line x1="40" :y1="yR(st.mu)" x2="630" :y2="yR(st.mu)" class="mean" />
      <text x="630" :y="yR(st.mu) - 3" text-anchor="end" class="t warn">组均值 μ = {{ st.mu.toFixed(3) }}</text>
      <text x="40" y="162" class="t">优势 Â = (r − μ) / σ</text>
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
      <div class="kv"><span><Tex text="组均值 $\mu$" /></span><b>{{ st.mu.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="组标准差 $\sigma$" /></span><b :class="{ bad: st.sd === 0 }">{{ st.sd.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="$\max\hat{A}$ / $\min\hat{A}$" /></span><b>{{ st.max.toFixed(2) }} / {{ st.min.toFixed(2) }}</b></div>
      <div class="kv"><span>被推高 / 被压低的回复</span><b :class="{ bad: st.sd === 0 }">{{ st.up }} / {{ st.down }}</b></div>
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
        <p v-if="st.sd === 0"><Tex text="组内奖励全相同: $\hat{A}$ 全为 0, 这 $G$ 次采样对梯度没有贡献。" /></p>
        <p v-else>正优势的回复 → 提高概率; 负优势 → 压低。把「比组里平均好」变成监督信号, 这是 R1 训练配方的主干。</p>
        <p><Tex text="目标里还有一项 $\beta\cdot\mathrm{KL}$ 惩罚: KL 把策略锚在参考模型附近, $\beta$ 越大越保守。它不参与优势的计算。" /></p>
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
