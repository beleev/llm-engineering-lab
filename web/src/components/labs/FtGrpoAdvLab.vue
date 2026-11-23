<!--
  GRPO / Dr.GRPO / DAPO 的优势与 token 权重实验台。
  只讲一件事: 同一组 (奖励, 长度), 三种算法给每个 token 分到的梯度权重不一样 ——
  GRPO 的 1/|o_i| 让"又长又错"的回答每个 token 挨的罚更轻 (长度偏置), 除以 σ 让太易/太难的题被放大;
  Dr.GRPO 两个都去掉; DAPO 改成 token 级平均, 并在全对/全错时直接丢弃该组 (动态采样)。
  std 用无偏估计 (N−1), 与 torch.std 默认一致。
-->
<template>
  <LabFrame
    title="GRPO、Dr.GRPO、DAPO — 同一组回答, 每个 token 分到多少梯度"
    sub="- 模拟什么: GRPO 针对同一 prompt 采样 G 条候选回复, 由可验证奖励函数给出二元打分 (正确得 1 分, 错误得 0 分), 计算组内相对优势 $\hat{A}$ (用单条奖励减去组均值 $\mu$, 标准 GRPO 进一步除以组标准差 $\sigma$), 再将序列级优势 $\hat{A}$ 分配至该回复的每个 token 上; 从奖励到优势的映射基础可参见「RM · GRPO · 蒸馏」章节的 GRPO 入门实验台。三种变体的核心差异集中于将优势分摊至各 token 时的归一化分母:
  - GRPO: 除以当前回复自身的序列长度 $|o_i|$。
  - Dr.GRPO: 除以预设常数最大长度 $L_{\max}$, 同时省去优势计算中除以组标准差 $\sigma$ 的步骤。
  - DAPO: 汇总全组所有 token 后按组内平均长度归一化 (即 token 级平均), 并在整组全对或全错时丢弃该组并重新采样 (动态采样)。
- 图上是什么: 主视图包含 6 行, 分别对应同一组内的 6 条候选回复:
  - 左侧状态圆圈: 标示验证判分结果 (绿色 ✓ 为正确, 红色 ✗ 为错误)。
  - 横条宽度: 表示该条回复包含的 token 数量。
  - 横条着色浓度: 映射该回复内单个 token 分配到的梯度权重幅值, 绿色代表正向推高, 红色代表负向压低, 颜色越深单步权重越大。
  - 右侧两列数值: Â 为该回复的组内优势值, 「每 token 权重 ‰」等于 Â 除以长度归一化分母后再乘以 1000。
- 可以做什么: 按界面顺序可执行以下交互:
  - 点击顶部三个按钮: 在 GRPO、Dr.GRPO 与 DAPO 三种算法间切换。
  - 点击「全对 / 全错」按钮: 将组内 6 条回复统一设为全部正确或全部错误。
  - 点击「换一组」按钮: 随机重新生成一组回复长度与对错结果。
  - 点击任意行左侧圆圈: 单独翻转该条回复的对错标签。
  - 拖动横条右端把手 (或聚焦后使用方向键): 在 20 至 400 token 范围内调整该条回复的长度。
- 应该看到: 在标准 GRPO 下拉长一条错误回复时, 其红色显著变浅, 单个 token 承受的负向惩罚随之被稀释; 切换至 Dr.GRPO 后拉长回复颜色保持不变。右侧汇总读数含义如下:
  - 「组均值 μ / 标准差 σ」: 当前 6 条回复奖励分数的组内统计量。
  - 「有梯度的回答」: 组内优势 Â 非零的样本条数, 在全对或全错情形下降为 0。
  - 最长与最短错误回复的单 token 惩罚比值 (对应读数「最长错 : 最短错」): 等于 1 时显示为绿色, 代表无长度偏置; 小于 1 时显示为红色, 代表错误回复越长单 token 惩罚越轻。
  - 「本组总更新量」: 组内 6 条回复的优势绝对值 |Â| 之和。"
    module="llm_finetune/methods/grpo.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '保持默认的 2 对 4 错配置, 按顺序完成以下操作:\n1. 在 GRPO 模式下将第 3 条 (初始长度为 60 token 的错误回复) 拖长至 400 token: 该回复内单个 token 承受的惩罚权重变为原来的几分之几?\n2. 切换至 Dr.GRPO 模式后重复上述拖拽, 观察单 token 权重是否变化。\n3. 点击「全对」按钮: 三种算法各自会如何处理这组样本?',
      answer: '结论:\n1. 在 GRPO 下将回复拖长至 400 token 后, 单 token 惩罚权重降至原来的 15% ($60/400$)。\n2. 在 Dr.GRPO 下单 token 惩罚权重保持不变。\n3. 整组全对时三种算法算出的优势 Â 均为 0; GRPO 与 Dr.GRPO 仍将该零梯度组计入当前 batch, 而 DAPO 会将其丢弃并重新采样。\n机制解析:\n- GRPO 的长度偏置来源: 标准 GRPO 目标函数采用 $\\tfrac{1}{G}\\sum_i \\tfrac{1}{|o_i|}\\sum_t \\hat{A}_i$, 即先在每条回复内部除以各自长度 $|o_i|$ 求平均, 再对组内 G 条样本取平均。当错误回复从 60 token 延长至 400 token 时, 单 token 分配到的负向惩罚缩减为 $60/400 = 15\\%$。这种分母设计变相鼓励模型在答错时拉长篇幅以摊薄逐 token 惩罚, 同时也偏好更短的正确回复, 构成了 R1 类强化学习训练中输出长度持续膨胀的非推理性诱因之一。\n- Dr.GRPO 的两项修正:\n  1. 将样本自身长度倒数 $1/|o_i|$ 替换为统一常数 $1/L_{\\max}$, 消除不同回复长度间的权重偏差。\n  2. 移除优势归一化分母中的组标准差 $\\sigma$; 在二元 0/1 奖励下, 「近乎全对」或「近乎全错」的极端题目组标准差 $\\sigma$ 最小, 除以 $\\sigma$ 会不当放大过易或过难样本的梯度权重。\n- 全对或全错退化组的处置差异: 当组内得分完全一致时, 每条样本均满足 $r - \\mu = 0$, 三种算法在该组上的梯度贡献严格为零, 对应 6 次在线采样的推理开销无法转化为参数更新信号。后续批处理策略存在分歧:\n  - GRPO 与 Dr.GRPO: 照常将该零梯度组纳入当前训练 batch 求均值, 稀释整个批次的有效梯度幅值。\n  - DAPO: 启用动态采样过滤机制直接剔除该组并补充采样新题目, 确保进入更新步的每个组均同时包含正负样本。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.name }}</button>
      </div>
      <div class="row">
        <button type="button" @click="setAll(1)">全对</button>
        <button type="button" @click="setAll(0)">全错</button>
        <button type="button" @click="reseed">换一组</button>
      </div>
    </template>

    <svg ref="svgEl" :viewBox="`0 0 640 ${G * ROW + 30}`" role="group" aria-label="6 条采样回答的奖励与长度">
      <text x="62" y="14" class="t">长度 |o| (token)</text>
      <text x="486" y="14" class="t">Â</text>
      <text x="540" y="14" class="t">每 token 权重 ‰</text>
      <g v-for="(o, i) in rows" :key="i" :transform="`translate(0, ${22 + i * ROW})`">
        <!-- 点击切换对错 -->
        <g class="tog" role="button" tabindex="0" :aria-label="`回答 ${i + 1}: ${o.r ? '正确' : '错误'}, 点击切换`" @click="flip(i)" @keydown.enter="flip(i)" @keydown.space.prevent="flip(i)">
          <circle cx="30" cy="14" r="12" :fill="o.r ? 'var(--left)' : 'var(--danger)'" />
          <text x="30" y="18.5" text-anchor="middle" class="sym">{{ o.r ? '✓' : '✗' }}</text>
        </g>
        <rect x="60" y="3" :width="o.len" height="22" rx="3" :fill="o.w >= 0 ? 'var(--left)' : 'var(--danger)'" :opacity="clamp(Math.abs(o.w) / WREF, 0.06, 1)" />
        <rect x="60" y="3" :width="o.len" height="22" rx="3" fill="none" stroke="var(--border-strong)" />
        <text x="66" y="18" class="t">{{ o.len }}</text>
        <!-- ★ 拖右端 = 改这条回答的长度 -->
        <rect
          class="draggable" :x="60 + o.len - 5" y="0" width="10" height="28" rx="3" fill="var(--text)"
          tabindex="0" role="slider" :aria-label="`回答 ${i + 1} 的长度`" :aria-valuenow="o.len" aria-valuemin="20" :aria-valuemax="LMAX"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setLen(i, x - 60) })"
          @keydown.right.prevent="setLen(i, o.len + 20)" @keydown.left.prevent="setLen(i, o.len - 20)"
        />
        <text x="486" y="18" class="v" :fill="o.adv > 0 ? 'var(--left)' : o.adv < 0 ? 'var(--danger)' : 'var(--text-dim)'">{{ fmt(o.adv, 2) }}</text>
        <text x="548" y="18" class="v" :fill="o.w > 0 ? 'var(--left)' : o.w < 0 ? 'var(--danger)' : 'var(--text-dim)'">{{ fmt(o.w * 1000, 2) }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span><Tex text="6 个奖励的组均值 $\mu$ / 标准差 $\sigma$" /></span><b>{{ st.mu.toFixed(2) }} / {{ st.sd.toFixed(2) }}</b></div>
      <div class="kv"><span>有梯度的回答 (Â 非零)</span><b :class="st.live ? 'good' : 'bad'">{{ st.live }} / {{ G }}</b></div>
      <div class="kv"><span>最长错 : 最短错 (每 token 罚的比值, 1 = 无长度偏置)</span><b :class="st.ratio === null ? '' : st.ratio > 0.99 ? 'good' : 'bad'">{{ st.ratio === null ? '—' : st.ratio.toFixed(2) }}</b></div>
      <div class="kv"><span><Tex text="本组总更新量 $\sum|\hat{A}|$" /></span><b>{{ st.total.toFixed(2) }}</b></div>
      <p class="lab-note">
        <template v-if="!st.live">
          <Tex text="当组内奖励完全相同时, 每条回复均满足 $r - \mu = 0$, 优势估计 $\hat{A}$ 全部归零, 该组 6 次采样对当前步参数梯度的贡献严格为零。" />
          {{ mode === 'dapo' ? 'DAPO 动态采样: 丢弃该组, 继续采到有对有错为止。' : '它仍占着 batch 的名额, 有效 batch 变小、梯度噪声变大。' }}
        </template>
        <template v-else><Tex :text="MODES.find((m) => m.id === mode).note" /></template>
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, range, sum } from '@/utils/labmath.js'

const G = 6, ROW = 36, LMAX = 400, WREF = 0.006
const MODES = [
  { id: 'grpo', name: 'GRPO', note: '$\\hat{A} = (r-\\mu)/\\sigma$, 每 token 权重 $= \\hat{A}/|o_i|$: 先在每条回答内部按长度平均。越长, 每个 token 分到的越少。' },
  { id: 'drgrpo', name: 'Dr.GRPO (去 σ, 去长度偏置)', note: '$\\hat{A} = r-\\mu$, 每 token 权重 $= \\hat{A}/L_{\\max}$ (常数 400): 长短回答的 token 一视同仁, 难度不同的题也不再被 $1/\\sigma$ 重新加权。' },
  { id: 'dapo', name: 'DAPO (token 级 loss)', note: '$\\hat{A} = (r-\\mu)/\\sigma$, 每 token 权重 $= \\hat{A}/(\\text{组内平均长度})$: 全组 token 先汇总再平均, 每个 token 一票, 长回答因此整体更有分量。' },
]
const mode = ref('grpo')
const seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

const sample = (s) => {
  if (s === 1) return [120, 300, 60, 200, 360, 100].map((len, i) => ({ r: +(i < 2), len }))   // 默认: 2 对 4 错
  const rand = mulberry32(s * 2741)
  return range(G).map(() => ({ r: +(rand() < 0.45), len: 40 + Math.round(rand() * 28) * 10 }))
}
const outs = ref(sample(1))
const reseed = () => { seed.value++; outs.value = sample(seed.value) }
const flip = (i) => { outs.value = outs.value.map((o, k) => (k === i ? { ...o, r: 1 - o.r } : o)) }
const setAll = (r) => { outs.value = outs.value.map((o) => ({ ...o, r })) }
const setLen = (i, len) => { outs.value = outs.value.map((o, k) => (k === i ? { ...o, len: clamp(Math.round(len / 10) * 10, 20, LMAX) } : o)) }

const st0 = computed(() => {
  const rs = outs.value.map((o) => o.r), mu = sum(rs) / G
  return { mu, sd: Math.sqrt(sum(rs.map((r) => (r - mu) ** 2)) / (G - 1)) }   // 无偏 std, 同 torch.std
})
const rows = computed(() => {
  const { mu, sd } = st0.value
  const meanLen = sum(outs.value.map((o) => o.len)) / G
  return outs.value.map((o) => {
    const adv = mode.value === 'drgrpo' ? o.r - mu : (o.r - mu) / (sd + 1e-4)
    // ★ 三种算法的全部区别: Â 除不除 σ, 以及 token 权重的分母
    const w = adv / (mode.value === 'grpo' ? o.len : mode.value === 'drgrpo' ? LMAX : meanLen)
    return { ...o, adv, w }
  })
})
const st = computed(() => {
  const wrong = rows.value.filter((o) => !o.r && o.adv !== 0).sort((a, b) => a.len - b.len)
  return {
    ...st0.value,
    live: rows.value.filter((o) => Math.abs(o.adv) > 1e-9).length,
    ratio: wrong.length > 1 && wrong[0].len !== wrong.at(-1).len ? wrong.at(-1).w / wrong[0].w : null,
    total: sum(rows.value.map((o) => Math.abs(o.adv))),
  }
})
const fmt = (v, n) => (v > 0 ? '+' : '') + v.toFixed(n)
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.v { font-size: 13px; font-family: "SF Mono", Menlo, monospace; }
.sym { font-size: 13px; fill: var(--bg-card); font-weight: 700; pointer-events: none; }
.tog { cursor: pointer; }
.tog:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
