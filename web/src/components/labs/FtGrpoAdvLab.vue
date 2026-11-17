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
    sub="- 模拟什么: GRPO 对同一个 prompt 采 G 条回答, 用可验证奖励打分 (对 1 错 0), 再把奖励变成每条的优势 $\hat{A}$ (组内优势: 这条的奖励减组均值 $\mu$, GRPO 还要除以组标准差 $\sigma$), 最后把 $\hat{A}$ 摊到这条回答的每个 token 上。奖励怎么变成优势, 见「RM · GRPO · 蒸馏」一章的 GRPO 入门实验台。三种算法的区别全在「摊」的分母:
        - GRPO: 除以这条自己的长度 $|o_i|$。
        - Dr.GRPO: 除以常数 $L_{\max}$ (而且不除 $\sigma$)。
        - DAPO: 把全组 token 汇总后除以组内平均长度 (token 级平均), 并在全对或全错时丢弃整组重采 (动态采样)。
      - 图上是什么: 6 行, 每行一条回答。各部分:
        - 左边的圆圈: 对错 (绿 ✓ 红 ✗)。
        - 横条的长度: 回答的 token 数。
        - 横条的颜色深浅: 这条回答里每个 token 分到的梯度权重, 绿是推高, 红是压低, 越浓越大。
        - 右侧两列数字: Â 是这条的优势, 「每 token 权重 ‰」是 Â 除以分母后乘 1000。
      - 可以做什么: 按界面顺序:
        - 顶部三个按钮: 切换算法。
        - 「全对 / 全错」: 把 6 条设成同一种结果。
        - 「换一组」: 随机抽。
        - 点某一行的圆圈: 翻转对错。
        - 拖横条右端的把手 (或聚焦后按方向键): 改这条的长度 (20 到 400)。
      - 应该看到: GRPO 下把一条错误回答拖长, 它的红色明显变淡, 每个 token 挨的罚变轻了; Dr.GRPO 下拖长不变色。右侧读数:
        - 「组均值 μ / 标准差 σ」: 这 6 个奖励的统计量。
        - 「有梯度的回答」: Â 非零的条数, 全对或全错时为 0。
        - 最长和最短两条错误回答每 token 罚的比值 (读数「最长错 : 最短错」): 等于 1 (绿) 说明没有长度偏置, 小于 1 (红) 说明越长罚越轻。
        - 「本组总更新量」: 6 个 |Â| 之和。"
    module="llm_finetune/methods/grpo.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '保持默认的 2 对 4 错, 按顺序做:\n1. 在 GRPO 下把第 3 条 (60 token 的错误回答) 拖到 400 token: 它每个 token 挨的罚变成原来的几分之几?\n2. 切到 Dr.GRPO 再拖一次。\n3. 点「全对」: 三种算法各自会发生什么?',
      answer: '答案:\n1. GRPO 下拖到 400 后每 token 的罚只剩原来的 15% (60/400)。\n2. Dr.GRPO 下不变。\n3. 全对时三种算法的 Â 全是 0, GRPO 和 Dr.GRPO 让这组白占 batch, DAPO 丢掉它重采。\n为什么:\n- GRPO 的 loss 是 $\\tfrac{1}{G}\\sum_i \\tfrac{1}{|o_i|}\\sum_t \\hat{A}_i$, 这个式子在说: 先在每条回答内部按长度平均, 再在组内平均。错误回答从 60 拖到 400 token, 每 token 的惩罚变成 60/400 = 15%。所以错误回答越长, 每个 token 受的罚越轻, 训练会把回复往长里推, 这是 R1 类训练里回复越来越长的一部分原因, 不全是「学会了思考」。\n- Dr.GRPO 改了两处:\n  1. $1/|o_i|$ 换成常数 $1/L_{\\max}$, 长短一视同仁。\n  2. 去掉除以 $\\sigma$, 因为 0/1 奖励下 $\\sigma$ 在「几乎全对」或「几乎全错」的题上最小, 除以它等于给太易、太难的题加权。\n- 全对时每条的 $r - \\mu = 0$, 三种算法的梯度都是零, 这 6 次采样的算力白花。之后的处理不同:\n  - GRPO 和 Dr.GRPO: 照样把它算进 batch, 稀释有效梯度。\n  - DAPO: 动态采样丢掉这组继续采, 直到 batch 里每组都有对有错。',
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
          <Tex text="组内奖励全相同, 每条的 $r - \mu$ 都是 0, $\hat{A}$ 全为 0, 这 6 次采样对梯度零贡献。" />
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
