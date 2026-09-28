<!--
  注意力掩码实验台 (窗口 + sink 对应 llm_infer/m16_attention_sinks)。
  只讲一件事: 同一套 QKV 投影, 换一张 mask 就是另一个模型。
  矩阵的行是 query (在看的位置), 列是 key (被看的位置)。
-->
<template>
  <LabFrame
    title="注意力掩码实验台 — 谁能看见谁"
    sub="同一套 QKV 投影, 换一张 mask 就是另一个模型: 全因果 = LLaMA, 带状 = Mistral SWA,
      带状+sink = StreamingLLM / GPT-OSS, top-k 稀疏 = DeepSeek DSA。
      每一行是一个 query, 每一列是一个 key。把鼠标移到某一行 (或聚焦矩阵后按上下键), 看这个 query 能看见谁。"
    module="llm_infer/m16"
    run="python -m llm_infer.m16_attention_sinks.demo"
    :challenge="{
      ask: '滑动窗口 W = 6。先猜: T 从 24 拖到 48, 可见格子变成几倍? KV cache 条目呢? 切到全因果再拖一次。',
      answer: '- 滑动窗口: 可见格子 129 → 273, 约 2.1 倍, 随 T 线性增长。KV 条目一直是 6。\n- 全因果: 可见格子 300 → 1176, 约 3.9 倍, 随 $T^2$ 增长。KV 条目 24 → 48。\n窗口把计算从 $O(T^2)$ 降到 $O(T \\cdot W)$, 把显存从 $O(T)$ 降到 $O(W)$。代价是窗口外的信息只能跨层接力传递。',
    }"
  >
    <template #controls>
      <div class="row" role="group" aria-label="选择注意力掩码模式">
        <button
          v-for="m in modes" :key="m.id" type="button"
          :class="{ active: mode === m.id }" :aria-pressed="mode === m.id" @click="mode = m.id"
        >{{ m.label }}</button>
      </div>
      <LabSlider v-model="T" label="序列长度 T" :min="8" :max="48" :step="4" />
      <LabSlider v-if="mode === 'swa' || mode === 'sink'" v-model="W" label="窗口 W" :min="2" :max="16" />
      <LabSlider v-if="mode === 'sink'" v-model="S" label="sink 数 S" :min="1" :max="4" />
      <LabSlider v-if="mode === 'topk'" v-model="K" label="稀疏候选 top-k" :min="2" :max="16" />
    </template>

    <svg :viewBox="`0 0 ${size + M} ${size + M}`" class="mask-grid" role="group" :aria-label="maskAriaLabel">
      <text :x="M" :y="M * 0.6" class="ax" :font-size="fs">列 = key j (被看的位置) →  0 … {{ T - 1 }}</text>
      <text :transform="`translate(${M * 0.4}, ${M}) rotate(90)`" class="ax" :font-size="fs">行 = query i (在看的位置) →  0 … {{ T - 1 }}</text>
      <g v-for="(row, i) in grid" :key="i" @mouseenter="q = i" @click="q = i">
        <rect
          v-for="(c, j) in row" :key="j"
          :x="M + j * cell" :y="M + i * cell" :width="cell - 0.6" :height="cell - 0.6" :class="'cell ' + c"
        />
      </g>
      <rect
        class="qrow" :x="M - 1" :y="M + q * cell - 0.8" :width="size + 1" :height="cell + 1"
        tabindex="0" role="slider" aria-label="选中的 query 行" :aria-valuenow="q" aria-valuemin="0" :aria-valuemax="T - 1"
        @keydown.up.prevent="q = Math.max(0, q - 1)" @keydown.down.prevent="q = Math.min(T - 1, q + 1)"
      />
    </svg>
    <ul class="mask-legend" aria-label="矩阵颜色图例">
      <li><span class="swatch visible" aria-hidden="true"></span>当前 query 可见</li>
      <li v-if="mode === 'sink'"><span class="swatch sink" aria-hidden="true"></span>永久保留的 sink</li>
      <li><span class="swatch evicted" aria-hidden="true"></span>被策略裁掉</li>
      <li><span class="swatch future" aria-hidden="true"></span>未来 token (因果屏蔽)</li>
    </ul>

    <template #stats>
      <div aria-live="polite" aria-atomic="true" class="live">
        <div class="mode-summary">
          <span>当前机制</span>
          <strong>{{ currentModeLabel }}</strong>
          <p>{{ modeSummary }}</p>
        </div>
        <div class="kv"><span>可见格子 (∝ 注意力 FLOPs)</span><b>{{ visibleCount }}</b></div>
        <div class="kv"><span>相对全因果的计算量</span><b :class="{ good: pctOfFull < 100 }">{{ pctOfFull }}%</b></div>
        <div class="kv"><span>推理 KV cache 条目上限</span><b :class="{ good: kv.n < T }">{{ kv.n }} = {{ kv.o }}</b></div>
        <div class="kv"><span>query {{ q }} 看得见的 key</span><b>{{ qSeen }} / {{ q + 1 }}</b></div>
        <p class="lab-note"><Tex :text="modeNote" /></p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { range, sum } from '@/utils/labmath.js'

const modes = [
  { id: 'full', label: '全因果 (LLaMA)' },
  { id: 'swa', label: '滑动窗口 (Mistral)' },
  { id: 'sink', label: '窗口+sink (StreamingLLM)' },
  { id: 'topk', label: 'top-k 稀疏 (DSA)' },
]

const mode = ref('swa')
const T = ref(24)
const W = ref(6)
const S = ref(2)
const K = ref(6)
const q = ref(23)          // 选中的 query 行, 默认最后一行
const cell = 12
watch(T, (t) => { q.value = t - 1 })

// 画布随 T 缩放, 边距和字号按比例取, 渲染出来的字大小基本不变
const size = computed(() => T.value * cell)
const M = computed(() => size.value * 0.07)
const fs = computed(() => size.value * 0.03)

// top-k 稀疏: 确定性伪随机挑历史位置 (模拟 indexer 选中的 token)
const hash = (i, j) => {
  const h = (i * 2654435761 + j * 40503) % 2147483647
  return (h ^ (h >> 7)) % 997
}

// ★ 整张 mask 一次算完: grid[i][j] 是 query i 对 key j 的状态
const grid = computed(() => range(T.value).map((i) => {
  // topk: 保留对角附近 2 个 + 按 hash 选出的 k-2 个
  const keep = mode.value === 'topk'
    ? new Set(range(Math.max(0, i - 1)).map((p) => [hash(i, p), p]).sort((a, b) => a[0] - b[0]).slice(0, Math.max(0, K.value - 2)).map(([, p]) => p))
    : null
  return range(T.value).map((j) => {
    if (j > i) return 'future'                       // 因果性永远成立
    if (mode.value === 'full') return 'visible'
    if (mode.value === 'topk') return i - j <= 1 || keep.has(j) ? 'visible' : 'evicted'
    if (j > i - W.value) return 'visible'
    return mode.value === 'sink' && j < S.value ? 'sink' : 'evicted'
  })
}))

const seen = (row) => row.filter((c) => c === 'visible' || c === 'sink').length
const visibleCount = computed(() => sum(grid.value.map(seen)))
const pctOfFull = computed(() => Math.round((visibleCount.value / ((T.value * (T.value + 1)) / 2)) * 100))
const qSeen = computed(() => seen(grid.value[q.value] ?? []))

const kv = computed(() => ({
  full: { n: T.value, o: 'O(T)' },
  swa: { n: Math.min(T.value, W.value), o: 'O(W)' },
  sink: { n: Math.min(T.value, W.value + S.value), o: 'O(S+W)' },
  topk: { n: T.value, o: 'O(T)' },
}[mode.value]))

const modeNote = computed(() => ({
  full: '每个位置看全部历史。计算 $O(T^2)$、KV cache $O(T)$, 是长上下文的双重瓶颈。',
  swa: '只看最近 $W$ 个。信息跨层接力, $L$ 层感受野 $\\approx L\\cdot W$。代码: llm_models/.../mistral.py',
  sink: '窗口滑动 + 永远保留开头 $S$ 个"注意力下水道"。代码: llm_infer/m16_attention_sinks',
  topk: 'DSA: cache 全保留 ($O(T)$ 显存), 但每步只对 indexer 选出的 $k$ 个算注意力: 省计算, 不省显存。',
}[mode.value]))

const currentModeLabel = computed(() => modes.find((item) => item.id === mode.value)?.label || mode.value)

const modeSummary = computed(() => ({
  full: `第 ${T.value} 个 token 可以回看全部 ${T.value} 个历史位置; 上下文越长, 计算与 KV 都持续增长。`,
  swa: `每个 query 只保留最近 ${Math.min(T.value, W.value)} 个位置; 旧信息必须跨层逐步传递。`,
  sink: `最近 ${Math.min(T.value, W.value)} 个位置负责局部信息, 开头 ${Math.min(T.value, S.value)} 个 sink 负责稳定全局锚点。`,
  topk: `全部 ${T.value} 个 KV 仍被保存, 但每个 query 只选择最多 ${Math.min(T.value, K.value)} 个位置参与重计算。`,
}[mode.value]))

const maskAriaLabel = computed(() =>
  `${currentModeLabel.value} 的 ${T.value}×${T.value} 因果掩码, 行是 query, 列是 key。${visibleCount.value} 个历史关系可见, 占全因果计算的 ${pctOfFull.value}%。${modeSummary.value}`
)
</script>

<style scoped>
.mask-grid { width: 100%; max-width: 440px; border-radius: var(--radius-sm); background: var(--code-bg); padding: 4px; }
.ax { fill: var(--text-muted); }
.qrow { fill: none; stroke: var(--text); stroke-width: 0.8; pointer-events: none; }
.qrow:focus-visible { outline: none; stroke: var(--accent); stroke-width: 1.6; }
.mask-legend { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 8px; list-style: none; color: var(--text-muted); font-size: 10.5px; }
.mask-legend li { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; border: 1px solid var(--border-strong); }
.swatch.visible { background: var(--accent); }
.swatch.sink { background: var(--eye); }
.swatch.evicted { background: var(--border); }
.swatch.future { background: transparent; }

.cell.visible { fill: var(--accent); }
.cell.sink { fill: var(--eye); }
.cell.evicted { fill: var(--border); }
.cell.future { fill: transparent; }

.live { display: flex; flex-direction: column; gap: 10px; }
.mode-summary { padding: 11px 13px; border: 1px solid color-mix(in srgb, var(--accent) 38%, var(--border)); border-radius: var(--radius-sm); background: var(--accent-soft); }
.mode-summary span { display: block; color: var(--accent); font-size: 9.5px; font-weight: 700; letter-spacing: 0.7px; }
.mode-summary strong { display: block; margin-top: 2px; font-size: 13px; }
.mode-summary p { margin-top: 4px; color: var(--text-muted); font-size: 11.5px; line-height: 1.65; text-wrap: pretty; }
</style>
