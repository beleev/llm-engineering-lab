<!--
  注意力掩码实验台 (窗口 + sink 对应 llm_infer/m16_attention_sinks)。
  只讲一件事: 同一套 QKV 投影, 换一张 mask 就是另一个模型。
  矩阵的行是 query (在看的位置), 列是 key (被看的位置)。
-->
<template>
  <LabFrame
    title="注意力掩码实验台 — 每个位置能看见哪些位置"
    sub="- 模拟什么: 自注意力机制中的掩码 (mask) 结构与计算复杂度边界。在注意力层中, 当前位置 token (作为 query) 会对前序位置 token (作为 key) 计算相关性打分并按权重聚合信息。注意力掩码规定了哪些 (query, key) 位置对允许参与打分: 被禁止的位置对在进入 softmax 前直接填充为 $-\infty$, 归一化后注意力权重严格等于 0。自回归语言模型最基础的约束是因果 (causal) 掩码, 即每个位置仅允许回看历史及自身、严禁窥视未来位置。在此基础上, 不同架构通过裁剪历史可见范围衍生出四类典型掩码模式:
  - 全因果掩码: 如标准 LLaMA 架构, 保留全部历史位置。
  - 带状滑动窗口: 如 Mistral 采用的滑动窗口注意力 (SWA), 仅回看最近固定长度的局部上下文。
  - 带状窗口加初始汇聚点 (sink): 如 StreamingLLM (对应源码 llm_infer/m16_attention_sinks), 在局部窗口之外永久保留序列最开头的若干 token。
  - top-k 动态稀疏: 如 DeepSeek 的动态稀疏注意力 (DSA), 每个 query 仅从历史中筛选得分最高的 $k$ 个 key 参与计算。
- 图上是什么: 一张维度为 $T \times T$ 的二维注意力可见性网格, 第 $i$ 行对应第 $i$ 个 query 位置, 第 $j$ 列对应第 $j$ 个 key 位置, 颜色图例位于矩阵下方:
  - 着色方格: 当前掩码策略下允许该 query 访问的可见 key 位置。
  - 右上三角空白区: 尚未生成的未来时刻位置, 被自回归因果约束统一屏蔽。
  - 浅色方格: 虽属于历史时刻、但已被当前窗口或稀疏策略裁剪淘汰的位置。
  - sink 模式专属着色列: 序列起始处被永久保留的注意力汇聚点 token。
  - 细线外框: 标示当前鼠标或键盘聚焦选中的特定 query 行。
- 可以做什么: 按界面从上到下依次调节参数:
  - 顶部一排按钮: 在 4 种注意力掩码机制之间切换。
  - 拖动「序列长度 T」: 改变上下文总长度与矩阵阶数。
  - 在滑动窗口与 sink 模式下拖动「窗口 W」: 调整每个 query 允许回看的最近历史 token 数量。
  - 在 sink 模式下拖动「sink 数 S」: 调整序列开头永久保留的汇聚点 token 数量。
  - 在 top-k 模式下拖动「稀疏候选 top-k」: 调整每个 query 最多保留的高分历史 key 数量。
  - 将鼠标悬停或点击矩阵任意一行 (键盘操作可聚焦矩阵后按上下方向键): 选中特定 query 位置并在右侧查看其实时可见的 key 数量。
- 应该看到: 拉长序列长度 $T$ 时, 全因果掩码的着色面积按 $T$ 的平方膨胀, 而滑动窗口仅随 $T$ 呈线性增长。右侧各项读数含义如下:
  - 「当前机制」: 简明概括当前激活的掩码规则。
  - 「可见格子数」: 整张矩阵中允许参与打分的着色方格总数, 直接正比于注意力核心打分与加权的浮点运算量 (FLOPs)。
  - 「相对全因果的计算量」: 当前可见方格数占全因果下三角总数 $T(T+1)/2$ 的百分比。
  - 「KV cache 条目上限」: 解码生成阶段每层需为单条序列常驻保留的 K 与 V 状态条目数。全因果模式随序列长度线性增长 ($O(T)$), 滑动窗口模式恒定封顶于 $W$ ($O(W)$), sink 模式恒定为 $S + W$; 而 top-k 动态稀疏模式虽然省去了大量注意力计算, 但由于未来新 query 可能重新选中任意早期历史位置, 其 KV cache 仍必须全量保留 ($O(T)$)。"
    module="llm_infer/m16"
    run="python -m llm_infer.m16_attention_sinks.demo"
    :challenge="{
      ask: '固定滑动窗口大小 $W = 6$。先预测: 当序列长度 $T$ 从 24 调大至 48 时, 可见格子数会变为原来的多少倍? KV cache 条目上限如何变化? 随后切换至全因果模式重复上述操作进行对比。',
      answer: '滑动窗口模式下可见格子数约为原来的 2 倍, 全因果模式下约为原来的 4 倍; KV cache 条目上限在滑动窗口下保持不变, 在全因果下则随之翻倍。\n- 滑动窗口 ($W = 6$): 可见格子数从 129 增至 273, 约为原来的 2.1 倍, 随 $T$ 呈线性增长 (除序列起始不足窗口长度的几行外, 每个 query 最多回看 6 个 key, 行数翻倍时总格子数近似翻倍)。KV cache 条目上限始终恒定为 6, 解码生成阶段只需在显存中滚动保留最近 6 个 token 的 K 与 V。\n- 全因果模式: 可见格子数从 300 增至 1176, 约为原来的 3.9 倍, 随 $T^2$ 呈平方级增长 (第 $i$ 行包含 $i + 1$ 个可见位置, 累加总和为 $T(T+1)/2$)。KV cache 条目上限则从 24 增至 48, 随 $T$ 线性扩张。\n局部滑动窗口将自注意力计算复杂度从 $O(T^2)$ 降至 $O(T \\cdot W)$, 并将推理期 KV cache 显存占用从 $O(T)$ 压缩至常数级 $O(W)$。其代价在于单层注意力无法直接读取跨出窗口的历史位置, 早期远距离信息只能依赖前序网络层逐层聚合至窗口内近邻 token 的隐状态表示中向后接力传递: 堆叠 $L$ 层网络时的理论有效感受野上限约为 $L \\times W$。',
    }"
  >
    <template #controls>
      <div class="row" role="group" aria-label="选择注意力掩码模式">
        <button
          v-for="m in modes" :key="m.id" type="button"
          :class="{ active: mode === m.id }" :aria-pressed="mode === m.id" @click="mode = m.id"
        >{{ m.label }}</button>
      </div>
      <LabSlider v-model="T" label="序列长度 T (矩阵的行数和列数)" :min="8" :max="48" :step="4" />
      <LabSlider v-if="mode === 'swa' || mode === 'sink'" v-model="W" label="窗口 W (每个 query 能回看最近几个)" :min="2" :max="16" />
      <LabSlider v-if="mode === 'sink'" v-model="S" label="sink 数 S (开头永久保留几个)" :min="1" :max="4" />
      <LabSlider v-if="mode === 'topk'" v-model="K" label="稀疏候选 top-k (每个 query 最多挑几个 key)" :min="2" :max="16" />
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
      <li><span class="swatch visible" aria-hidden="true"></span>这个 query 能看见的 key</li>
      <li v-if="mode === 'sink'"><span class="swatch sink" aria-hidden="true"></span>永久保留的 sink (开头几个 token)</li>
      <li><span class="swatch evicted" aria-hidden="true"></span>被当前策略裁掉的历史</li>
      <li><span class="swatch future" aria-hidden="true"></span>未来 token (因果屏蔽, 永远看不见)</li>
    </ul>

    <template #stats>
      <div aria-live="polite" aria-atomic="true" class="live">
        <div class="mode-summary">
          <span>当前机制</span>
          <strong>{{ currentModeLabel }}</strong>
          <p>{{ modeSummary }}</p>
        </div>
        <div class="kv"><span>可见格子数 (正比于注意力计算量 FLOPs)</span><b>{{ visibleCount }}</b></div>
        <div class="kv"><span>相对全因果的计算量</span><b :class="{ good: pctOfFull < 100 }">{{ pctOfFull }}%</b></div>
        <div class="kv"><span>生成时每层要存的 KV cache 条目上限</span><b :class="{ good: kv.n < T }">{{ kv.n }} = {{ kv.o }}</b></div>
        <div class="kv"><span>query {{ q }} 看得见的 key 数 / 它和之前的位置数</span><b>{{ qSeen }} / {{ q + 1 }}</b></div>
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
  sink: `最近 ${Math.min(T.value, W.value)} 个位置负责局部信息, 开头 ${Math.min(T.value, S.value)} 个 sink 一直保留, 接住多余的注意力。`,
  topk: `全部 ${T.value} 个 KV 仍被保存, 但每个 query 只选择最多 ${Math.min(T.value, K.value)} 个位置参与注意力计算。`,
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
