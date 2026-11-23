<!-- 混合模型状态快照实验台 (对应 llm_infer/m35_hybrid_state_cache): PrefixStateCache.match 的命中取整 + demo [3] 的字节账。 -->
<template>
  <LabFrame
    title="混合模型的前缀缓存 — 状态只能在快照处恢复, 快照越密越占显存"
    sub="- 模拟什么: Qwen3-Next-80B-A3B 这类混合模型 48 层里有 36 层是 Gated DeltaNet, 每条序列只有一份定长状态; 另 12 层是全注意力, 照常每个 token 存一行 KV。缓存里有一条 32,768 token 的序列, 新请求和它共享前 L 个 token。注意力层的 KV 截到前 L 行就能用, 递推层的状态却切不出第 L 位的样子, 只能从不超过 L 的最近一份快照恢复。字节数按 config.json 和 bf16 估算, 与 Python demo [3] 相同。
- 图上是什么: 上面的长条是缓存里的序列, 竖线是状态快照的位置:
  - 绿色: 可用命中, 从最近一份快照恢复。
  - 黄色: 公共前缀里落在最后一份快照之后的部分, 还得现算。
  - 灰色: 和新请求不同的部分。
  - 下面两根横条比较整段 KV 和全部快照的字节数。
- 可以做什么: 切换状态按 bf16 还是 fp32 存; 拖动「快照间隔」; 拖动长条上的圆点 (或聚焦后按左右方向键) 改变公共前缀长度 L。
- 应该看到: 间隔越大, 黄色越长, 快照横条越短。右侧依次是可用命中占公共前缀的比例 (≥ 90% 标绿)、要现算的公共 token、快照与 KV 的字节数 (快照更大时标红), 以及快照开始比 KV 小的间隔分界。"
    module="llm_infer/m35"
    run="python -m llm_infer.m35_hybrid_state_cache.demo"
    :challenge="{
      ask: '状态按 bf16 存、快照间隔 512 时, 快照的总字节是 KV 的几倍? 间隔至少调到多少, 快照才比 KV 小? 代价是什么?',
      answer: '约 3.1 倍 (2,412 MiB 对 768 MiB)。一份快照要存 36 层、每层 32 个 128×128 的状态和短卷积缓存, 约 37.7 MiB, 相当于 1608 个 token 的 KV; 间隔 512 时每 512 个 token 就多存这么一份。\n间隔要超过 1608 token, 在 2 的幂里就是 2048: 16 份快照共 603 MiB, 比 KV 的 768 MiB 小。\n代价是命中变粗: 公共前缀只能用到 2048 的整数倍, 最多有 2047 个公共 token 要重算。状态改用 fp32 时一份快照翻倍, 分界点变成 3216 token。Marconi 的做法是不按固定间隔存, 只在 radix 树的分叉点和最后一次 decode 之后存, 把快照留给真会被复用的位置。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: dtype === 2 }" @click="dtype = 2">状态 bf16</button>
        <button type="button" :class="{ active: dtype === 4 }" @click="dtype = 4">状态 fp32</button>
      </div>
      <LabSlider v-model="logI" label="快照间隔" :min="7" :max="14" :format="(v) => `${2 ** v} token`" />
      <LabSlider v-model="L" label="公共前缀 L" :min="1" :max="T - 1" :step="1" unit=" token" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} 150`" role="group" aria-label="缓存序列上的快照位置与命中">
      <text x="0" y="14" class="ax">缓存里的序列 (32,768 token), 竖线 = 状态快照</text>
      <rect x="0" y="24" :width="W" height="34" class="seg-rest" />
      <rect x="0" y="24" :width="px(hit)" height="34" class="seg-hit" />
      <rect :x="px(hit)" y="24" :width="px(L) - px(hit)" height="34" class="seg-redo" />
      <line v-for="p in ticks" :key="p" :x1="px(p)" :x2="px(p)" y1="22" y2="60" class="tick" />
      <g
        class="handle" tabindex="0" role="slider" aria-label="公共前缀长度" :aria-valuemin="1" :aria-valuemax="T - 1" :aria-valuenow="L"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (L = clampL(Math.round((x / W) * T))) })"
        @keydown.left.prevent="L = clampL(L - 256)" @keydown.right.prevent="L = clampL(L + 256)"
      >
        <line :x1="px(L)" :x2="px(L)" y1="18" y2="66" />
        <circle :cx="px(L)" cy="72" r="7" />
      </g>
      <text :x="Math.min(px(L), W - 4)" y="94" class="ax" :text-anchor="px(L) > W - 90 ? 'end' : 'middle'">L = {{ L.toLocaleString() }}</text>
      <g v-for="(b, i) in bars" :key="b.name" :transform="`translate(0, ${106 + i * 22})`">
        <text x="0" y="12" class="ax">{{ b.name }}</text>
        <rect x="150" y="2" :width="Math.max(1, (b.mib / barMax) * (W - 240))" height="12" :class="b.cls" />
        <text :x="156 + (b.mib / barMax) * (W - 240)" y="12" class="ax">{{ b.mib.toLocaleString(undefined, { maximumFractionDigits: 0 }) }} MiB</text>
      </g>
    </svg>
    <p class="legend">
      <span><i class="seg-hit" />可用命中</span><span><i class="seg-redo" />公共但要现算</span><span><i class="seg-rest" />不同的部分</span>
    </p>

    <template #stats>
      <div class="kv"><span>可用命中 h (占公共前缀)</span><b :class="hit / L >= 0.9 ? 'good' : 'bad'">{{ hit.toLocaleString() }} ({{ (hit / L * 100).toFixed(1) }}%)</b></div>
      <div class="kv"><span>公共但要现算的 token</span><b>{{ (L - hit).toLocaleString() }}</b></div>
      <div class="kv"><span>{{ nSnap }} 份快照 / 整段 KV</span><b :class="snapMiB > kvMiB ? 'bad' : 'good'">{{ snapMiB.toFixed(0) }} / {{ kvMiB.toFixed(0) }} MiB</b></div>
      <div class="kv"><span>快照比 KV 小的间隔分界</span><b>&gt; {{ breakeven.toFixed(0) }} token</b></div>
      <div class="lab-note">
        <p>一份快照 = 36 层 × (32 头 × 128 × 128 + 短卷积 3 × 8192) × {{ dtype }} B = {{ (snap / 2 ** 20).toFixed(1) }} MiB; 每 token 的 KV = 12 层 × 2 × 2 头 × 256 × 2 B = 24,576 B。</p>
        <p><Tex text="命中 $h = \lfloor \min(L, T-1) / I \rfloor \cdot I$, $I$ 是快照间隔。" /></p>
        <p>Python demo [3] 的三档间隔, 按当前的 L 和状态精度:</p>
        <ul class="pts">
          <li v-for="r in sweep" :key="r.i"><b>间隔 {{ r.i }}:</b> 快照 {{ r.mib.toFixed(0) }} MiB, 公共前缀里要现算 {{ r.redo.toLocaleString() }} token。</li>
        </ul>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import Tex from '@/components/Tex.vue'
import { clamp, range } from '@/utils/labmath.js'
import { useDrag } from '@/composables/useDrag.js'

const W = 640, T = 32768
const KV_TOK = 12 * 2 * 2 * 256 * 2                        // Qwen3-Next: 12 个注意力层, 2 个 KV 头 × 256 维, bf16
const STATE_ELEMS = 36 * (32 * 128 * 128 + 3 * 8192)       // 36 个 Gated DeltaNet 层的状态 + 短卷积缓存
const svg = ref(null)
const { start } = useDrag()
const dtype = ref(2), logI = ref(11), L = ref(20000)

const clampL = (v) => clamp(v, 1, T - 1)
const px = (tok) => (tok / T) * W
const I = computed(() => 2 ** logI.value)
const hit = computed(() => Math.floor(Math.min(L.value, T - 1) / I.value) * I.value)   // ★ 命中向下取整到快照位置
const nSnap = computed(() => Math.floor(T / I.value))
const ticks = computed(() => (nSnap.value > 128 ? [] : range(nSnap.value).map((i) => (i + 1) * I.value)))
const snap = computed(() => STATE_ELEMS * dtype.value)
const snapMiB = computed(() => (nSnap.value * snap.value) / 2 ** 20)
const kvMiB = (T * KV_TOK) / 2 ** 20
const breakeven = computed(() => snap.value / KV_TOK)
const bars = computed(() => [
  { name: '整段 KV (12 层)', mib: kvMiB, cls: 'bar-kv' },
  { name: `快照 ${nSnap.value} 份 (36 层)`, mib: snapMiB.value, cls: snapMiB.value > kvMiB ? 'bar-bad' : 'bar-ok' },
])
const barMax = computed(() => Math.max(...bars.value.map((b) => b.mib)))
const sweep = computed(() => [512, 2048, 8192].map((i) => ({
  i, mib: (Math.floor(T / i) * snap.value) / 2 ** 20, redo: L.value - Math.floor(Math.min(L.value, T - 1) / i) * i,
})))
</script>

<style scoped>
svg { min-width: 520px; }
.ax { font-size: 11px; fill: var(--text-muted); }
.seg-hit { fill: var(--left); background: var(--left); }
.seg-redo { fill: var(--warn); background: var(--warn); }
.seg-rest { fill: var(--border-strong); background: var(--border-strong); }
.tick { stroke: var(--text); stroke-width: 1; opacity: 0.55; pointer-events: none; }
.handle { cursor: ew-resize; outline: none; }
.handle line { stroke: var(--accent); stroke-width: 2; }
.handle circle { fill: var(--accent); }
.handle:focus-visible circle { stroke: var(--text); stroke-width: 2; }
.bar-kv { fill: var(--accent); }
.bar-ok { fill: var(--left); }
.bar-bad { fill: var(--danger); }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
</style>
