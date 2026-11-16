<!--
  MinHash-LSH 的 S 曲线 (对应 llm_train/m17_data_pipeline:lsh_dedup)。
  只讲一件事: 签名切 b 段 × r 行, 任一段全同才成候选; 成候选的概率 1-(1-s^r)^b 是一条 S 曲线, b 和 r 决定它在哪里拐弯。
  竖线位置取自 m17 demo 第 [1] 段输出: 轻改转载 Jaccard 均值 0.69 / 最低 0.48, 模板页 0.32, 校验线 0.5。
-->
<template>
  <LabFrame
    title="MinHash-LSH — 近似去重时, 哪两篇文档会被拿来比较?"
    sub="- 模拟什么: 网页语料里同一篇文章常被转载几十次, 每次改几个字。去重要找出这些「差不多一样」的文档对。两篇文档有多像, 用 Jaccard 相似度衡量: 把每篇切成所有连续 5 个词的片段 (5-gram), 两篇的片段集合交集大小除以并集大小, 1 是完全一样, 0 是毫无重合。几百万篇两两全比算不起, 所以用两步: MinHash 把每篇文档压成 128 个整数的签名 (签名里相同位置的比例就是 Jaccard 的估计); LSH (局部敏感哈希) 再把 128 位签名切成 $b$ 段、每段 $r$ 位 ($b \cdot r = 128$), 任何一段 $r$ 位全部相同的两篇就扔进同一个桶, 只有同桶的才真正拿来比较。
      - 图上是什么: 横轴是两篇文档的真实 Jaccard $s$, 纵轴是它们至少有一段签名全同、落进同一个桶 (成为候选) 的概率, 公式是 $1-(1-s^r)^b$。曲线是 S 形: 左边接近 0, 右边接近 1, 中间陡升的位置叫拐点, 约在 $(1/b)^{1/r}$。三条竖虚线是 m17 语料里实测的三类文档对: 红线「模板页 0.32」是只共享页头页脚、正文不同的两页 (不该删); 绿线「转载均值 0.69」是轻改过的转载 (该删); 黄线「校验线 0.5」是成候选之后的最终判定线。曲线上的圆点是你选的一对文档, 旁边标着它的 $s$ 和成候选的概率。
      - 可以做什么: 点「32 段 × 4 行」等预设按钮换一组 $b$、$r$; 拖「段数 b」和「每段行数 r」滑杆自由调 (两者乘起来不一定是 128, 右侧会显示哈希总数)。在图上拖那个圆点 (或用左右方向键) 换一个 Jaccard 值, 看这对文档成候选的概率。
      - 应该看到: $r$ 越大曲线越往右、越陡; $b$ 越大曲线越往左。右侧「哈希总数 b·r」等于 128 时变绿, 表示这组切法用的签名长度和 m17 一样; 「S 曲线拐点」是概率从低翻到高的位置; 「最低的转载 (0.48) 成候选」是最难抓的那篇转载 (它和原件的 Jaccard 只有 0.48) 被拿来比较的概率, 高于 0.7 变绿; 「模板页 (0.32) 成候选」是模板页被白白拿来比较的概率, 低于 0.3 变绿。两个读数很难同时变绿, 这就是切法的两难。"
    module="llm_train/m17"
    run="python -m llm_train.m17_data_pipeline.demo"
    :challenge="{
      ask: '想让 Jaccard 0.3 的模板页更少被拿来比较, 同时不漏掉 Jaccard 0.5 的转载。只能用 128 个哈希 ($b \\cdot r = 128$), 该往哪边调?',
      answer: '调不出两全其美的, 只能二选一, 所以真实做法是分两道关。\n- 先看两个参数各管什么: 一段 $r$ 位全部相同的概率是 $s^r$, $r$ 越大这个概率掉得越快, 曲线整体右移、变陡, 相似度低的对更难成候选。$b$ 是段数, 段越多撞上任何一段的机会越多, 曲线左移。拐点约在 $(1/b)^{1/r}$。\n- $b=32, r=4$ (m17 用的): 拐点 0.42。$s=0.3$ 的模板页成候选概率 0.23, $s=0.5$ 的转载 0.87。模板页会被白比一些, 但转载基本不漏。\n- $b=16, r=8$: 拐点 0.71。模板页几乎不会成候选 (接近 0), 可 $s=0.5$ 的转载也只剩 0.06, 大量转载直接漏掉。\n128 个哈希里没有一种切法能两边都好。所以 m17 让 LSH 偏宽 (宁可多给候选), 再对每个候选对用签名估出的 Jaccard 是否 ≥ 0.5 做最终判定。两个阈值分开调: S 曲线管「谁被拿来比较」, 决定召回和算量; 校验线管「比完之后删不删」。被白比一次的模板页只是多花一点算力, 不会被误删。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="p in PRESETS" :key="p.b" type="button" :class="{ active: b === p.b && r === p.r }" @click="(b = p.b), (r = p.r)">
          {{ p.b }} 段 × {{ p.r }} 行
        </button>
      </div>
      <LabSlider v-model="b" label="段数 b" :min="1" :max="64" />
      <LabSlider v-model="r" label="每段行数 r" :min="1" :max="16" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="成为候选的概率随 Jaccard 变化的 S 曲线, 曲线上的圆点可拖">
      <line :x1="X0" :x2="W - 10" :y1="py(0)" :y2="py(0)" class="axis" />
      <line :x1="X0" :x2="X0" :y1="py(0)" :y2="py(1)" class="axis" />
      <g v-for="t in [0, 0.2, 0.4, 0.6, 0.8, 1]" :key="t">
        <text :x="px(t)" :y="py(0) + 14" class="tick" text-anchor="middle">{{ t }}</text>
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <text :x="W - 10" :y="py(0) - 6" class="tick" text-anchor="end">真实 Jaccard s →</text>
      <g v-for="m in MARKS" :key="m.label">
        <line :x1="px(m.s)" :x2="px(m.s)" :y1="py(0)" :y2="py(1)" :class="m.cls" />
        <text :x="px(m.s) + 3" :y="py(1) + 10 + m.dy" class="mark">{{ m.label }}</text>
      </g>
      <polyline :points="curve" class="curve" />
      <line :x1="px(thr)" :x2="px(thr)" :y1="py(0)" :y2="py(0) - 8" class="thr" />
      <g class="draggable" tabindex="0" role="slider" aria-label="文档对的 Jaccard" :aria-valuenow="s"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (s = clamp(Math.round(pxInv(x) * 100) / 100, 0, 1)) })"
        @keydown.right.prevent.stop="s = clamp(+(s + 0.01).toFixed(2), 0, 1)" @keydown.left.prevent.stop="s = clamp(+(s - 0.01).toFixed(2), 0, 1)">
        <line :x1="px(s)" :x2="px(s)" :y1="py(0)" :y2="py(P(s))" class="guide" />
        <circle :cx="px(s)" :cy="py(P(s))" r="7" class="dot" />
        <text :x="px(s) + (s > 0.8 ? -10 : 10)" :y="py(P(s)) + (P(s) > 0.85 ? 22 : 4)" class="hl" :text-anchor="s > 0.8 ? 'end' : 'start'">s={{ s.toFixed(2) }} → {{ P(s).toFixed(3) }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>哈希总数 b·r (签名长度, 128 和 m17 一致)</span><b :class="b * r === 128 ? 'good' : ''">{{ b * r }}</b></div>
      <div class="kv"><span>S 曲线拐点 (1/b)^(1/r), 概率从低翻到高的位置</span><b>{{ thr.toFixed(2) }}</b></div>
      <!-- 拖动的那个点的读数直接写在图上, 这里不重复 -->
      <div class="kv"><span>最难抓的转载 (Jaccard 0.48) 成候选的概率</span><b :class="P(0.48) > 0.7 ? 'good' : 'bad'">{{ P(0.48).toFixed(3) }}</b></div>
      <div class="kv"><span>模板页 (Jaccard 0.32) 被白比一次的概率</span><b :class="P(0.32) < 0.3 ? 'good' : 'bad'">{{ P(0.32).toFixed(3) }}</b></div>
      <div class="lab-note">
        <p>★ 成候选只是「被拿来比较」, 不是「被删」。m17 对每个候选对还要用签名估出的 Jaccard 是否 ≥ 0.5 再校验一次, 校验过了才判为重复。所以模板页成候选只是多花算力, 不会被误删。</p>
        <p>m17 实测 (b=32、r=4, 不随滑杆变化): 120 篇该删的重复里抓到 96.7% (召回 0.967), 误删 0 篇, 一共只比较了 162 对; 对照组「精确哈希」(整篇逐字相同才算重复) 召回只有 0.258, 因为改一个词哈希就全变了。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const PRESETS = [{ b: 32, r: 4 }, { b: 64, r: 2 }, { b: 16, r: 8 }, { b: 8, r: 16 }]
// m17 demo [1] 段: 轻改转载与原件 Jaccard 均值 0.69 / 最低 0.48; 模板页之间 0.32; 校验阈值 Ĵ ≥ 0.5
const MARKS = [
  { s: 0.32, label: '模板页 0.32', cls: 'm-bad', dy: 0 },
  { s: 0.5, label: '校验线 0.5', cls: 'm-chk', dy: 14 },
  { s: 0.69, label: '转载均值 0.69', cls: 'm-good', dy: 0 },
]
const b = ref(32), r = ref(4), s = ref(0.6), svg = ref(null)
const { start } = useDrag()

const P = (x) => 1 - (1 - x ** r.value) ** b.value // ★ 任一段 r 行全同的概率
const thr = computed(() => (1 / b.value) ** (1 / r.value))

const W = 560, H = 240, X0 = 34
const px = (v) => X0 + v * (W - X0 - 14)
const pxInv = (x) => (x - X0) / (W - X0 - 14)
const py = (v) => 14 + (1 - v) * (H - 40)
const curve = computed(() => range(201).map((i) => `${px(i / 200)},${py(P(i / 200))}`).join(' '))
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.thr { stroke: var(--accent); stroke-width: 2; }
.guide { stroke: var(--text-dim); stroke-dasharray: 2 3; }
.dot { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.hl { font-size: 11px; fill: var(--text); font-family: "SF Mono", Menlo, monospace; }
.mark { font-size: 9px; fill: var(--text-muted); }
.m-bad { stroke: var(--danger); stroke-dasharray: 4 3; }
.m-good { stroke: var(--left); stroke-dasharray: 4 3; }
.m-chk { stroke: var(--warn); stroke-dasharray: 4 3; }
</style>
