<!--
  MinHash-LSH 的 S 曲线 (对应 llm_train/m17_data_pipeline:lsh_dedup)。
  只讲一件事: 签名切 b 段 × r 行, 任一段全同才成候选; 成候选的概率 1-(1-s^r)^b 是一条 S 曲线, b 和 r 决定它在哪里拐弯。
  竖线位置取自 m17 demo 第 [1] 段输出: 轻改转载 Jaccard 均值 0.69 / 最低 0.48, 模板页 0.32, 校验线 0.5。
-->
<template>
  <LabFrame
    title="MinHash-LSH — 近似去重时, 哪两篇文档会被拿来比较?"
    sub="- 模拟什么: 互联网网页语料中同一篇文章常被多处转载并伴随少量字词改动, 语料去重需要高效识别这类近似重复文档对。两篇文档的相似程度通常采用 Jaccard 相似度衡量: 将每篇文档切分为所有连续 5 个词组成的片段集合 (5-gram), 计算两集合交集大小除以并集大小, 取值 1 代表完全相同, 取值 0 代表毫无重合。由于数百万篇文档两两精确比对的复杂度过高, 工程上通常分两步筛选:
  1. MinHash 将每篇文档压缩为由 128 个整数组成的定长签名 (两签名对应位置相等的比例即为真实 Jaccard 相似度的无偏估计)。
  2. LSH (局部敏感哈希) 将 128 位签名均分为 $b$ 段、每段包含 $r$ 行 ($b \cdot r = 128$); 只要两篇文档在任意一段上的 $r$ 个哈希值完全一致便归入同一哈希桶, 后续仅对同桶内的候选文档对执行签名复核。
- 图上是什么: 横轴为文档对的真实 Jaccard 相似度 $s$, 纵轴为两篇文档至少有一段签名完全一致从而落入同一桶成为候选对的概率, 解析公式为 $1-(1-s^r)^b$。曲线呈典型 S 形: 低相似度区趋近于 0, 高相似度区趋近于 1, 中段陡峭跃升处的拐点约位于 $(1/b)^{1/r}$。曲线上的圆点代表当前选定的测试文档对, 旁侧标注其相似度 $s$ 与对应的候选召回概率。图中三条竖直虚线标出 m17 语料库实测的三类典型参考值:
  - 红线「模板页 0.32」: 仅共享网站页头页脚而正文完全不同的两篇网页 (不应删除)。
  - 绿线「转载均值 0.69」: 经过轻微改写的转载文章对均值 (应当去重)。
  - 黄线「校验线 0.5」: 进入候选桶后用于最终判定是否删除的相似度阈值。
- 可以做什么: 控件从上到下依次为:
  - 点击「32 段 × 4 行」等预设按钮快速切换常用的 $(b, r)$ 分段组合。
  - 拖动「段数 b」与「每段行数 r」滑杆自由组合参数 (乘积不必固定为 128, 右侧面板会实时显示所需哈希总数)。
  - 在图上拖动测试圆点 (或使用左右方向键) 改变真实 Jaccard 相似度, 观察对应文档对成为候选对的概率变化。
- 应该看到: 增大每段行数 $r$ 会使 S 曲线整体右移且变得更陡峭; 增大段数 $b$ 则使曲线左移。右侧后两项指标很难在同一组参数下同时变绿, 直观体现了固定签名长度下的召回与误报权衡。右侧读数含义如下:
  - 「哈希总数 b·r」: 等于 128 时显示绿色, 表示当前分段方案消耗的签名总长度与 m17 保持一致。
  - 「S 曲线拐点」: 候选概率从低值快速跃升至高值的阈值位置。
  - 「最低的转载 (0.48) 成候选」: 改动幅度最大、与原文 Jaccard 仅为 0.48 的转载样本被成功召回进候选桶的概率, 高于 0.7 时显示绿色。
  - 「模板页 (0.32) 成候选」: 仅共享网页模板的非重复文档对被误送入候选桶做无谓比对的概率, 低于 0.3 时显示绿色。"
    module="llm_train/m17"
    run="python -m llm_train.m17_data_pipeline.demo"
    :challenge="{
      ask: '若希望进一步减少 Jaccard 为 0.3 的模板页进入候选比对, 同时不漏掉 Jaccard 为 0.5 的改写转载, 在固定使用 128 个哈希值 ($b \\cdot r = 128$) 的约束下应向哪个方向调整参数?',
      answer: '在固定 128 位签名的约束下无法单靠 LSH 分段同时压低误报并保持高召回, 只能在两者间权衡, 因此工程实践普遍采用「宽进严出」的两级过滤架构。\n- 两个分段参数的作用机制: 单段 $r$ 个哈希值全部相等的概率为 $s^r$, 增大 $r$ 会使低相似度区间的碰撞概率急剧衰减, 曲线整体右移且变陡, 低相似度文档对更难进入候选桶; 段数 $b$ 越多则至少撞中一段的机会越大, 曲线整体左移。曲线拐点约位于 $(1/b)^{1/r}$。\n- 当取 $b=32, r=4$ (即 m17 默认配置) 时: 拐点位于 0.42。此时 $s=0.3$ 的模板页成为候选对的概率为 0.23, 而 $s=0.5$ 的边缘转载召回概率达 0.87; 虽然会混入少量模板页增加后续校验开销, 但基本不会漏检真实转载。\n- 若改为 $b=16, r=8$: 拐点右移至 0.71。此时模板页进入候选桶的概率几乎降至 0, 但 $s=0.5$ 的转载召回率也骤降至 0.06, 导致大批改写转载被直接漏检。\n在总哈希数固定为 128 的前提下, 没有任何整数切分能兼顾两端。因此 m17 将第一级 LSH 设得偏宽松以保障候选召回率, 随后在第二级对每个候选对计算完整 128 位签名的估计 Jaccard 值, 仅当估计值 ≥ 0.5 时才最终判定为重复文档。两道阈值职责分离:\n- LSH 的 S 曲线负责控制「哪些文档对进入精筛比对」, 决定召回上限与比对计算量。\n- 二次校验线负责裁决「精筛比对后是否实际删除」, 保障去重准确率。\n因此偶然落入候选桶的模板页仅会多消耗一次签名比对算力, 不会在二次校验中被误删。',
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
        <p>★ 进入候选桶仅代表「送入精筛比对」, 并不直接等同于「判定删除」。m17 会对每个候选对用完整签名估计的 Jaccard 相似度是否达到 0.5 执行二次复核, 复核通过才标记为重复文档; 因此模板页进入候选桶仅增加少量比对开销, 不会被误删。</p>
        <p>m17 语料库实测基准 (分段参数取 b 为 32、r 为 4, 不随滑杆变化):</p>
        <ul class="pts">
          <li><b>LSH:</b> 120 篇该删的重复里抓到 96.7% (召回 0.967), 误删 0 篇, 一共只比较了 162 对。</li>
          <li><b>对照组「精确哈希」:</b> 整篇逐字相同才算重复, 召回只有 0.258, 因为改一个词哈希就全变了。</li>
        </ul>
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
