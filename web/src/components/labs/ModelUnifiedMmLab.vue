<!--
  统一多模态掩码实验台 (对应 llm_models/models/multimodal/transfusion.py:transfusion_mask)。
  只讲一件事: 一条序列里文本因果、图像段内双向, 这张掩码怎样同时满足 "按文本画图" 和 "看图说话"。
  掩码规则与 Python 一致: M[i, j] = (j ≤ i) OR (i、j 属于同一张图)。
-->
<template>
  <LabFrame
    title="统一多模态掩码 — 文本因果, 同一张图内部双向"
    sub="- 模拟什么: Transfusion 把文本 token 和图像 patch 排进同一条序列, 交给同一个 transformer。文本位置做 next-token 预测, 所以只能看前面; 图像位置做扩散去噪, 一张图的所有 patch 在同一个噪声水平上一起预测, 所以同一张图内部互相可见。两条规则合起来就是一张掩码: $M[i, j] = (j \le i)$ 或 ($i$、$j$ 属于同一张图)。BOI / EOI 是包住图像的开始和结束标记。
- 图上是什么: 行是 query (正在计算的位置), 列是 key (被看的位置), 格子颜色:
  - 实心格: 因果关系本来就允许看到的位置。
  - 亮色格: 只因为同属一张图才看得到的 「未来」 patch, 关掉段内双向时它们消失。
  - 空格: 看不到。
  - 网格上方和左侧的标签: bos 与汉字是文本, BOI / EOI 是标记, p 开头的是图像 patch; 交错布局里 · 后面的数字是第几张图。
- 可以做什么: 按界面从上到下:
  - 三个布局按钮: 文本在前 (按文本画图)、图像在前 (看图说话)、文图交错 (两张图)。
  - 「图像段内双向」 开关: 关掉后整张掩码退化成纯因果, 对照看图像位置少了什么。
  - 拖动 「每张图的 patch 数 N」。
  - 鼠标悬停或点击某一行, 也可以聚焦网格后按上下方向键, 选中一个 query, 右侧读数跟着变。
- 应该看到: 右侧读数:
  - 「可见对数」: 整张掩码里可见的格子数, 括号里是纯因果的数目。N = 16、文本在前时是 351 对 231, 与 infer 脚本打印的一致。
  - 「图像前文本看到的后方 patch」: 任何布局下都是 0, 所以文本在前时提示词不会被后面的图改写, 生成可以先写字再画图。
  - 「选中位置看得到的 patch」: 选中图像后的文本时等于整张图的 patch 数; 选中第一个 patch 时, 开着双向是 N, 关掉只剩 1。"
    module="llm_models/models/multimodal/transfusion.py"
    run="python -m llm_models.run_models.multimodal.transfusion.infer_transfusion"
    :challenge="{
      ask: '切到「图像在前」, 选中第一个 patch 那一行, 先预测: 关掉「图像段内双向」后它能看到几个 patch? 为什么扩散去噪必须打开这一项, 而图像前的文本却一直一个 patch 都看不到?',
      answer: '答案: 关掉后第一个 patch 只能看到它自己 (1 个), 打开时能看到整张图的 N 个。\n- 扩散去噪需要段内双向: 每一步去噪是对整张图在同一个噪声水平 $t$ 上同时预测所有 patch 的速度, 左上角的 patch 要和右下角协调形状和位置。纯因果时第一个 patch 只能凭自己的噪声猜, 整张图就退化成逐 patch 的光栅顺序, 那又回到了 Chameleon 式的逐 token 生成。infer 脚本里改最后一个 patch, 第一个 patch 的输出变化 1.5e-02; 换成纯因果掩码后变化为 0。\n- 图像前的文本保持因果: 双向只发生在同一张图内部, 文本和图像之间仍按先后顺序。这样按文本画图时提示词的表示不随图像变化, 看图说话时标题能读到整张图; infer 脚本里换掉整张图, 图像前文本的 logits 变化恰好为 0。\n- 代价: 图像段内每一步去噪都要整段前向。图像去噪完成后它的 K/V 不再变, 后面的文本可以像普通自回归一样复用缓存。',
    }"
  >
    <template #controls>
      <div class="row" role="group" aria-label="选择序列布局">
        <button v-for="l in LAYOUTS" :key="l.id" type="button" :class="{ active: layoutId === l.id }" :aria-pressed="layoutId === l.id" @click="layoutId = l.id">{{ l.label }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: bidir }" :aria-pressed="bidir" @click="bidir = !bidir">图像段内双向: {{ bidir ? '开' : '关 (纯因果)' }}</button>
      </div>
      <LabSlider v-model="N" label="每张图的 patch 数 N" :min="2" :max="16" />
    </template>

    <svg :viewBox="`0 0 ${M + S * cell} ${M + S * cell}`" class="grid" role="group" :aria-label="ariaLabel">
      <text v-for="(tok, j) in seq" :key="'c' + j" :x="M + j * cell + cell / 2" :y="M - 4" class="lbl" :class="tok.kind" :font-size="fs" text-anchor="start" :transform="`rotate(-60 ${M + j * cell + cell / 2} ${M - 4})`">{{ tok.name }}</text>
      <g v-for="(row, i) in grid" :key="i" @mouseenter="q = i" @click="q = i">
        <text :x="M - 4" :y="M + i * cell + cell * 0.75" class="lbl" :class="seq[i].kind" :font-size="fs" text-anchor="end">{{ seq[i].name }}</text>
        <rect v-for="(c, j) in row" :key="j" :x="M + j * cell" :y="M + i * cell" :width="cell - 1" :height="cell - 1" :class="['cell', c, { dimcell: i !== q }]">
          <title>{{ seq[i].name }} 看 {{ seq[j].name }}: {{ TIP[c] }}</title>
        </rect>
      </g>
      <rect
        class="qrow" :x="M - 1" :y="M + q * cell - 1" :width="S * cell + 1" :height="cell + 1"
        tabindex="0" role="slider" aria-label="选中的 query 行" :aria-valuenow="q" aria-valuemin="0" :aria-valuemax="S - 1" :aria-valuetext="seq[q].name"
        @keydown.up.prevent="q = Math.max(0, q - 1)" @keydown.down.prevent="q = Math.min(S - 1, q + 1)"
      />
    </svg>
    <ul class="legend" aria-label="颜色图例">
      <li><span class="sw causal" aria-hidden="true"></span>因果可见</li>
      <li><span class="sw bidir" aria-hidden="true"></span>同一张图才可见的后方 patch</li>
      <li><span class="sw hidden" aria-hidden="true"></span>看不到</li>
    </ul>

    <template #stats>
      <div class="kv"><span>可见对数 (纯因果)</span><b :class="visible > causalPairs ? 'good' : ''">{{ visible }} ({{ causalPairs }})</b></div>
      <div class="kv"><span>图像前文本看到的后方 patch</span><b :class="leak === 0 ? 'good' : 'bad'">{{ leak }}</b></div>
      <div class="kv"><span>选中 {{ seq[q].name }} 看得到的 patch / 序列里的 patch</span><b :class="seenPatches > 0 ? 'good' : ''">{{ seenPatches }} / {{ totalPatches }}</b></div>
      <div class="kv"><span>第一张图的第一个 patch 看得到的同图 patch</span><b :class="firstSees === N ? 'good' : 'bad'">{{ firstSees }} / {{ N }}</b></div>
      <div class="lab-note">
        <p><Tex text="规则: $M[i, j] = (j \le i)$ 或 (同一张图)。亮色格的个数是每张图 $N(N-1)/2$。" /></p>
        <p>训练时两种目标落在不同位置: 文本位置算 next-token CE, 图像位置算去噪 MSE, 总损失是两者加权和。</p>
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

const LAYOUTS = [
  { id: 'text_first', label: '文本在前 (按文本画图)' },
  { id: 'image_first', label: '图像在前 (看图说话)' },
  { id: 'interleaved', label: '文图交错 (两张图)' },
]
const TIP = { causal: '因果可见', bidir: '同一张图, 双向可见', hidden: '看不到' }
const layoutId = ref('text_first'), bidir = ref(true), N = ref(4), q = ref(0)

// 序列: 每项 { name, kind: text | mark | img, img: 图号或 -1 }
const seq = computed(() => {
  const t = (name) => ({ name, kind: 'text', img: -1 })
  const image = (k) => [
    { name: 'BOI', kind: 'mark', img: -1 },
    ...range(N.value).map((p) => ({ name: `p${p}` + (layoutId.value === 'interleaved' ? `·${k}` : ''), kind: 'img', img: k })),
    { name: 'EOI', kind: 'mark', img: -1 },
  ]
  if (layoutId.value === 'text_first') return [t('bos'), t('红'), t('圆'), ...image(0)]
  if (layoutId.value === 'image_first') return [t('bos'), ...image(0), t('红'), t('圆')]
  return [t('bos'), t('猫'), ...image(0), t('和'), t('狗'), ...image(1), t('比')]
})
const S = computed(() => seq.value.length)
watch(S, (s) => { q.value = Math.min(q.value, s - 1) })

// ★ 与 transfusion_mask 同一条规则: 因果, 或同属一张图 (开关关掉时只剩因果)
const grid = computed(() => seq.value.map((a, i) => seq.value.map((b, j) => {
  if (j <= i) return 'causal'
  return bidir.value && a.img >= 0 && a.img === b.img ? 'bidir' : 'hidden'
})))

const visible = computed(() => sum(grid.value.map((row) => row.filter((c) => c !== 'hidden').length)))
const causalPairs = computed(() => (S.value * (S.value + 1)) / 2)
const totalPatches = computed(() => seq.value.filter((x) => x.kind === 'img').length)
// 文本 / 标记位置看到排在它后面的 patch 数: 应恒为 0
const leak = computed(() => sum(seq.value.map((a, i) => (a.kind === 'img' ? 0 : grid.value[i].filter((c, j) => j > i && c !== 'hidden' && seq.value[j].kind === 'img').length))))
const seenPatches = computed(() => grid.value[q.value].filter((c, j) => c !== 'hidden' && seq.value[j].kind === 'img').length)
const firstSees = computed(() => {
  const i = seq.value.findIndex((x) => x.kind === 'img')
  return grid.value[i].filter((c, j) => c !== 'hidden' && seq.value[j].img === seq.value[i].img).length
})

// 画布: 格子边长固定, 左上留出标签空间
const cell = 14
const M = 34
const fs = 8
const ariaLabel = computed(() => `${S.value}×${S.value} 注意力掩码, 行是 query, 列是 key。可见 ${visible.value} 对, 纯因果 ${causalPairs.value} 对; 当前选中 ${seq.value[q.value].name}, 看得到 ${seenPatches.value} 个图像 patch。`)
</script>

<style scoped>
.grid { width: 100%; max-width: 560px; min-width: 300px; background: var(--code-bg); border-radius: var(--radius-sm); padding: 4px; }
.lbl { fill: var(--text-muted); font-family: var(--font-mono, monospace); }
.lbl.img { fill: var(--eye); }
.lbl.mark { fill: var(--warn); }
.cell.causal { fill: var(--accent); }
.cell.bidir { fill: var(--eye); }
.cell.hidden { fill: var(--border); opacity: 0.35; }
.cell.dimcell { opacity: 0.55; }
.cell.hidden.dimcell { opacity: 0.2; }
.qrow { fill: none; stroke: var(--text); stroke-width: 1; pointer-events: none; }
.qrow:focus-visible { outline: none; stroke: var(--accent); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 8px; list-style: none; padding: 0; color: var(--text-muted); font-size: 11px; }
.legend li { display: inline-flex; align-items: center; gap: 5px; }
.sw { width: 10px; height: 10px; border-radius: 2px; border: 1px solid var(--border-strong); }
.sw.causal { background: var(--accent); }
.sw.bidir { background: var(--eye); }
.sw.hidden { background: transparent; }
</style>
