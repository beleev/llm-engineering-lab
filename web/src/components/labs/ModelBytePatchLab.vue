<!--
  BLT 熵切分实验台 (对应 llm_models/models/language_models/byte_latent.py:entropy_patch_ids)。
  只讲一件事: 下一字节熵高的地方开新 patch, 全局模型的步数按信息量分配; 固定长度切分则不管词在哪里一刀切下去。
  熵值是 infer_byte_latent 里训好的熵模型对同一段文本的实测输出, 切分逻辑与 Python 一致。
-->
<template>
  <LabFrame
    title="BLT 熵切分 — 难猜的字节开新 patch, 好猜的字节并进前一个"
    sub="- 模拟什么: Byte Latent Transformer 不用 tokenizer, 直接读 UTF-8 字节。一个小的字节级语言模型先给每个位置算下一字节熵 $H(x_t)$, 也就是只看前文时这个字节有多难猜; $H(x_t) \gt \theta$ 的位置开一个新 patch。大模型 (全局 latent transformer) 每个 patch 只走一步, 所以 patch 数就是全局模型的步数。
- 图上是什么: 64 根柱子是一段文本的 64 个字节, 柱高是熵 (nats), 值来自 infer_byte_latent 里训好的熵模型。柱子下面是字节本身, ASCII 原样显示, 汉字的 3 个 UTF-8 字节显示成十六进制; 最下面一行是词。交替的浅色底块是切出来的 patch, 红色虚线是阈值 $\theta$。
- 可以做什么: 用两个按钮在熵切分和固定长度切分之间切换。熵切分下可以上下拖动红线右端的圆点, 或拖动阈值滑杆; 固定切分下拖动 patch 长度滑杆。鼠标悬停在某根柱子上, 会标出它所在的 patch。
- 应该看到: 阈值落在词内的熵 (不超过 0.25, 样本开头没有前文的几个字节除外) 和词首的熵 (1.3 到 2.0) 之间时, 一个词正好是一个 patch, 英文词 8 字节和中文词 10 字节都只占全局模型一步。固定切分在 patch 数相同的情况下会把很多词切成两半。右侧依次读: patch 数, 平均每个 patch 多少字节, 全局步数和字节数之比, 被切成两半的词有几个。"
    module="llm_models/models/language_models/byte_latent.py"
    run="python -m llm_models.run_models.language_models.byte_latent.infer_byte_latent"
    :challenge="{
      ask: '把熵切分的阈值调到让平均 patch 长度约为 8 字节, 再切到固定 8 字节切分。两种切分的全局步数几乎一样, 你预计哪一种训练后在留出集上的 bits-per-byte 更低?',
      answer: '答案: 熵切分更低。train_byte_latent 实测 (seed 0, 600 步): 熵切分 BPB 0.220, 固定 8 字节切分 0.229, 两者都是平均 8 字节一个 patch; 换种子 1、2 时分别是 0.221 对 0.228、0.221 对 0.230。\n原因分三步:\n1. 这份合成数据里, 下一个词由上一个词决定, 而一个词的几个后继共用同一个词尾, 只有词首能分辨上一个词是谁。\n2. 局部模型只看得到约 5 个字节, 够不到上一个词的词首, 这条信息只能由全局模型送过来; 而全局输出 $g_j$ 只汇总 patch $j$ 之前的 patch。\n3. 熵切分的边界落在词首, 预测新词时上一个词完整地在前面的 patch 里; 固定切分常常从上一个词中间下刀, 词首和词尾落进不同 patch, 预测新词时上一个词的词首可能还没进入全局模型。\n这说明切分的价值在于边界放在哪里, 而不只是 patch 有多长。每字节一个 patch 的对照 (全局步数多 8 倍) BPB 为 0.219。这个差距来自刻意植入的结构, 玩具规模下不能外推到真实文本。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'entropy' }" @click="mode = 'entropy'">熵切分 (BLT)</button>
        <button type="button" :class="{ active: mode === 'fixed' }" @click="mode = 'fixed'">固定长度切分 (MegaByte)</button>
      </div>
      <LabSlider v-if="mode === 'entropy'" v-model="theta" label="熵阈值 θ" :min="0" :max="2" :step="0.05" unit=" nats" />
      <LabSlider v-else v-model="K" label="patch 长度" :min="1" :max="16" unit=" 字节" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H0 + 44}`" class="chart" role="group" aria-label="每个字节的下一字节熵与切出的 patch">
      <rect v-for="(p, j) in spans" :key="'p' + j" :x="X0 + p.start * BW" :y="0" :width="(p.end - p.start) * BW" :height="H0 + 26"
            class="band" :class="[j % 2 ? 'b1' : 'b0', { hl: hover != null && pid[hover] === j }]" />
      <rect v-for="(h, i) in ENT" :key="'b' + i" :x="X0 + i * BW + 1" :y="yOf(Math.min(h, HMAX))" :width="BW - 2" :height="H0 - yOf(Math.min(h, HMAX))"
            class="bar" :class="{ hot: mode === 'entropy' && boundary[i] }" @pointerenter="hover = i" @pointerleave="hover = null">
        <title>字节 {{ i }} ({{ label(i) }}): H = {{ isFinite(h) ? h.toFixed(2) : '无前文, 恒为边界' }}, patch {{ pid[i] }}</title>
      </rect>
      <text v-for="(b, i) in BYTES" :key="'t' + i" :x="X0 + i * BW + BW / 2" :y="H0 + 12" class="byte">{{ label(i) }}</text>
      <text v-for="w in WORDS" :key="'w' + w.start" :x="X0 + ((w.start + w.end) / 2) * BW" :y="H0 + 38" class="word" :class="{ cut: isCut(w) }">{{ w.text }}</text>
      <template v-if="mode === 'entropy'">
        <line :x1="X0" :x2="W - 12" :y1="yOf(theta)" :y2="yOf(theta)" class="th" />
        <circle :cx="W - 12" :cy="yOf(theta)" r="6" class="handle draggable" tabindex="0" role="slider" aria-label="拖动熵阈值"
                :aria-valuenow="theta" aria-valuemin="0" aria-valuemax="2"
                @pointerdown="start($event, { svg, onMove })"
                @keydown.up.prevent="setTheta(theta + 0.05)" @keydown.down.prevent="setTheta(theta - 0.05)" />
      </template>
      <text :x="2" :y="yOf(1) + 3" class="axis">1.0</text><text :x="2" :y="yOf(2) + 8" class="axis">2.0</text>
    </svg>

    <template #stats>
      <div class="kv"><span>patch 数 = 全局模型步数</span><b>{{ spans.length }}</b></div>
      <div class="kv"><span>平均每个 patch 的字节数</span><b>{{ (N / spans.length).toFixed(2) }}</b></div>
      <div class="kv"><span>全局步数 / 字节数</span><b :class="spans.length / N <= 0.2 ? 'good' : 'bad'">{{ (spans.length / N).toFixed(3) }}</b></div>
      <div class="kv"><span>被切成两半的词 (共 {{ WORDS.length }} 个)</span><b :class="cutCount === 0 ? 'good' : 'bad'">{{ cutCount }}</b></div>
      <div class="lab-note">
        <p>留出集 bits-per-byte, 实测, 不随左侧变化 (train_byte_latent, seed 0, 600 步, 同一模型和数据, 只换切分):</p>
        <ul class="pts">
          <li><b>熵切分:</b> 平均 8.03 字节/patch, BPB 0.220。</li>
          <li><b>固定 8 字节:</b> 平均 8.00 字节/patch, BPB 0.229。</li>
          <li><b>每字节一个 patch:</b> 全局步数多 8 倍, BPB 0.219。</li>
        </ul>
        <p>BPB 是每个字节平均要多少比特, 与分词方式无关, 字节模型和 token 模型可以直接比较。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

// infer_byte_latent 打印的样本与熵 (熵模型训 400 步, seed 0); 位置 0 没有前文, 记 Infinity
const TEXT = 'redwolf redbear oldbear 大模型 redlynx 老模型 大模型 biglynx'
const BYTES = Array.from(new TextEncoder().encode(TEXT)).slice(0, 64)
const ENT = [Infinity, 0.92, 0.47, 0.97, 0.15, 0.08, 0.10, 0.07, 1.98, 0.04, 0.05, 0.23, 0.05, 0.16, 0.18, 0.06,
  1.90, 0.06, 0.07, 0.23, 0.05, 0.12, 0.19, 0.06, 1.90, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05,
  0.05, 0.04, 1.36, 0.04, 0.05, 0.08, 0.05, 0.05, 0.04, 0.04, 1.90, 0.06, 0.07, 0.05, 0.04, 0.05,
  0.05, 0.05, 0.05, 0.04, 1.30, 0.09, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 1.42, 0.07]
const N = BYTES.length, BW = 12, X0 = 22, H0 = 110, HMAX = 2.2, W = X0 + N * BW + 24
// 词的字节区间: 按空格 (0x20) 切, 区间不含空格
const WORDS = (() => {
  const out = []; let s = 0
  for (let i = 0; i <= N; i++) if (i === N || BYTES[i] === 32) {
    if (i > s) out.push({ start: s, end: i, text: new TextDecoder().decode(new Uint8Array(BYTES.slice(s, i))) })
    s = i + 1
  }
  return out
})()

const mode = ref('entropy'), theta = ref(1.2), K = ref(8), hover = ref(null)
const svg = ref(null)
const { start } = useDrag()
const yOf = (h) => H0 - (h / HMAX) * (H0 - 6)
const setTheta = (v) => { theta.value = Math.round(clamp(v, 0, 2) * 20) / 20 }
const onMove = ({ y }) => setTheta(((H0 - y) / (H0 - 6)) * HMAX)
const label = (i) => (BYTES[i] >= 32 && BYTES[i] < 127 ? String.fromCharCode(BYTES[i]) : BYTES[i].toString(16).toUpperCase())

// ★ 边界规则与 entropy_patch_ids / fixed_patch_ids 一致: 位置 0 恒为边界
const boundary = computed(() => ENT.map((h, i) => i === 0 || (mode.value === 'entropy' ? h > theta.value : i % K.value === 0)))
const pid = computed(() => { let p = -1; return boundary.value.map((b) => (p += b ? 1 : 0)) })
const spans = computed(() => {
  const out = []
  boundary.value.forEach((b, i) => { if (b) out.push({ start: i, end: i + 1 }); else out[out.length - 1].end = i + 1 })
  return out
})
const isCut = (w) => pid.value[w.start] !== pid.value[w.end - 1]
const cutCount = computed(() => WORDS.filter(isCut).length)
</script>

<style scoped>
.chart { width: 100%; min-width: 760px; display: block; }
.band.b0 { fill: var(--accent); opacity: 0.08; }
.band.b1 { fill: var(--left); opacity: 0.14; }
.band.hl { opacity: 0.32; }
.bar { fill: var(--text-dim); }
.bar.hot { fill: var(--warn); }
.byte { font-size: 7px; text-anchor: middle; fill: var(--text-muted); font-family: var(--mono, monospace); }
.word { font-size: 9px; text-anchor: middle; fill: var(--text); }
.word.cut { fill: var(--danger); }
.th { stroke: var(--danger); stroke-width: 1.2; stroke-dasharray: 4 3; }
.handle { fill: var(--danger); }
.axis { font-size: 8px; fill: var(--text-dim); }
</style>
