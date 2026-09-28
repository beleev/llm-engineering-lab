<!--
  μTransfer (对应 llm_train/m20_mup:init_and_lrs / train)。
  只讲一件事: SP 下最优 lr 随宽度左移, 小模型调好的 lr 搬不过去; μP 下三个宽度的最优点对齐。
  loss 表逐格抄自 m20 demo 输出的 [SP] / [μP] 两张表 (150 步 Adam, 线性衰减到 0, 最后 10 步平均)。
  这是真训出来的数, 浏览器里不重跑, 只做查表和比较。
-->
<template>
  <LabFrame
    title="μP — 小模型调好的 lr 能直接搬到大模型吗?"
    sub="横轴是学习率 ($\log_2$), 纵轴是训练 loss (对数)。三条线是宽 32 / 128 / 512 的 MLP, 空心圈是各自的最优点。
      宽 32 是 base 模型。点图上任一列, 假设这就是你在 base 上调出的 lr, 看它搬到宽 512 后亏多少。"
    module="llm_train/m20"
    run="python -m llm_train.m20_mup.demo"
    :challenge="{
      ask: 'SP 下宽度 ×16, 最优 lr 应该缩多少? m20 实测缩了多少?',
      answer: '- 按直觉: 宽 16 倍, lr 该缩 16 倍 (左移 4 格)。\n- 实测: −5 → −11, 左移 6 格, 缩了 64 倍, 比 $1/\\text{width}$ 还多。宽 512 的 SP 曲线在 −9 到 −7 之间也不光滑, 是短训练加大 lr 的不稳定。\n- μP: 隐藏层和输出层的 lr 除以 $m = \\text{width}/32$, 输出层初始化也多除 $m$。三个宽度的最优点都落在 $2^{-5}$, 且同一 lr 下越宽越好。\n直觉的来由: Adam 每个元素的更新约等于 lr, 与梯度大小无关。隐藏层有 width 个输入, width 个同向小更新叠加, 激活的变化 $\\propto \\text{lr} \\cdot \\text{width}$。\n注意: 最优点附近很平 (宽 512: 0.0090 / 0.0088 / 0.0101)。常数 lr 时, μP 的最优会在 −5 / −7 / −8 之间漂。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: pz === 'sp' }" @click="pz = 'sp'">SP (标准参数化)</button>
        <button type="button" :class="{ active: pz === 'mup' }" @click="pz = 'mup'">μP</button>
        <button type="button" @click="pick = best(0)">用 base 的最优 lr</button>
      </div>
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="三个宽度的 loss 随学习率变化">
      <g v-for="t in [0.01, 0.03, 0.1, 0.3]" :key="t">
        <line :x1="X0" :x2="W - 10" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <g v-for="(lg, i) in LRS" :key="lg">
        <rect :x="px(i) - CW / 2" y="4" :width="CW" :height="H - 26" class="col" :class="{ on: pick === i }" tabindex="0" role="button"
          :aria-label="`选 lr = 2^${lg}`" @click="pick = i" @keydown.enter="pick = i" />
        <text :x="px(i)" :y="H - 8" class="tick" text-anchor="middle">{{ lg }}</text>
      </g>
      <g v-for="(w, k) in WIDTHS" :key="w" class="curves">
        <polyline :points="TABLE[pz][k].map((v, i) => `${px(i)},${py(v)}`).join(' ')" fill="none" :stroke="COLORS[k]" stroke-width="2" />
        <circle :cx="px(best(k))" :cy="py(TABLE[pz][k][best(k)])" r="6" fill="none" :stroke="COLORS[k]" stroke-width="2" />
        <circle :cx="px(pick)" :cy="py(TABLE[pz][k][pick])" r="3.5" :fill="COLORS[k]" />
        <text :x="W - 12" :y="16 + k * 13" class="leg" text-anchor="end" :fill="COLORS[k]">宽 {{ w }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>最优 log₂ lr (宽 32/128/512)</span><b :class="aligned ? 'good' : 'bad'">{{ [0, 1, 2].map((k) => LRS[best(k)]).join(' / ') }}</b></div>
      <div class="kv"><span>选的 lr = 2^{{ LRS[pick] }}: 宽 512 的 loss</span><b>{{ TABLE[pz][2][pick].toFixed(4) }}</b></div>
      <div class="kv"><span>宽 512 自己的最优</span><b>{{ TABLE[pz][2][best(2)].toFixed(4) }}</b></div>
      <div class="kv"><span>搬过去亏了</span><b :class="ratio < 1.2 ? 'good' : 'bad'">×{{ ratio.toFixed(2) }}</b></div>
      <p class="lab-note">★ SP 与 μP 的全部差别在 init_and_lrs 的两处 /m。宽 32 时 m=1, 两张表的第一行完全相同。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

const WIDTHS = [32, 128, 512], LRS = [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4]
const COLORS = ['var(--eye)', 'var(--accent)', 'var(--left)']
// m20 demo 输出, 每行 = 一个宽度在 log2(lr) = -13 … -4 上的 loss
const TABLE = {
  sp: [
    [0.5766, 0.4213, 0.2755, 0.1557, 0.1011, 0.0644, 0.0395, 0.0262, 0.0186, 0.0248],
    [0.1647, 0.1101, 0.0722, 0.0431, 0.0257, 0.0184, 0.0162, 0.0288, 0.0426, 0.0503],
    [0.0406, 0.0258, 0.0215, 0.0227, 0.0371, 0.0628, 0.0358, 0.063, 0.0976, 0.091],
  ],
  mup: [
    [0.5766, 0.4213, 0.2755, 0.1557, 0.1011, 0.0644, 0.0395, 0.0262, 0.0186, 0.0248],
    [0.3559, 0.3025, 0.2026, 0.0887, 0.0507, 0.0218, 0.0123, 0.0107, 0.0101, 0.0128],
    [0.3585, 0.3162, 0.2201, 0.0737, 0.0418, 0.0147, 0.0095, 0.009, 0.0088, 0.0101],
  ],
}
const pz = ref('sp'), pick = ref(8)
const best = (k) => TABLE[pz.value][k].reduce((bi, v, i, xs) => (v < xs[bi] ? i : bi), 0)
const aligned = computed(() => best(0) === best(1) && best(1) === best(2))
const ratio = computed(() => TABLE[pz.value][2][pick.value] / TABLE[pz.value][2][best(2)]) // ★ base 的 lr 直接用到宽 512

const W = 560, H = 250, X0 = 40, CW = 44, YLO = Math.log10(0.006), YHI = Math.log10(0.7)
const px = (i) => X0 + 16 + (i / (LRS.length - 1)) * (W - X0 - 40)
const py = (v) => 8 + (1 - (Math.log10(v) - YLO) / (YHI - YLO)) * (H - 34)
</script>

<style scoped>
.grid { stroke: var(--border); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.leg { font-size: 11px; font-family: "SF Mono", Menlo, monospace; }
.col { fill: transparent; cursor: pointer; }
.col:hover { fill: var(--accent-soft); }
.col.on { fill: var(--accent-soft); }
.curves { pointer-events: none; }
</style>
