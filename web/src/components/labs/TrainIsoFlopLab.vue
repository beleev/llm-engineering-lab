<!--
  IsoFLOP 曲线 (对应 llm_train/m19_scaling_laws:compute_optimal)。
  只讲一件事: 算力 C = 6ND 定死, 模型越大数据就越少; loss 对 N 是 U 形, 最低点就是 N_opt。
  拟合常数取自 m19 demo 第 [2] 段: E/A/α/B/β = 0.0080/1.213/1.075/460.3/1.125。
  6 个实测点取自第 [1] 段网格的反对角线 (C ≈ 7.99e6), 即 demo 打印的 IsoFLOP 行。曲线在前端按公式现算。
-->
<template>
  <LabFrame
    title="IsoFLOP — 钱一定, 模型做多大?"
    sub="曲线是拟合出的 $L(N, D)$ 在 $D = C/6N$ 上的取值, 横轴是参数量 $N$ (对数)。拖算力滑杆, 或者拖曲线上的点换一个模型大小。
      算力对齐 $C \approx 8 \times 10^6$ 时, 圆点是 m19 真训出来的 6 个模型。"
    module="llm_train/m19"
    run="python -m llm_train.m19_scaling_laws.demo"
    :challenge="{
      ask: '算力从 1e7 加到 1e9 (×100), 最优模型大小变几倍? 这个结论能搬到 LLM 上吗?',
      answer: '- 本例约 ×10.5: $N_{\\text{opt}}$ 从 100 到 1054, 因为 $a = \\beta/(\\alpha+\\beta) = 0.511$, $100^{0.511} \\approx 10.5$。数据量也约 ×9.5。\n- 不能照搬: 这里 $\\alpha \\approx 1.08$、$\\beta \\approx 1.13$, Chinchilla 论文是 0.34 / 0.28, 差了 3 倍多。$a$ 碰巧接近 0.5, 只因为这里 $\\alpha \\approx \\beta$。\n- 最优 D/N ≈ 150: 也不是论文的 20。一个回归样本和一个语言 token 的信息量不可比。\n能带走的是方法: 小规模扫一个 $(N, D)$ 网格, 拟合 $L(N,D)$, 在 $C = 6ND$ 上求最小。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lgC" label="算力 C" :min="6.5" :max="9.5" :step="0.05" :format="(v) => '1e' + v.toFixed(2)" />
      <div class="row">
        <button type="button" :class="{ active: Math.abs(lgC - LG_ISO) < 0.01 }" @click="lgC = LG_ISO">对齐实测 C ≈ 8e6</button>
        <button v-for="c in [7, 8, 9]" :key="c" type="button" :class="{ active: lgC === c }" @click="lgC = c">C = 1e{{ c }}</button>
        <button type="button" @click="lgN = Math.log10(opt.N)">把点放到最优</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="固定算力下 loss 随模型大小变化的 U 形曲线, 曲线上的圆点可拖">
      <g v-for="t in [0.01, 0.03, 0.1, 0.3]" :key="'y' + t">
        <line :x1="X0" :x2="W - 10" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <g v-for="n in [10, 100, 1000, 10000]" :key="'x' + n">
        <text :x="px(Math.log10(n))" :y="H - 6" class="tick" text-anchor="middle">N={{ n }}</text>
      </g>
      <line :x1="X0" :x2="W - 10" :y1="py(E)" :y2="py(E)" class="floor" />
      <text :x="W - 12" :y="py(E) - 4" class="tick" text-anchor="end">E = 0.008 (噪声底)</text>
      <polyline :points="curve" class="curve" />
      <line :x1="px(Math.log10(opt.N))" :x2="px(Math.log10(opt.N))" :y1="py(opt.L)" :y2="py(E)" class="optl" />
      <g v-if="aligned">
        <circle v-for="p in ISO" :key="p.N" :cx="px(Math.log10(p.N))" :cy="py(p.L)" r="4.5" class="meas" :class="{ best: p.N === 81 }" />
      </g>
      <g class="draggable" tabindex="0" role="slider" aria-label="模型参数量 N (对数)" :aria-valuenow="Math.round(10 ** lgN)"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (lgN = clamp(pxInv(x), LO, HI)) })"
        @keydown.right.prevent.stop="lgN = clamp(lgN + 0.05, LO, HI)" @keydown.left.prevent.stop="lgN = clamp(lgN - 0.05, LO, HI)">
        <circle :cx="px(lgN)" :cy="py(L(10 ** lgN))" r="7" class="dot" />
        <text :x="px(lgN) + (lgN > 3.2 ? -10 : 10)" :y="py(L(10 ** lgN)) - 8" class="hl" :text-anchor="lgN > 3.2 ? 'end' : 'start'">N={{ Math.round(10 ** lgN) }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>最优 N_opt / D_opt</span><b>{{ Math.round(opt.N) }} / {{ Math.round(opt.D) }}</b></div>
      <div class="kv"><span>最优 D/N</span><b>{{ (opt.D / opt.N).toFixed(0) }}</b></div>
      <div class="kv"><span>你选的 N={{ Math.round(10 ** lgN) }}: D / loss</span><b>{{ Math.round(10 ** lgC / 6 / 10 ** lgN) }} / {{ L(10 ** lgN).toFixed(4) }}</b></div>
      <div class="kv"><span>比最优多亏</span><b :class="waste < 0.1 ? 'good' : 'bad'">+{{ (waste * 100).toFixed(0) }}%</b></div>
      <div class="lab-note">
        <p>★ 本例 N_opt ∝ C^0.511; 套 Chinchilla 论文的指数 (0.34 / 0.28) 是 C^0.45。本例指数是玩具 teacher 的谱决定的, 不代表 LLM。</p>
        <p>m19 实测, 不随滑杆变化: C ≈ 8e6 时最优是 N=81 (0.0285), 拟合预测 89。这 6 个点也参与了拟合, 不是样本外检验。</p>
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

// m19 demo [2] 拟合结果
const E = 0.008, A = 1.213, AL = 1.075, B = 460.3, BE = 1.125
// m19 demo IsoFLOP 行 (C ≈ 7.99e6): N → 实测 MSE
const ISO = [[21, 0.0603], [41, 0.0312], [81, 0.0285], [161, 0.0338], [321, 0.0632], [641, 0.0853]].map(([N, L]) => ({ N, L }))
const LG_ISO = Math.log10(7.99e6)
const LO = 1, HI = 4
const lgC = ref(LG_ISO), lgN = ref(Math.log10(641)), svg = ref(null)
const { start } = useDrag()
const aligned = computed(() => Math.abs(lgC.value - LG_ISO) < 0.01)

const L = (N) => E + A / N ** AL + B / (10 ** lgC.value / 6 / N) ** BE // ★ D = C / 6N
// 闭式解, 与 compute_optimal 相同
const opt = computed(() => {
  const a = BE / (AL + BE), G = ((AL * A) / (BE * B)) ** (1 / (AL + BE))
  const N = G * (10 ** lgC.value / 6) ** a
  return { N, D: 10 ** lgC.value / 6 / N, L: L(N) }
})
const waste = computed(() => L(10 ** lgN.value) / opt.value.L - 1)

const W = 560, H = 250, X0 = 40, YLO = Math.log10(0.005), YHI = Math.log10(0.4)
const px = (lg) => X0 + ((lg - LO) / (HI - LO)) * (W - X0 - 14)
const pxInv = (x) => LO + ((x - X0) / (W - X0 - 14)) * (HI - LO)
const py = (v) => 10 + (1 - (clamp(Math.log10(v), YLO, YHI) - YLO) / (YHI - YLO)) * (H - 34)
const curve = computed(() => range(151).map((i) => LO + (i / 150) * (HI - LO)).filter((lg) => L(10 ** lg) < 0.4).map((lg) => `${px(lg)},${py(L(10 ** lg))}`).join(' '))
</script>

<style scoped>
.grid { stroke: var(--border); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.floor { stroke: var(--text-dim); stroke-dasharray: 2 4; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.optl { stroke: var(--left); stroke-dasharray: 4 3; }
.meas { fill: var(--eye); stroke: var(--bg); stroke-width: 1.5; }
.meas.best { fill: var(--left); }
.dot { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.hl { font-size: 11px; fill: var(--text); font-family: "SF Mono", Menlo, monospace; }
</style>
