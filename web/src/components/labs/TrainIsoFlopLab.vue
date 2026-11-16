<!--
  IsoFLOP 曲线 (对应 llm_train/m19_scaling_laws:compute_optimal)。
  只讲一件事: 算力 C = 6ND 定死, 模型越大数据就越少; loss 对 N 是 U 形, 最低点就是 N_opt。
  拟合常数取自 m19 demo 第 [2] 段: E/A/α/B/β = 0.0080/1.213/1.075/460.3/1.125。
  6 个实测点取自第 [1] 段网格的反对角线 (C ≈ 7.99e6), 即 demo 打印的 IsoFLOP 行。曲线在前端按公式现算。
-->
<template>
  <LabFrame
    title="IsoFLOP — 算力预算定死了, 模型该做多大?"
    sub="- 模拟什么: 开训前要定两个数: 模型多大 (参数量 $N$) 和喂多少训练数据 (样本数 $D$)。训练一共要花的算力约是 $C = 6ND$ (每个参数每个样本前向 2 次、反向 4 次浮点运算), 所以预算 $C$ 一旦定死, 模型做大就只能少看数据, 反之亦然。scaling law 是把最终 loss 写成 $L(N, D) = E + A/N^\alpha + B/D^\beta$ 的经验公式: $E$ 是数据本身的噪声 (谁也消不掉), 第二项是「模型太小」的代价, 第三项是「数据太少」的代价。这五个常数是 m19 真训了 36 个小 MLP 之后拟合出来的 (E=0.008, A=1.213, α=1.075, B=460.3, β=1.125)。IsoFLOP 曲线就是固定 $C$, 只换 $N$ 和 $D$ 的分法, 看 loss 怎么变。
      - 图上是什么: 横轴是参数量 $N$ (对数刻度), 纵轴是 loss (对数刻度)。曲线是把 $D = C/6N$ 代进拟合公式后 loss 随 $N$ 的变化, 呈 U 形: 左边模型太小, 右边模型太大、数据不够。底部那条虚线是 $E = 0.008$ 的噪声底, 再多算力也到不了它下面。绿色竖虚线标着这条曲线的最低点 $N_{\text{opt}}$。可拖的大圆点是你选的模型大小。只有算力对齐到 $C \approx 8 \times 10^6$ 时, 曲线上还会出现 6 个小圆点, 那是 m19 在这个算力下真训出来的 6 个模型的实测 loss, 绿色那个是实测最好的 (N=81)。
      - 可以做什么: 拖「算力 C」滑杆改预算 (1e6.5 到 1e9.5), 整条曲线会跟着移动; 点「对齐实测 C ≈ 8e6」回到有实测点的那条; 点「C = 1e7 / 1e8 / 1e9」跳到整数量级; 「把点放到最优」把大圆点挪到曲线最低处。在图上拖大圆点 (或左右方向键) 换一个模型大小。
      - 应该看到: 算力越大, U 形的谷底越往右 (最优模型越大) 也越往下 (loss 越低)。右侧「最优 N_opt / D_opt」是当前算力下拟合公式算出的最优参数量和数据量; 「最优 D/N」是每个参数平均分到多少训练样本, 本例约 150; 「你选的 N」后面是按 $D = C/6N$ 推出的数据量和拟合公式给的 loss; 「比最优多亏」是你选的 loss 比谷底高了百分之几, 低于 10% 变绿。把点拖到两头, 都会看到亏得很多: 模型太小和太大都不划算。"
    module="llm_train/m19"
    run="python -m llm_train.m19_scaling_laws.demo"
    :challenge="{
      ask: '算力从 1e7 加到 1e9 (×100), 最优模型大小变几倍? 这个结论能搬到 LLM 上吗?',
      answer: '本例约 10.5 倍; 这个数字不能搬到 LLM 上, 能搬的是方法。\n- 为什么是 10.5 倍: 把 $D = C/6N$ 代进 $L(N,D)$ 对 $N$ 求最小, 得到 $N_{\\text{opt}} \\propto C^a$, 指数 $a = \\beta/(\\alpha+\\beta)$。本例 $\\alpha = 1.075$、$\\beta = 1.125$, 所以 $a = 0.511$, $100^{0.511} \\approx 10.5$。读数里能直接看到: C=1e7 时 $N_{\\text{opt}}$ 是 100, C=1e9 时是 1054。剩下的指数 $1 - a = 0.489$ 归数据, 所以 $D_{\\text{opt}}$ 也涨约 9.5 倍, 算力大致一半给模型、一半给数据。\n- 为什么不能照搬: 指数来自数据。这里 $\\alpha \\approx 1.08$、$\\beta \\approx 1.13$, 是 m19 那个玩具数据生成器 (teacher 网络的第 k 个单元权重按 1/k 衰减) 决定的; Chinchilla 论文在真实语言数据上拟合的是 0.34 / 0.28, 差了 3 倍多。$a$ 碰巧接近 0.5 只是因为这里 $\\alpha$ 和 $\\beta$ 恰好差不多, 换个数据生成器就变。\n- D/N 也对不上: 本例最优是每个参数约 150 个样本, 论文是约 20 个 token。这里一个「样本」是一个回归数据点, 和一个语言 token 的信息量不能比。\n能带走的是方法: 在小规模上扫一个 $(N, D)$ 网格真训一遍, 拟合出五个常数, 再在 $C = 6ND$ 的约束下求最小值, 就得到大预算下该用多大的模型。',
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
      <div class="kv"><span>当前算力下的最优: 参数量 N_opt / 数据量 D_opt</span><b>{{ Math.round(opt.N) }} / {{ Math.round(opt.D) }}</b></div>
      <div class="kv"><span>最优 D/N (每个参数分到几个样本)</span><b>{{ (opt.D / opt.N).toFixed(0) }}</b></div>
      <div class="kv"><span>你选的 N={{ Math.round(10 ** lgN) }}: 能喂的数据量 D / 拟合给的 loss</span><b>{{ Math.round(10 ** lgC / 6 / 10 ** lgN) }} / {{ L(10 ** lgN).toFixed(4) }}</b></div>
      <div class="kv"><span>比最优多亏 (loss 高出谷底的百分比)</span><b :class="waste < 0.1 ? 'good' : 'bad'">+{{ (waste * 100).toFixed(0) }}%</b></div>
      <div class="lab-note">
        <p>★ 最优点怎么算: 把 D = C/6N 代进 L(N, D), 对 N 求最小, 得到 N_opt ∝ C^a, a = β/(α+β)。本例 a = 0.511, 算力翻 100 倍最优模型大约翻 10.5 倍; 套 Chinchilla 论文的指数 (0.34 / 0.28) 则是 C^0.45。本例的指数由玩具数据生成器的谱决定, 不代表 LLM。</p>
        <p>m19 实测, 不随滑杆变化: C ≈ 8e6 时真训出来的 6 个模型里最好的是 N=81 (loss 0.0285), 拟合公式预测的最优是 N=89, 对得上。这 6 个点也参与了拟合, 所以这只是自洽检查, 不是样本外检验。</p>
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
