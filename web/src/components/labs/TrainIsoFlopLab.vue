<!--
  IsoFLOP 曲线 (对应 llm_train/m19_scaling_laws:compute_optimal)。
  只讲一件事: 算力 C = 6ND 定死, 模型越大数据就越少; loss 对 N 是 U 形, 最低点就是 N_opt。
  拟合常数取自 m19 demo 第 [2] 段: E/A/α/B/β = 0.0080/1.213/1.075/460.3/1.125。
  6 个实测点取自第 [1] 段网格的反对角线 (C ≈ 7.99e6), 即 demo 打印的 IsoFLOP 行。曲线在前端按公式现算。
-->
<template>
  <LabFrame
    title="IsoFLOP — 算力预算定死了, 模型该做多大?"
    sub="- 模拟什么: 预训练启动前需确定两个核心规模参数: 模型参数量 $N$ 与训练样本数 $D$。总训练算力近似满足 $C = 6ND$ (每个参数在单个样本上前向传播约需 2 次浮点运算, 反向传播约需 4 次), 因此在给定算力预算 $C$ 下, 增大模型规模就必须缩减训练数据量, 反之亦然。IsoFLOP 曲线固定算力预算 $C$, 观察不同 $(N, D)$ 分配比例下最终训练损失的变化规律。scaling law 将最终损失建模为经验幂律公式 $L(N, D) = E + A/N^\alpha + B/D^\beta$, 本实验台的五个拟合常数取自 m19 在 36 个小型 MLP 上实测训练后的拟合结果 (分别为 E 取 0.008、A 取 1.213、α 取 1.075、B 取 460.3、β 取 1.125)。公式三项的物理含义如下:
  - $E$: 数据固有的不可约噪声下界 (任意增大模型与数据均无法消除)。
  - $A/N^\alpha$: 模型容量不足引入的逼近误差。
  - $B/D^\beta$: 训练样本不足引入的有限数据误差。
- 图上是什么: 横轴为模型参数量 $N$ (对数刻度), 纵轴为最终 loss (对数刻度)。
  - 实线为将 $D = C/6N$ 代入幂律公式后得到的等算力 loss 曲线, 整体呈 U 形: 左翼受限于模型容量过小, 右翼受限于模型过大而训练数据不足。
  - 底部水平虚线标示 $E = 0.008$ 的不可约噪声底, 投入再多算力也无法低于该界限。
  - 绿色竖直虚线标出当前算力下曲线谷底对应的最优参数量 $N_{\text{opt}}$。
  - 可拖拽的大圆点代表当前选定的模型规模。
  - 当算力预算对齐至 $C \approx 8 \times 10^6$ 时, 曲线旁会额外显示 6 个小圆点, 对应 m19 在该算力预算下实际训练的 6 个模型实测 loss, 其中绿色圆点为实测最优模型 (N 为 81)。
- 可以做什么: 控件从上到下依次为:
  - 拖动「算力 C」滑杆调节总算力预算 (范围为 1e6.5 至 1e9.5), 整条等算力曲线随之平移。
  - 点击「对齐实测 C ≈ 8e6」回到带有实测散点的算力档位。
  - 点击「C = 1e7 / 1e8 / 1e9」快捷跳转至整数量级算力。
  - 点击「把点放到最优」将大圆点直接移至当前曲线谷底。
  - 在图上左右拖动大圆点 (或使用左右方向键) 尝试不同模型参数量。
- 应该看到: 算力预算越大, U 形曲线的谷底越向右下方移动, 即最优模型规模增大的同时最低可达 loss 下降; 将圆点拖向左右两端均会导致显著的性能损失。右侧读数含义如下:
  - 「最优 N_opt / D_opt」: 当前算力下由拟合公式解析求得的最优参数量与最优样本数。
  - 「最优 D/N」: 最优配置下平均每个参数分配到的训练样本数, 本例约为 150。
  - 「你选的 N」: 当前选定参数量、按 $D = C/6N$ 折算的样本数及拟合公式预测的 loss。
  - 「比最优多亏」: 当前选定配置的预测 loss 相对谷底最优值的超出百分比, 低于 10% 时显示为绿色。"
    module="llm_train/m19"
    run="python -m llm_train.m19_scaling_laws.demo"
    :challenge="{
      ask: '当总算力预算从 1e7 增至 1e9 (扩大 100 倍) 时, 最优模型参数量会扩大多少倍? 这一具体倍数能否直接套用到真实 LLM 预训练中?',
      answer: '结论:\n- 本例中最优模型规模约扩大 10.5 倍。\n- 该具体数值不能直接套用到真实 LLM 上, 可迁移的是拟合与外推的方法论。\n逐项解析:\n- 10.5 倍的推导: 将算力约束 $D = C/6N$ 代入 $L(N,D)$ 并对 $N$ 求极小值, 可得幂律关系 $N_{\\text{opt}} \\propto C^a$, 其中分配指数 $a = \\beta/(\\alpha+\\beta)$。本例拟合常数为 $\\alpha = 1.075$、$\\beta = 1.125$, 代入得 $a = 0.511$, 故算力扩大 100 倍时最优参数量扩大 $100^{0.511} \\approx 10.5$ 倍。右侧读数可直接验证: 当 C 为 1e7 时 $N_{\\text{opt}}$ 为 100, 当 C 为 1e9 时增至 1054。剩余指数 $1 - a = 0.489$ 分配给数据量, 因此 $D_{\\text{opt}}$ 同步扩大约 9.5 倍, 新增算力大致由模型规模与数据量对半平分。\n- 数值不可直接迁移的原因: 幂律指数取决于底层数据分布的谱结构。此处 $\\alpha \\approx 1.08$、$\\beta \\approx 1.13$ 由 m19 合成回归任务的玩具数据生成器 (teacher 网络第 k 个隐单元权重按 1/k 衰减) 所决定; 而 Chinchilla 论文在自然语言语料上拟合出的指数分别为 0.34 与 0.28, 与本例相差 3 倍以上。本例中 $a$ 接近 0.5 仅源于合成任务的 $\\alpha$ 与 $\\beta$ 数值恰好相近, 更换数据分布后指数即随之改变。\n- 最优数据参数比 D/N 同样不可通约: 本例最优配置约为每参数 150 个回归样本, 而 Chinchilla 论文约为每参数 20 个语言 token。单个合成回归样本与单个自然语言 token 所承载的信息量无法直接等同。\n真正可迁移至大模型工程的是完整实验方法:\n1. 在小算力尺度上扫描一组 $(N, D)$ 网格并完成真实训练。\n2. 由实测损失点拟合出经验公式的五个常数。\n3. 在 $C = 6ND$ 算力约束下解析求解极小值点, 据此确定大规模预算下的最优模型参数量与数据配比。',
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
        <p>★ 最优点怎么算:</p>
        <ol class="pts">
          <li>把 D = C/6N 代进 L(N, D)。</li>
          <li>对 N 求最小。</li>
          <li>得到 N_opt ∝ C^a, a = β/(α+β)。</li>
        </ol>
        <p>本例拟合分配指数 a 为 0.511, 算力扩大 100 倍时最优模型规模约扩大 10.5 倍; 若代入 Chinchilla 论文的自然语言指数 (0.34 / 0.28), 最优参数量则按 C^0.45 缩放。本例指数由合成玩具数据生成器的谱衰减规律决定, 不代表真实 LLM 的缩放常数。</p>
        <p>m19 固定实测读数 (不随滑杆变化): 当算力 C 约为 8e6 时, 实际训练的 6 个模型中表现最优的是参数量 N 为 81 的配置 (实测 loss 为 0.0285), 与拟合公式预测的最优值 N 为 89 基本吻合。由于这 6 个实测点本身参与了五参数拟合, 该对比仅作为拟合自洽性检验, 而非样本外泛化验证。</p>
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
