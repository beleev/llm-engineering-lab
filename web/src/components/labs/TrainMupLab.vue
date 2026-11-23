<!--
  μTransfer (对应 llm_train/m20_mup:init_and_lrs / train)。
  只讲一件事: SP 下最优 lr 随宽度左移, 小模型调好的 lr 搬不过去; μP 下三个宽度的最优点对齐。
  loss 表逐格抄自 m20 demo 输出的 [SP] / [μP] 两张表 (150 步 Adam, 线性衰减到 0, 最后 10 步平均)。
  这是真训出来的数, 浏览器里不重跑, 只做查表和比较。
  SP 与 μP 只差三处 / m: 输出层初始化的标准差一处 (方差除 m²), 隐藏层和输出层的 lr 各一处。
-->
<template>
  <LabFrame
    title="μP — 在小模型上调好的学习率, 能直接搬到大模型吗?"
    sub="- 模拟什么: 超大规模模型受算力预算所限通常只能完整训练一次, 无法直接在大模型上网格搜索最优学习率 (lr)。工程上的理想路径是先在窄宽度的小规模代理模型 (称为 base 模型) 上扫描出最优 lr, 再零改动直接迁移至目标宽模型。然而在标准参数化 (SP, 即常规初始化配合全网统一 lr) 下这一直接迁移并不成立: 网络宽度越大, 最优 lr 越小, 原样照搬会导致宽模型收敛严重受损或发散。μP (最大更新参数化) 通过按相对宽度比例 $m$ (定义为当前模型宽度除以 base 宽度) 修正三处缩放规则, 使最优 lr 不再随模型加宽而漂移, 从而实现超参数跨尺度零样本迁移 (μTransfer)。m20 离线实测训练了 3 种宽度乘 10 档学习率再乘 2 种参数化共 60 个小型 MLP, 本实验台直接加载并展示该实测结果表。μP 相对 SP 的三处缩放改动如下:
  - 输出层权重初始化标准差额外除以 $m$。
  - 隐藏层参数的学习率除以 $m$。
  - 输出层参数的学习率除以 $m$。
- 图上是什么: 横轴为学习率的 $\log_2$ 对数值 (范围为 −13 至 −4, 每向右一格对应 lr 翻倍), 纵轴为最终训练 loss (对数刻度, 越低代表收敛越优)。
  - 三条折线分别展示宽度为 32、128 与 512 的 MLP 在各档学习率下的最终 loss, 曲线颜色与右上角图例一一对应。
  - 每条折线上的空心圆圈标出该宽度模型自身取得最低 loss 的最优 lr 档位。
  - 竖直高亮列标示当前选定的候选迁移 lr, 三条折线在该列上的实心圆点对应三种宽度模型采用该同一 lr 时的实测 loss。
- 可以做什么: 控件从上到下依次为:
  - 点击「SP (标准参数化)」与「μP」切换参数化方案, 图上三条宽度曲线将整体切换。
  - 点击图中任意学习率列 (或聚焦后使用左右方向键切换), 将该档位视作在窄模型 (宽度 32) 上调优选定的待迁移 lr。
  - 点击「用 base 的最优 lr」一键选中宽度 32 代理模型的最优学习率列。
- 应该看到: 在 SP 模式下三个宽度的最优空心圈自右向左依次散开 (宽度 32 位于 −5, 宽度 128 位于 −7, 宽度 512 左移至 −11), 模型越宽则最优 lr 越小; 切换至 μP 模式后, 三个宽度的最优空心圈整齐对齐在同一列 −5 上, 且在同一 lr 下模型越宽曲线越靠下 (即加宽稳定带来单调收益)。若在 SP 下强行把 base 模型的最优 lr 迁移至宽度 512 模型, 最终 loss 会劣化 4.5 倍。右侧读数含义如下:
  - 「最优 log₂ lr」: 并列显示三种宽度各自的最优学习率对数值, 三者完全一致时显示绿色。
  - 「选的 lr」对应行: 将当前选定的迁移 lr 原样应用于宽度 512 目标模型所得的实测 loss。
  - 「宽 512 自己的最优」: 宽度 512 模型遍历全部候选 lr 所能达到的最低基准 loss。
  - 「搬过去亏了」: 迁移所得 loss 除以目标模型自身最优 loss 的比值, 取值 1.00 代表零损耗迁移。"
    module="llm_train/m20"
    run="python -m llm_train.m20_mup.demo"
    :challenge="{
      ask: '在标准参数化 (SP) 下将网络宽度扩大至 16 倍, 理论上最优学习率应当缩小多少倍? m20 的实测结果实际缩小了多少倍?',
      answer: '按更新量阶数理论推导应当缩小 16 倍, 而 m20 网格实测缩小了 64 倍。\n- 理论缩放机制: 根源在于 Adam 优化器的逐元素归一化特性。Adam 对单个权重参数的单步更新幅值近似等于 lr, 几乎不随梯度幅值本身缩放。隐藏层每个输出神经元连接着 width 个输入权重, 当 width 个同向小更新在内积求和中相干叠加时, 下一层激活值的单步变化量便按 $\\propto \\text{lr} \\cdot \\text{width}$ 随宽度线性增长。为保证特征激活的更新幅度在加宽时保持尺度不变, 学习率必须按 $1/\\text{width}$ 反比缩减: 宽度扩大 16 倍时最优 lr 应缩小 16 倍, 对应在 $\\log_2$ 横轴上向左平移 4 格。\n- 实测读数与有限网格误差: SP 模式下实测最优 $\\log_2$ lr 从宽度 32 的 −5 左移至宽度 512 的 −11, 共左移 6 格即缩小 64 倍, 超出了渐近理论预测的 16 倍。超出部分受限于本实验的小规模测量条件而无法精确剥离: 宽度 512 的 SP 损失曲线在 −9 至 −7 三格间出现非单调跳动 (依次为 0.0371、0.0628、0.0358), 源于仅训练 150 步叠加较大步长引发的短期振荡, 同时候选学习率网格以 2 倍为步进间距, 离散分辨率有限。\n- μP 的修正机制: 令相对宽度倍数 $m = \\text{width}/32$, 将隐藏层与输出层的学习率分别除以 $m$, 同时将输出层初始化标准差额外除以 $m$ (即初始化方差除以 $m^2$), 输入层保持不变。修正后三种宽度的实测最优点统一落在 $2^{-5}$ 档位, 且在该档位下随宽度增加训练 loss 单调下降 (依次为 0.0186、0.0101、0.0088)。\n同时需注意实验边界: 谷底附近的损失曲面相当平坦, 宽度 512 在 −7、−6、−5 三档下的 loss 分别为 0.0095、0.0090 与 0.0088, 差距极小, 离散网格最小值恰好严格落在 −5 带有一定偶然性。此外 m20 采用了线性衰减至 0 的学习率调度; 若改用恒定常数学习率, μP 在不同宽度下的实测最优点会在 −5、−7、−8 之间出现小幅漂移, 跨宽度对齐不如退火调度下整洁。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: pz === 'sp' }" @click="pz = 'sp'">SP (标准参数化)</button>
        <button type="button" :class="{ active: pz === 'mup' }" @click="pz = 'mup'">μP</button>
        <button type="button" @click="pick = best(0)">用 base 的最优 lr</button>
      </div>
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="三个宽度的 loss 随学习率变化, 点一列选学习率">
      <g v-for="t in [0.01, 0.03, 0.1, 0.3]" :key="t">
        <line :x1="X0" :x2="W - 10" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <!-- 10 列只占一个 Tab 停靠点, 左右方向键换列 -->
      <g class="cols" tabindex="0" role="slider" aria-label="在 base 上选的 log2 学习率, 左右方向键切换"
        :aria-valuemin="LRS[0]" :aria-valuemax="LRS[LRS.length - 1]" :aria-valuenow="LRS[pick]"
        @keydown.left.prevent="pick = Math.max(0, pick - 1)" @keydown.right.prevent="pick = Math.min(LRS.length - 1, pick + 1)">
        <g v-for="(lg, i) in LRS" :key="lg">
          <rect :x="px(i) - CW / 2" y="4" :width="CW" :height="H - 26" class="col" :class="{ on: pick === i }" @click="pick = i" />
          <text :x="px(i)" :y="H - 8" class="tick" text-anchor="middle">{{ lg }}</text>
        </g>
      </g>
      <g v-for="(w, k) in WIDTHS" :key="w" class="curves">
        <polyline :points="TABLE[pz][k].map((v, i) => `${px(i)},${py(v)}`).join(' ')" fill="none" :stroke="COLORS[k]" stroke-width="2" />
        <circle :cx="px(best(k))" :cy="py(TABLE[pz][k][best(k)])" r="6" fill="none" :stroke="COLORS[k]" stroke-width="2" />
        <circle :cx="px(pick)" :cy="py(TABLE[pz][k][pick])" r="3.5" :fill="COLORS[k]" />
        <text :x="W - 12" :y="16 + k * 13" class="leg" text-anchor="end" :fill="COLORS[k]">宽 {{ w }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>三个宽度各自的最优 log₂ lr (宽 32 / 128 / 512, 相同才算迁移成功)</span><b :class="aligned ? 'good' : 'bad'">{{ [0, 1, 2].map((k) => LRS[best(k)]).join(' / ') }}</b></div>
      <div class="kv"><span>把选的 lr = 2^{{ LRS[pick] }} 原样用到宽 512: loss</span><b>{{ TABLE[pz][2][pick].toFixed(4) }}</b></div>
      <div class="kv"><span>宽 512 自己扫出来的最低 loss</span><b>{{ TABLE[pz][2][best(2)].toFixed(4) }}</b></div>
      <div class="kv"><span>搬过去亏了 (上一行 ÷ 这一行, 1.00 是没亏)</span><b :class="ratio < 1.2 ? 'good' : 'bad'">×{{ ratio.toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>★ SP 与 μP 在代码实现上的全部差异仅在于 init_and_lrs 函数中的三处按比例系数 m 缩放 (其中 m 定义为当前宽度除以 32):</p>
        <ul class="pts">
          <li>输出层初始化的标准差除一次 (方差就是除 m²)。</li>
          <li>隐藏层的 lr 除一次。</li>
          <li>输出层的 lr 除一次。</li>
        </ul>
        <p>输入层的初始化与学习率均保持不变, 因为其扇入维度固定为原始输入特征维, 不随网络隐藏层宽度扩展而改变。</p>
        <p>宽度 32 即为参考基准 base 模型, 此时相对宽度倍数 m 为 1, 三处除以 1 后参数与学习率均无变化, 因此 SP 与 μP 下宽度 32 的损失曲线完全重合。</p>
        <p>图中三条曲线的数据直接取自 m20 离线实测结果表 (每个数据点均由 150 步 Adam 训练、学习率线性衰减至 0 并取末尾 10 步训练 loss 均值得到), 浏览器前端仅执行查表渲染而不重复运行训练。</p>
      </div>
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
.cols { outline: none; }
.cols:focus-visible .col.on { stroke: var(--accent); stroke-width: 2; }
.curves { pointer-events: none; }
</style>
