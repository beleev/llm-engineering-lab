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
    sub="- 模拟什么: 大模型只训得起一次, 没法反复试学习率 (lr)。自然的想法是先在一个窄的小模型 (叫 base 模型) 上扫出最好的 lr, 再原样用到宽模型上。标准参数化 (SP, 就是默认的初始化加各层同一个 lr) 下这行不通: 网络越宽, 最优 lr 越小, 搬过去就亏。μP (最大更新参数化) 按宽度改三处: 输出层初始化的标准差多除一个 $m$, 隐藏层和输出层的 lr 各除一个 $m$, 其中 $m$ 是宽度除以 base 宽度。改完之后最优 lr 不再随宽度移动, 这就是「μTransfer」。m20 真训了 3 个宽度 × 10 个 lr × 2 种参数化, 共 60 个小 MLP, 这里只查它的结果表。
      - 图上是什么: 横轴是学习率的 $\log_2$ 值 (−13 到 −4, 每格 lr 翻一倍), 纵轴是训练 loss (对数刻度, 越低越好)。三条折线是宽 32、128、512 的 MLP 在各个 lr 下的 loss, 颜色对应右上角的图例。每条线上的空心圈是这个宽度自己的最优 lr。高亮的那一列是你选中的 lr, 三条线上的实心小点是三个宽度在这个 lr 下的 loss。
      - 可以做什么: 点「SP (标准参数化)」和「μP」切换两种参数化, 三条线会整个换掉。点图上任一列 (或聚焦后用左右方向键), 假设这就是你在 base 模型 (宽 32) 上调出来的 lr; 点「用 base 的最优 lr」直接选中宽 32 的最优列。
      - 应该看到: SP 下三个空心圈从右往左散开 (宽 32 在 −5, 宽 128 在 −7, 宽 512 在 −11), 越宽最优 lr 越小; 切到 μP 三个圈对齐在同一列 −5, 而且同一 lr 下越宽的线越靠下 (越宽越好)。右侧「最优 log₂ lr」列出三个宽度各自的最优点, 三者相同时变绿; 「选的 lr: 宽 512 的 loss」是把你选的 lr 原样用到宽 512 上得到的 loss; 「宽 512 自己的最优」是它在所有 lr 里能达到的最低 loss; 「搬过去亏了」是前者除以后者, 1.00 表示一点没亏, SP 下用 base 的最优 lr 搬过去会亏 4.5 倍。"
    module="llm_train/m20"
    run="python -m llm_train.m20_mup.demo"
    :challenge="{
      ask: 'SP 下宽度 ×16, 最优 lr 应该缩多少? m20 实测缩了多少?',
      answer: '按机制推应该缩 16 倍, 实测缩了 64 倍。\n- 为什么该缩: 原因在 Adam 优化器。Adam 对每个参数的更新量约等于 lr, 和梯度大小几乎无关。隐藏层每个输出单元有 width 个输入权重, width 个方向一致的小更新叠在一起, 下一层激活的变化就 $\\propto \\text{lr} \\cdot \\text{width}$。要让激活的变化不随宽度涨, lr 就得按 $1/\\text{width}$ 缩: 宽 16 倍, lr 缩 16 倍, 在 $\\log_2$ 轴上左移 4 格。\n- 实测: SP 下最优 $\\log_2$ lr 从宽 32 的 −5 移到宽 512 的 −11, 左移 6 格, 缩了 64 倍, 比机制预测的还多。多出来的这部分这组实验分不清: 宽 512 的 SP 折线在 −9 到 −7 之间来回跳 (0.0371、0.0628、0.0358), 是只训 150 步加大 lr 带来的不稳定, lr 网格又是 2 倍一格, 分辨率不够。\n- μP 怎么修: 设 $m = \\text{width}/32$, 隐藏层和输出层的 lr 各除以 $m$, 输出层初始化的标准差也多除一个 $m$ (方差除 $m^2$), 输入层不动。改完三个宽度的最优点都落在 $2^{-5}$, 而且同一 lr 下越宽 loss 越低 (0.0186、0.0101、0.0088)。\n要注意最优点附近很平: 宽 512 在 −7、−6、−5 三格的 loss 是 0.0095、0.0090、0.0088, 几乎一样, 所以 argmin 正好落在 −5 有运气成分。m20 用的是 lr 线性衰减到 0; 换成常数 lr, μP 的最优点会在 −5、−7、−8 之间漂, 迁移没这么干净。',
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
        <p>★ SP 与 μP 的全部差别是 init_and_lrs 里的三处除以 m (m = 宽度 / 32): 输出层初始化的标准差除一次 (方差就是除 m²), 隐藏层的 lr 除一次, 输出层的 lr 除一次。输入层不动, 因为它的输入维度不随宽度变。</p>
        <p>宽 32 是 base 模型, m=1, 除以 1 什么都不变, 所以两种参数化下宽 32 那条线完全相同。</p>
        <p>三条曲线是 m20 demo 真训出来的结果表 (每个点是 150 步 Adam、lr 线性衰减到 0、最后 10 步 loss 的平均), 浏览器里不重跑, 只查表。</p>
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
