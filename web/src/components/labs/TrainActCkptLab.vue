<!--
  激活重算 (对应 llm_train/m07_activation_checkpointing:run)。
  只讲一件事: 峰值 ≈ L/k (常驻的段边界) + k (当前段重算出来的瞬时激活), k≈√L 时最小。
-->
<template>
  <LabFrame
    title="激活重算 — 只存段边界、反向时重跑, 段长 k 怎么选?"
    sub="- 模拟什么: 反向传播计算第 i 层的参数梯度时, 必须用到前向阶段第 i 层的输入激活张量。标准训练将全部 L 层的输入激活连同末端输出一并保存在显存中, 共需驻留 $L + 1$ 份激活 (本实验按每层边界激活计为 1 份单位显存)。激活重算 (activation checkpointing) 仅在每隔 k 层的分段边界处常驻保存一份检查点激活, 当反向传播回退至某一段时, 再从该段入口检查点重新执行一次段内前向计算, 临时恢复该段各层的内部激活并在求导后立即释放。由此整个训练步的峰值激活显存约为 $L / k$ (常驻段边界数) 加上 $k$ (当前正在回传那一段内的临时重算激活数), 当 $k$ 取 $\sqrt{L}$ 附近时达到理论最小值; 付出的代价是全网约多执行一次前向传播, 而计算所得梯度与全量缓存方案逐位严格一致, 本质是以额外算力换取显存空间。
- 图上是什么: 上方为显存峰值随分段长度变化的曲线图, 下方为单步执行期间各层激活驻留状态序列:
  - 上方曲线图: 横轴为分段长度 k, 纵轴为单步训练期间的峰值激活份数 (即任意时刻显存中同时存在的激活张量最大份数)。
  - 曲线与标记: 实线描绘各 k 值对应的实测峰值, 绿色竖直虚线标示理论最优点 $\sqrt{L}$, 黄色圆点代表当前选定的分段长度 k。
  - 下方状态格: 依次对应 a0 至 aL 共 $L + 1$ 个位置的激活张量 ($a_i$ 为第 i 层输入激活, aL 为末层输出)。
  - 格子颜色 (对应图例): 深色表示前向阶段持久化保存的段边界检查点, 亮色表示反向阶段在当前段内临时重算出的激活, 灰色表示当前时刻未占用显存; 带高亮描边的格子代表当前帧正在执行前向或反向计算的层位置。
- 可以做什么: 自上而下操作各控件:
  - 拖动「层数 L」与「段长 k」调节网络深度与检查点间隔 ($k = 1$ 等价于逐层全存, $k = L$ 等价于仅保存网络总输入); 也可直接左右拖动上方曲线上的黄色圆点修改 k。
  - 使用帧播放器逐帧观察一次完整前向加反向过程中各层激活的分配与释放轨迹: 前向阶段仅段边界依次点亮留存, 反向阶段每进入一段先批量重算点亮段内亮色格, 再随梯度逐层回传依次释放。
  - 每次调整参数后, 播放器会自动定位至显存占用达到峰值的那一帧。
- 应该看到: 两个极端配置 $k = 1$ 与 $k = L$ 的激活显存峰值完全相同, 均高达 $L + 1$ 份: 逐层全存自然占满全部激活, 而仅存网络首层输入时, 反向第一步就必须一次性重算出整条 L 层链路的全部内部激活, 峰值同样回升至 $L + 1$ 份。右侧读数含义如下:
  - 「此刻活着的激活」: 当前播放帧显存中实际驻留的激活总份数。
  - 「峰值」: 完整前向与反向全过程中的最大并发激活份数, 达到全局最优时呈绿色、接近全存基线时呈红色。
  - 「最优 k」: 当前层数 L 下使峰值显存最小的理想分段长度。
  - 底部开销读数: 统计全步累计调用的前向层次数 (不重算基线为 L 次, 启用重算后额外多算 $L - L / k$ 次) 以及折算后的总训练算力增幅百分比 (按单层前向与反向计算量之比 1:2 折算, 理论额外算力上限为 +33%)。"
    module="llm_train/m07"
    run="python -m llm_train.m07_activation_checkpointing.demo"
    :challenge="{
      ask: '设网络深度 $L = 36$ 层, 对比 $k = 1$ (逐层全存)、$k = 6$ 与 $k = 36$ (仅存首层输入) 三种配置: 各自的峰值激活显存分别是多少份? 哪一种配置最节省显存?',
      answer: '答案: $k = 1$ 与 $k = 36$ 的峰值激活均为 37 份, 而 $k = 6$ 仅需 12 份, 显存占用最低。\n- $k = 1$ (逐层全存): 前向阶段将 a0 至 a36 全部保留在显存中, 峰值占用 37 份, 反向阶段无需任何重算, 作为标准基线。\n- $k = 36$ (仅存输入): 前向阶段虽然仅留存了首尾边界 a0 与 a36 两份激活, 但反向传播第一步计算第 35 层梯度时即依赖输入激活 a35, 必须从 a0 出发连续重跑前 35 层前向计算, 期间产生的临时激活 a1 至 a35 必须同时驻留显存以供该段回传使用, 导致峰值显存重新回到 37 份, 额外付出了一次完整前向计算却未能降低显存峰值。\n- $k = 6 = \\sqrt{L}$: 前向阶段每隔 6 层保存 a0、a6、a12、a18、a24、a30、a36 共 7 个常驻段边界; 当反向传播回退至最末一段时, 只需从检查点 a30 出发重算段内临时激活 a31 至 a35 共 5 份, 并发显存峰值仅为 $7 + 5 = 12$ 份 (在播放器自动停靠的峰值帧中可直接核对这两部分格子数)。\n理论上峰值激活份数近似服从 $L / k + k$, 其中常驻边界项随 $k$ 增大而递减、段内临时重算项随 $k$ 增大而递增, 两者之和在 $k = \\sqrt{L}$ 处取得极小值。与此同时, 重算引入的额外算力开销几乎与分段长度 $k$ 无关: 无论如何划分段长, 每个分段内除段首检查点外的其余各层均恰好重算一次, 累计多跑的前向层数恒接近一整趟前向传播, 按前向与反向计算量 1:2 折算的总时间开销始终控制在 +33% 以内。因此工程调优时只需围绕最小化显存峰值来选取 $k$, 无需在重算算力上做额外权衡。',
    }"
  >
    <template #controls>
      <LabSlider v-model="L" label="层数 L" :min="4" :max="64" />
      <LabSlider v-model="k" label="段长 k (每 k 层存一次)" :min="1" :max="L" />
      <StepPlayer :stepper="stepper" :label="frame.label" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="峰值激活随段长变化曲线, 曲线上的圆点可拖">
      <line :x1="X0" :x2="W - 10" :y1="H - 24" :y2="H - 24" class="axis" />
      <line :x1="X0" :x2="X0" :y1="10" :y2="H - 24" class="axis" />
      <text :x="X0 - 6" :y="py(L + 1) + 4" class="tick" text-anchor="end">{{ L + 1 }}</text>
      <text :x="X0 - 6" :y="py(best.peak) + 4" class="tick" text-anchor="end">{{ best.peak }}</text>
      <text :x="W - 10" :y="H - 8" class="tick" text-anchor="end">段长 k →</text>
      <text :x="X0 + 4" :y="14" class="tick">峰值激活份数</text>
      <line :x1="px(Math.sqrt(L))" :x2="px(Math.sqrt(L))" :y1="14" :y2="H - 24" class="sqrt" />
      <text :x="px(Math.sqrt(L)) + 4" :y="26" class="sqrt-t">√L = {{ Math.sqrt(L).toFixed(1) }}</text>
      <polyline :points="curve.map((c) => `${px(c.k)},${py(c.peak)}`).join(' ')" class="curve" />
      <circle
        :cx="px(k)" :cy="py(cur.peak)" r="8" class="handle draggable" tabindex="0" role="slider"
        aria-label="段长 k" :aria-valuenow="k"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (k = Math.round(clamp(pxInv(x), 1, L))) })"
        @keydown.right.prevent="k = Math.min(L, k + 1)" @keydown.left.prevent="k = Math.max(1, k - 1)"
      />
    </svg>

    <div class="cells acts" :style="{ gridTemplateColumns: `repeat(${Math.min(L + 1, 22)}, minmax(18px, 1fr))` }">
      <span
        v-for="i in L + 1" :key="i" class="cell"
        :class="[frame.live[i - 1] === 'ckpt' ? 'on' : frame.live[i - 1] === 'tmp' ? 'hot' : 'dim', { now: frame.at === i - 1 }]"
        :title="`a${i - 1}: 第 ${i - 1} 层的输入` + (frame.live[i - 1] ? '' : ' (此刻不在显存里)')"
      >{{ i - 1 }}</span>
    </div>
    <p class="legend">
      <span class="cell on" /> 前向存下的段边界 <span class="cell hot" /> 反向时重算出来的 <span class="cell dim" /> 此刻不在显存里
    </p>

    <template #stats>
      <div class="kv"><span>此刻活着的激活 (当前帧显存里的份数)</span><b>{{ frame.count }}</b></div>
      <div class="kv"><span>峰值 (k = {{ k }} 时整个过程的最大值)</span><b :class="cur.peak === best.peak ? 'good' : cur.peak > L * 0.8 ? 'bad' : ''">{{ cur.peak }} 份</b></div>
      <div class="kv"><span>最优 k = {{ best.k }} 时的峰值</span><b>{{ best.peak }} 份</b></div>
      <div class="kv"><span>前向层调用 {{ cur.fwd }} 次 (不重算是 {{ L }} 次), 多出的计算量 (按前向:反向 = 1:2 折算)</span><b :class="{ bad: cur.extra > 0.3 }">+{{ (cur.extra * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">★ 显存峰值出现在反向传播刚完成最后一段内部重算的时刻: 此时前向保存的全部常驻段边界尚未释放, 同时叠加了最后一段刚重算出的段内临时激活, 合计约 $L / k + k$ 份。在确定性前向算子下, 重算恢复的激活与原始前向激活逐位一致, 反向求得的参数梯度因此也与全量缓存基线逐位相同 (m07 脚本实测最大误差为 0.0), 本质仅是在显存空间与前向算力之间做等价置换。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const L = ref(36), k = ref(6), svg = ref(null)
const { start } = useDrag()
watch(L, (v) => { if (k.value > v) k.value = v })

// 和 Python 的 run() 逐行对应: live 记录每个激活是 ckpt(前向存的) / tmp(重算出的) / 无
const simulate = (Ln, seg, wantFrames) => {
  const live = Array(Ln + 1).fill(null)
  let peak = 0, fwd = 0
  const frames = []
  const snap = (label, at) => {
    const count = live.filter(Boolean).length
    peak = Math.max(peak, count)
    if (wantFrames) frames.push({ label, at, live: [...live], count })
  }
  live[0] = 'ckpt'; snap('输入', 0)
  for (let i = 0; i < Ln; i++) {
    fwd++
    if ((i + 1) % seg === 0 || i + 1 === Ln) live[i + 1] = 'ckpt' // 只在段边界保存
    snap(`前向 层${i}`, i + 1)
  }
  for (let s0 = Math.floor((Ln - 1) / seg) * seg; s0 >= 0; s0 -= seg) {
    const end = Math.min(s0 + seg, Ln)
    for (let i = s0; i < end - 1; i++) { fwd++; live[i + 1] = 'tmp'; snap(`重算 层${i}`, i + 1) } // ★ 瞬时激活也占显存
    for (let i = end - 1; i >= s0; i--) { live[i + 1] = null; snap(`反向 层${i}`, i) }
  }
  return { peak, fwd, frames, extra: (fwd - Ln) / (3 * Ln) }
}

const curve = computed(() => range(L.value).map((i) => ({ k: i + 1, ...simulate(L.value, i + 1, false) })))
const best = computed(() => curve.value.reduce((a, b) => (b.peak < a.peak ? b : a)))
const cur = computed(() => simulate(L.value, k.value, true))
const stepper = useStepper(() => cur.value.frames.length, { interval: 160 })
const frame = computed(() => cur.value.frames[Math.min(stepper.step.value, cur.value.frames.length - 1)])
// 换参数后停在峰值那一帧: 先看到最挤的时刻
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.frames.findIndex((f) => f.count === c.peak) }, { immediate: true })

const W = 460, H = 190, X0 = 40
const px = (kk) => X0 + ((kk - 1) / Math.max(1, L.value - 1)) * (W - X0 - 20)
const pxInv = (x) => 1 + ((x - X0) / (W - X0 - 20)) * (L.value - 1)
const py = (p) => 18 + (1 - p / (L.value + 1)) * (H - 46)
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.sqrt { stroke: var(--left); stroke-dasharray: 4 3; }
.sqrt-t { font-size: 10px; fill: var(--left); }
.handle { fill: var(--warn); stroke: var(--bg-card); stroke-width: 2; }
.acts { margin-top: 12px; }
.acts .cell { min-width: 18px; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 8px; font-size: 11px; color: var(--text-dim); }
.legend .cell { min-width: 14px; width: 14px; height: 14px; }
</style>
