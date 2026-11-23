<!--
  学习率调度 (对应 llm_train/m10_training_stability:warmup_cosine_lr / wsd_lr)。
  只讲一件事: cosine 的每一步 LR 都依赖总步数, 想加训就得从头来; WSD 的稳定段与总步数无关, 任何稳定段的 checkpoint 都能接着训。
-->
<template>
  <LabFrame
    title="学习率调度 Cosine vs WSD — 训到一半想加训, 要回退多少步重训?"
    sub="- 模拟什么: 学习率 (lr) 控制每一步参数更新的步长, 训练过程中通常沿预设调度曲线动态变化: 训练初期从极小值线性爬升至峰值 (即 warmup 阶段, 因为 Adam 的二阶矩估计在最初几百步方差极大, 直接使用大步长极易导致训练发散), 训练后期再逐步衰减以使参数稳定收敛至损失谷底。本实验台模拟「原计划训练 $T_0$ 步, 到达终点后希望追加预算继续训练至 $T_1$ 步」的工程场景: 在已完成的步数中, 学习率轨迹与「从一开始便按总步数 $T_1$ 规划」完全一致的区间可直接复用; 学习率不一致的区间则必须回退至分叉点前的 checkpoint 重新训练。对比两种主流调度策略:
  - cosine 调度: 将 warmup 结束后每一步的 lr 定义为「当前步数除以总步数」的余弦函数, 一旦修改总步数, 整条衰减曲线将全局改变。
  - WSD (warmup–stable–decay) 调度: 在 warmup 后长时间保持峰值 lr 恒定不变, 仅在末尾约 10% 的步数内执行快速退火 (decay), 其稳定段公式完全不依赖总步数。
- 图上是什么: 横轴为训练步数 (范围为 0 至 2000), 纵轴为当前 lr 相对峰值 lr 的归一化比例。
  - 实线表示按原计划总步数 $T_0$ 执行的 lr 轨迹。
  - 虚线表示若从第 0 步起即按新计划总步数 $T_1$ 规划所对应的目标 lr 轨迹。
  - 红色阴影标出已训练区间内实线与虚线发生偏离的部分, 阴影越宽代表需要废弃重训的算力浪费越多。
  - 横轴下方设有三个可拖拽彩色手柄, 分别对应 warmup 结束步数、原计划总步数 $T_0$ 与追加后总步数 $T_1$。
- 可以做什么: 控件从上到下依次为:
  - 点击「Warmup + Cosine」或「WSD」切换学习率调度器。
  - 左右拖动横轴下方的三个手柄 (或聚焦后使用方向键以 10 步为单位微调), 修改 warmup 长度、原计划步数与新计划总步数。
  - 在 WSD 模式下拖动「退火占比」滑杆调整末尾退火段占总步数的比例。
  - 拖动「最终 LR / 峰值 LR」调整退火终点的最低学习率比例。
- 应该看到: 在 cosine 调度下, warmup 一结束实线与虚线便立即分叉, 红色阴影几乎覆盖整个已训练区间; 在 WSD 调度下, 两条曲线在漫长的稳定段完全重合, 红色阴影仅局限于原计划末尾的短退火段。右侧读数含义如下:
  - 「仍算数的」: 已完成的 $T_0$ 步中学习率与新计划严格一致的可复用步数 (从第 0 步起累计至首次偏离处为止)。
  - 「需要回退重训」: 等于 $T_0$ 减去可复用步数, 当一半以上已训步数可保留时显示绿色。
  - 末行读数: 对比第 $T_0$ 步实际使用的 lr 与新计划在该步本应取用的目标 lr。"
    module="llm_train/m10"
    run="python -m llm_train.m10_training_stability.demo"
    :challenge="{
      ask: '原计划训练 1000 步, 训完后观察到 loss 仍在持续下降, 希望追加预算延长至 1600 步。采用 cosine 与 WSD 调度时各自需要回退多少步重新训练?',
      answer: '结论:\n- cosine 调度需回退 900 步, 除 warmup 外几乎全部重训。\n- WSD 调度仅需回退末尾 100 步。\n机制解析:\n- cosine 调度: warmup 结束后每一步的学习率均由 $f((\\text{step} - \\text{warmup}) / (\\text{总步数} - \\text{warmup}))$ 决定。当总步数从 1000 改为 1600 时分母随之改变, 导致 warmup 后同一步数对应的 lr 全部失配 (m10 实测将总步数从 100 改为 200 时, 前 90 步的 lr 最大偏差达 5.96e-4)。因此已完成的 1000 步中仅有前 100 步 warmup 可直接复用, 右侧读数显示需回退 900 步。若不愿回退重训, 只能从第 1000 步的 checkpoint 强行将学习率重新拉高 (re-warmup), 但突变的大步长会导致已收敛的训练 loss 先出现显著反弹。\n- WSD 调度: 稳定段始终保持峰值 lr 不变, 计算公式完全不含总步数项, 因此无论计划总步数如何设定, 前 900 步 (原退火段起点之前) 的学习率轨迹完全一致, 需要废弃的仅为原计划末尾的 100 步退火段。续训流程分为三步:\n  1. 回退至原退火段起点的 checkpoint (即第 900 步)。\n  2. 维持峰值 lr 继续在稳定段训练至第 1440 步。\n  3. 在最后 10% 区间 (即末尾 160 步) 执行退火收敛即可。\n这正是 MiniCPM、DeepSeek-V3 与 Kimi 等大模型普遍采用 WSD 调度的工程动因: 主干训练全程维持峰值 lr 持续推进, 任意时刻均可从主干 checkpoint 分叉出一段短退火分支以产出正式模型, 追加预训练语料或开展多规模对比实验均无需重跑主干。退火起点前的稳定段 checkpoint 即为所有下游退火分支的公共祖先。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'cosine' }" @click="mode = 'cosine'">Warmup + Cosine</button>
        <button type="button" :class="{ active: mode === 'wsd' }" @click="mode = 'wsd'">WSD (Warmup-Stable-Decay)</button>
      </div>
      <LabSlider v-if="mode === 'wsd'" v-model="decayFrac" label="WSD 退火占比" :min="0.05" :max="0.4" :step="0.05" :format="(v) => Math.round(v * 100) + '%'" />
      <LabSlider v-model="minRatio" label="最终 LR / 峰值 LR" :min="0" :max="0.3" :step="0.05" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="group" aria-label="学习率随训练步数的变化, 三个手柄可拖动">
      <line :x1="X0" :x2="W - 10" :y1="py(0)" :y2="py(0)" class="axis" />
      <g v-for="t in [0, 500, 1000, 1500, 2000]" :key="t">
        <line :x1="px(t)" :x2="px(t)" :y1="py(0)" :y2="py(0) + 4" class="axis" />
        <text :x="px(t)" :y="py(0) + 15" class="tick" text-anchor="middle">{{ t }}</text>
      </g>
      <text :x="X0 - 6" :y="py(1) + 3" class="tick" text-anchor="end">峰值</text>
      <line :x1="X0" :x2="W - 10" :y1="py(1)" :y2="py(1)" class="gridl" />
      <polygon :points="gapPoly" class="gap" />
      <polyline :points="poly(t1)" class="ideal" />
      <polyline :points="poly(t0)" class="plan" />
      <g v-for="h in handles" :key="h.key" class="draggable" tabindex="0" role="slider" :aria-label="h.label" :aria-valuenow="h.v"
        @pointerdown="start($event, { svg, onMove: ({ x }) => h.set(Math.round(pxInv(x) / 10) * 10) })"
        @keydown.right.prevent="h.set(h.v + 10)" @keydown.left.prevent="h.set(h.v - 10)">
        <line :x1="px(h.v)" :x2="px(h.v)" :y1="py(1) - 8" :y2="py(0)" :stroke="h.color" stroke-dasharray="2 3" />
        <rect :x="px(h.v) - 9" :y="py(0) + 20" width="18" height="18" rx="4" :fill="h.color" />
        <text :x="px(h.v)" :y="py(0) + 52" class="hl" text-anchor="middle" :fill="h.color">{{ h.short }} {{ h.v }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>已训 {{ t0 }} 步里仍「算数」的 (lr 和新计划一致的步数)</span><b :class="keepFrac > 0.5 ? 'good' : 'bad'">{{ keep }} 步</b></div>
      <div class="kv"><span>加训到 {{ t1 }} 步需要回退重训的</span><b :class="keepFrac > 0.5 ? 'good' : 'bad'">{{ t0 - keep }} 步</b></div>
      <div class="kv"><span>第 {{ t0 }} 步的 lr (相对峰值): 实际用的 vs 按新计划本该用的</span><b>{{ lr(t0 - 1, t0).toFixed(2) }} vs {{ lr(t0 - 1, t1).toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>「仍算数」的统计口径: 从第 0 步起逐一比对实线与虚线的 lr, 一旦首次出现数值不一致即停止累计; 因为从分叉步开始模型优化轨迹已偏离新计划, 后续即使单点数值偶合也无法等价衔接。</p>
        <p>cosine 在 warmup 之后每一步都把 total_steps 放在分母里, 总步数一改整条曲线都变。★ WSD 的稳定段 <code class="inline">return base_lr</code> 里没有 total_steps, 只有退火段依赖它。</p>
        <p>两种调度器开头设置 warmup 的作用一致: Adam 的二阶矩估计在训练最初几百步统计样本不足、误差较大, 先以小学习率平稳度过梯度剧烈波动的初始阶段。</p>
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

// 默认 cosine: 先看到「加训要回退 900 步」, 再切 WSD 看它怎么治
const mode = ref('cosine'), warm = ref(100), t0 = ref(1000), t1 = ref(1600), decayFrac = ref(0.1), minRatio = ref(0.1), svg = ref(null)
const { start } = useDrag()
const XMAX = 2000

// 与 Python 逐行相同 (base_lr = 1)
const lr = (step, total) => {
  const w = warm.value, mr = minRatio.value
  if (step < w) return (step + 1) / w
  if (mode.value === 'cosine') {
    const p = (step - w) / Math.max(1, total - w)
    return mr + (1 - mr) * 0.5 * (1 + Math.cos(Math.PI * p))
  }
  const decayStart = total - Math.floor(total * decayFrac.value)
  if (step < decayStart) return 1 // ★ 稳定段: 与 total 无关
  return 1 - (1 - mr) * ((step - decayStart) / Math.max(1, total - decayStart))
}

// 已训过的 [0, T0) 里, 有多少步的 LR 与 "按 T1 规划" 的一致 (从头数, 第一次不一致就停)
const keep = computed(() => {
  let s = 0
  while (s < t0.value && Math.abs(lr(s, t0.value) - lr(s, t1.value)) < 1e-9) s++
  return s
})
const keepFrac = computed(() => keep.value / t0.value)

const W = 620, H = 250, X0 = 40
const px = (s) => X0 + (s / XMAX) * (W - X0 - 14)
const pxInv = (x) => ((x - X0) / (W - X0 - 14)) * XMAX
const py = (v) => 22 + (1 - v) * (H - 90)
const pts = (total) => range(Math.floor(total / 5) + 1).map((i) => Math.min(i * 5, total - 1)).map((s) => [px(s), py(lr(s, total))])
const poly = (total) => pts(total).map((p) => p.join(',')).join(' ')
const gapPoly = computed(() => {
  const ss = range(Math.floor(t0.value / 5) + 1).map((i) => Math.min(i * 5, t0.value - 1))
  const top = ss.map((s) => `${px(s)},${py(lr(s, t1.value))}`)
  const bot = ss.map((s) => `${px(s)},${py(lr(s, t0.value))}`).reverse()
  return [...top, ...bot].join(' ')
})

const handles = computed(() => [
  { key: 'w', short: 'warmup', label: 'warmup 结束步', v: warm.value, color: 'var(--eye)', set: (v) => (warm.value = clamp(v, 10, t0.value - 100)) },
  { key: 't0', short: 'T₀', label: '原计划总步数', v: t0.value, color: 'var(--accent)', set: (v) => (t0.value = clamp(v, warm.value + 100, t1.value)) },
  { key: 't1', short: 'T₁', label: '新计划总步数', v: t1.value, color: 'var(--left)', set: (v) => (t1.value = clamp(v, t0.value, XMAX)) },
])
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.gridl { stroke: var(--border); stroke-dasharray: 2 4; }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.plan { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.ideal { fill: none; stroke: var(--left); stroke-width: 2; stroke-dasharray: 6 4; }
.gap { fill: var(--danger); opacity: 0.25; }
.hl { font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
</style>
