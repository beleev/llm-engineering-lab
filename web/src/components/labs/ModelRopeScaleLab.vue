<!--
  RoPE 长度外推实验台 (对应 llm_models/layers/core/position_encoding.py:scaled_inv_freq)。
  只讲一件事: 低频维度在训练长度内连一圈都没转完, 位置一超长它们就看到 “没见过的角度”;
  PI / NTK-aware / YaRN 是三种把角度压回训练范围的办法, 代价各不相同。公式与 Python 逐项一致。
-->
<template>
  <LabFrame
    title="RoPE 长度外推 — 哪些频率该压缩, 哪些该放着不动"
    sub="- 模拟什么: 旋转位置编码 (RoPE) 将 token 位置索引 m 映射为一组二维平面旋转角: 将 q 与 k 向量每相邻两维视作一个二维平面, 第 i 个平面按固有角频率 $\theta_i$ 旋转 $m \cdot \theta_i$ 弧度。高频分量旋转快, 负责刻画局部相邻 token 的先后次序; 低频分量旋转慢, 负责承载长距离相对位置信息。当模型在上下文长度 $L = 2048$ 上完成预训练后, 一旦推理位置超过 2048, 低频分量便会旋转至训练期从未见过的角度区间, 导致注意力得分失真。PI、NTK-aware 与 YaRN 是三种将超长位置旋转角映射回训练分布范围的外推缩放方案, 实验公式与 llm_models/layers/core/position_encoding.py 中的 scaled_inv_freq 逐项对应 ($\text{d\_head} = 64$, 共 32 个频率分量)。
- 图上是什么: 左侧为各频率波长柱状图, 右侧为单频率相位表盘:
  - 左侧柱图: 每根柱子代表一个频率分量 i, 从左 (高频) 至右 (低频) 排列; 柱高表示该分量的波长, 即旋转满一周所需的 token 跨度 (对数纵轴)。
  - 柱上短横线: 未缩放时的原始波长基准; 若柱高高于横线, 说明该频率已被拉长波长 (即压缩角频率)。
  - 两条水平虚线: 分别标示原始训练长度 2048 与扩展后的目标长度 $2048 \times s$。
  - 红色柱子: 表示该频率在训练期内尚未旋转满一周, 且当前查询位置 m 已使其旋转至训练期未覆盖的越界角度。
  - 底部位置标尺: 指示当前 token 位置 m, 着色区段标示原始训练覆盖范围 (0 至 2048)。
  - 右侧相位表盘: 展示当前选中频率分量的旋转状态; 粗圆弧为训练期内遍历过的角度范围, 指针为当前位置 m 处的旋转角, 指针偏出粗圆弧即代表发生角度越界。
- 可以做什么: 按界面控件顺序操作:
  - 顶部方案按钮: 切换 4 种位置编码方案 (不缩放 / PI / NTK-aware / YaRN)。
  - 拖动「扩展倍数 s」: 设定目标上下文窗口相对于原始训练长度的扩展倍率。
  - 拖动「token 位置 m」或直接拖动底部标尺上的圆点: 将观测位置移至 2048 以外的超长区间。
  - 点击任意频率柱: 切换右侧表盘所展示的频率分量 (聚焦柱图后也可按键盘左右方向键切换)。
- 应该看到: 依次切换四种方案并对照右侧读数:
  - 不缩放时: 位置 m 一旦超出 2048, 右侧最低频的 11 根柱子立即变红, 因为这 11 个低频分量在训练长度内均未转满一周。
  - 切换至 PI: 红柱全部消除, 但「相邻 token 分辨率」降至 $1/s$。
  - 切换至 NTK-aware: 相邻分辨率保持 100%, 但中低频区段仍残留部分红柱。
  - 切换至 YaRN: 高频分量保持不变、低频分量压缩 s 倍, 红柱全部消除且相邻分辨率维持 100%, 同时额外给出 mscale 注意力温度补偿读数。
  - 「越界的维度数」: 当前位置下变红的频率柱数量。
  - 「相邻 token 分辨率」: 最高频分量缩放后保留的角频率比例, 100% 表示相邻 token 之间的旋转角差不变, 模型仍能精细区分紧邻位置。
  - 「完全没被改动的频率数」与「最低频被压缩倍数」: 分别量化各方案对高频端与低频端的改动幅度。"
    module="llm_models/layers/core/position_encoding.py"
    run="python -m llm_models.run_models.foundation.rope_scaling.infer_rope_scaling"
    :challenge="{
      ask: '将扩展倍数拖至 8, 并将位置 m 拖至目标长度 16384, 依次切换四种方案观察。请先思考: 哪种方案能够消除全部红柱但「相邻 token 分辨率」最低? 哪种方案保持了满分分辨率却仍残留越界红柱?',
      answer: '答案: PI 消除了全部红柱但相邻分辨率最低; NTK-aware 保持了 100% 分辨率却仍残留红柱; YaRN 则同时兼顾了零越界与满分局部精度。\n- PI (位置内插): 将全部频率统一除以扩展倍数 8, 等价于将位置索引 m 线性压缩为 $m/8$, 使所有平面的旋转角严格落回训练区间, 因此不产生任何红柱。但最高频分量同样被除以 8, 导致相邻两个 token 的转角差缩小为原来的 $1/8$, 局部相对位置变得模糊, 模型无法区分紧邻与相隔数格的 token, 必须经过后续微调才能恢复局部辨识度。\n- NTK-aware: 不直接等比缩放频率, 而是放大 RoPE 公式中的基数 base, 使缩放倍率随维度平滑分布:\n  - 最高频分量保持不变 (相邻分辨率维持 100%)。\n  - 最低频分量恰好压缩 8 倍。\n  - 中间维度按指数规律过渡。然而中低频区段内那些训练期本就未转满一周的维度仅被压缩了 $8^{2i/(d-2)} &lt; 8$ 倍, 压缩幅度不足以抵消 8 倍长度扩展, 旋转角依然超出训练范围, 因此残留红柱。\n- YaRN: 同时保住两端特性, 既无红柱又维持 100% 相邻分辨率。它按各维度在训练长度内旋转的圈数 $r$ 进行分段处理:\n  - $r &gt; 32$ 圈的高频维度: 完全不缩放、直接外推 (训练期已完整覆盖全部 $[0, 2\\pi)$ 角度)。\n  - $r &lt; 1$ 圈的低频维度: 按 PI 方式除以 $s$ 线性内插。\n  - 介于两者之间的中频维度: 在不缩放与线性内插之间平滑线性过渡。\n此外, YaRN 还将注意力 logit 乘以温度补偿系数 $\\text{mscale} = 0.1 \\cdot \\ln s + 1$, 用于抵消上下文大幅拉长后 softmax 分布整体趋于平坦所导致的熵增。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <LabSlider v-model="s" label="扩展倍数 s (目标长度 = 2048 × s)" :min="1" :max="16" unit="×" />
      <LabSlider v-model="pos" label="token 位置 m" :min="0" :max="PMAX" :step="64" />
    </template>

    <div class="wrap">
      <svg ref="svg" viewBox="0 0 520 330" class="chart" role="group" aria-label="各频率波长">
        <template v-for="e in [1, 2, 3, 4, 5, 6]" :key="e">
          <line :x1="X0" :x2="X1" :y1="py(e)" :y2="py(e)" class="grid" /><text :x="X0 - 5" :y="py(e) + 3" class="yl">1e{{ e }}</text>
        </template>
        <!-- 32 根柱子只占一个 Tab 停靠点, 左右方向键换选中的那根 -->
        <g class="bars" tabindex="0" role="slider" aria-label="选中的频率编号, 左右方向键切换" aria-valuemin="0" :aria-valuemax="NF - 1" :aria-valuenow="sel"
          :aria-valuetext="`频率 ${sel}, ${cur.ood ? '当前角度从未见过' : '当前角度在见过的范围内'}`"
          @keydown.left.prevent="sel = Math.max(0, sel - 1)" @keydown.right.prevent="sel = Math.min(NF - 1, sel + 1)">
          <g v-for="f in freqs" :key="f.i" class="barg" @click="sel = f.i">
            <rect :x="bx(f.i)" :y="py(f.lw)" :width="BW" :height="Y1 - py(f.lw)" class="bar" :class="{ ood: f.ood, sel: sel === f.i }" />
            <line :x1="bx(f.i) - 2" :x2="bx(f.i) + BW + 2" :y1="py(f.lw0)" :y2="py(f.lw0)" class="orig" />
            <text v-if="f.i % 4 === 0 || f.i === NF - 1" :x="bx(f.i) + BW / 2" :y="Y1 + 12" class="xl">{{ f.i }}</text>
          </g>
        </g>
        <line :x1="X0" :x2="X1" :y1="py(Math.log10(L))" :y2="py(Math.log10(L))" class="train" />
        <text :x="X0 + 4" :y="py(Math.log10(L)) - 4" class="tl">训练长度 L = 2048</text>
        <line v-if="s > 1" :x1="X0" :x2="X1" :y1="py(Math.log10(L * s))" :y2="py(Math.log10(L * s))" class="target" />
        <text v-if="s > 1" :x="X0 + 4" :y="py(Math.log10(L * s)) - 4" class="tg">目标长度 {{ L * s }}</text>
        <text :x="X0" :y="Y1 + 26" class="cap">频率编号 i (左 = 高频, 管局部顺序; 右 = 低频, 管远距离) · 横线 = 原始波长</text>
        <!-- 位置标尺: 直接拖 -->
        <line :x1="X0" :x2="X1" y1="306" y2="306" class="ruler" />
        <rect :x="X0" y="302" :width="px(L) - X0" height="8" class="seen" />
        <text :x="px(L)" y="326" class="xl">2048</text><text :x="px(L * s)" y="326" class="xl" v-if="s > 1">{{ L * s }}</text>
        <g class="draggable" tabindex="0" role="slider" aria-label="拖动 token 位置" :aria-valuenow="pos"
          @pointerdown="start($event, { svg, onMove })" @keydown.left.prevent="pos = clamp(pos - 256, 0, PMAX)" @keydown.right.prevent="pos = clamp(pos + 256, 0, PMAX)">
          <rect :x="px(pos) - 14" y="290" width="28" height="32" fill="transparent" />
          <circle :cx="px(pos)" cy="306" r="8" class="handle" :class="{ bad: nOod > 0 }" />
        </g>
      </svg>

      <svg viewBox="-70 -78 140 160" class="dial" role="img" :aria-label="`频率 ${sel} 的角度表盘`">
        <circle r="50" class="ring" />
        <path :d="arc" class="seenarc" />
        <line x1="0" y1="0" :x2="50 * Math.cos(ang)" :y2="-50 * Math.sin(ang)" class="hand" :class="{ bad: cur.ood }" />
        <circle :cx="50 * Math.cos(ang)" :cy="-50 * Math.sin(ang)" r="5" class="tip" :class="{ bad: cur.ood }" />
        <text y="-64" class="dt">频率 {{ sel }} · 训练期转 {{ cur.turns < 10 ? cur.turns.toFixed(2) : cur.turns.toFixed(0) }} 圈</text>
        <text y="76" class="dt">{{ cur.ood ? '当前角度从未见过' : '当前角度在见过的范围内' }}</text>
      </svg>
    </div>

    <template #stats>
      <div class="kv"><span>位置 m={{ pos }} 处越界的维度数 (红柱)</span><b :class="nOod ? 'bad' : 'good'">{{ nOod }} / {{ NF }}</b></div>
      <div class="kv"><span><Tex text="相邻 token 分辨率 $\theta'_0/\theta_0$ (最高频还剩多少)" /></span><b :class="hi < 0.6 ? 'bad' : 'good'">{{ (hi * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>完全没被改动的频率数</span><b>{{ untouched }} / {{ NF }}</b></div>
      <div class="kv"><span>最低频被压缩了几倍</span><b>{{ lo.toFixed(2) }}×</b></div>
      <div class="lab-note">
        <p><Tex :text="MODES.find((m) => m.id === mode).note" /></p>
        <p v-if="mode === 'yarn'">此刻注意力温度补偿 mscale = {{ mscale.toFixed(4) }} (logit 乘这个数); 其余三种方案恒为 1。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const D = 64, NF = D / 2, BASE = 10000, L = 2048, PMAX = 32768, BETA_FAST = 32, BETA_SLOW = 1 // 与 infer_rope_scaling 同设定
const MODES = [
  { id: 'none', label: '不缩放 (直接外推)', note: '直接外推: 高频维度转过无数圈, 什么角度都见过, 没事。低频维度训练时只走过一小段弧, 超长后走到弧外, 注意力分数失真, 困惑度急剧上升。' },
  { id: 'pi', label: 'PI 位置内插', note: "PI: $\\theta' = \\theta/s$, 等价于把位置 $m$ 压成 $m/s$。所有角度回到训练范围, 但高频也被压, 局部顺序信息被抹糊。Python 的 scaled_inv_freq 没有这一支, 这里只作对照。" },
  { id: 'ntk', label: 'NTK-aware', note: "NTK-aware: $\\text{base}' = \\text{base} \\cdot s^{d/(d-2)}$。$\\theta_0$ 不变、$\\theta_{\\text{last}}$ 恰好 $\\div s$, 中间按指数过渡。免微调就能用, 但中段低频压得不够。" },
  { id: 'yarn', label: 'YaRN', note: 'YaRN (NTK-by-parts): 按训练期圈数 $r$ 分段, $r>32$ 不动, $r<1$ 按 PI $\\div s$, 中间线性混合; 另乘 mscale 调 softmax 温度。' },
]
const mode = ref('none'), s = ref(4), pos = ref(8192), sel = ref(26)
const svg = ref(null)
const { start } = useDrag()

// ★ 与 scaled_inv_freq 同式
const theta0 = range(D / 2).map((i) => BASE ** (-(2 * i) / D))
const theta = computed(() => theta0.map((t, i) => {
  if (mode.value === 'none' || s.value === 1) return t
  if (mode.value === 'pi') return t / s.value
  if (mode.value === 'ntk') return (BASE * s.value ** (D / (D - 2))) ** (-(2 * i) / D)
  const r = (L * t) / (2 * Math.PI)
  const g = clamp((r - BETA_SLOW) / (BETA_FAST - BETA_SLOW), 0, 1)
  return ((1 - g) * t) / s.value + g * t
}))
const mscale = computed(() => (mode.value === 'yarn' && s.value > 1 ? 0.1 * Math.log(s.value) + 1 : 1))

const freqs = computed(() => theta.value.map((t, i) => {
  const seen = L * theta0[i]                       // 训练期转过的最大弧度
  return {
    i, lw: Math.log10((2 * Math.PI) / t), lw0: Math.log10((2 * Math.PI) / theta0[i]),
    turns: seen / (2 * Math.PI), seen,
    ood: seen < 2 * Math.PI && pos.value * t > seen * 1.0001, // 没转满一圈, 且当前角度超出见过的弧
  }
}))
const nOod = computed(() => freqs.value.filter((f) => f.ood).length)
const hi = computed(() => theta.value[0] / theta0[0])
const lo = computed(() => theta0[D / 2 - 1] / theta.value[D / 2 - 1])
const cur = computed(() => freqs.value[sel.value])
const ang = computed(() => (pos.value * theta.value[sel.value]) % (2 * Math.PI))
const arc = computed(() => {
  const a = Math.min(cur.value.seen, 2 * Math.PI - 1e-3)
  return `M 50 0 A 50 50 0 ${a > Math.PI ? 1 : 0} 0 ${(50 * Math.cos(a)).toFixed(2)} ${(-50 * Math.sin(a)).toFixed(2)}`
})

const X0 = 40, X1 = 510, Y0 = 12, Y1 = 250, BW = 10
const py = (lg) => Y1 - ((clamp(lg, 0.5, 6) - 0.5) / 5.5) * (Y1 - Y0)
const bx = (i) => X0 + 6 + i * ((X1 - X0 - 12) / NF)
// YaRN 下 γ=1 (训练期转够 32 圈) 的高频维度: 完全不动
const untouched = computed(() => theta.value.filter((t, i) => Math.abs(t / theta0[i] - 1) < 1e-9).length)
const px = (m) => X0 + (m / PMAX) * (X1 - X0)
const onMove = ({ x }) => { pos.value = clamp(Math.round((((x - X0) / (X1 - X0)) * PMAX) / 64) * 64, 0, PMAX) }
</script>

<style scoped>
.wrap { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.chart { flex: 1 1 380px; min-width: 0; }
.dial { flex: 0 0 150px; width: 150px; }
.grid { stroke: var(--border); }
.yl { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 9px; fill: var(--text-dim); }
.cap { font-size: 10px; fill: var(--text-dim); }
.bars { outline: none; }
.barg { cursor: pointer; }
.bar { fill: color-mix(in srgb, var(--left) 45%, transparent); stroke: var(--left); }
.bar.ood { fill: color-mix(in srgb, var(--danger) 45%, transparent); stroke: var(--danger); }
.bar.sel { stroke: var(--accent); stroke-width: 2.5; }
.bars:focus-visible .bar.sel { stroke: var(--text); stroke-width: 3; }
.orig { stroke: var(--text-muted); stroke-width: 1.5; }
.train { stroke: var(--eye); stroke-dasharray: 5 4; }
.tl { text-anchor: start; font-size: 10px; fill: var(--eye); }
.target { stroke: var(--accent); stroke-dasharray: 2 3; }
.tg { text-anchor: start; font-size: 10px; fill: var(--accent); }
.ruler { stroke: var(--border-strong); stroke-width: 2; }
.seen { fill: color-mix(in srgb, var(--left) 50%, transparent); }
.handle { fill: var(--left); stroke: var(--bg); stroke-width: 2; }
.handle.bad { fill: var(--danger); }
.ring { fill: none; stroke: var(--border-strong); }
.seenarc { fill: none; stroke: var(--left); stroke-width: 7; stroke-linecap: round; }
.hand { stroke: var(--left); stroke-width: 2; }
.hand.bad, .tip.bad { stroke: var(--danger); fill: var(--danger); }
.tip { fill: var(--left); }
.dt { text-anchor: middle; font-size: 9px; fill: var(--text-muted); }
</style>
