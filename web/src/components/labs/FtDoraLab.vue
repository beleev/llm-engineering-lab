<!--
  DoRA 实验台: 权重 = 幅度 m × 方向 V/‖V‖。
  只讲一件事: LoRA 用一个旋钮 (ΔV) 同时改长度和方向, 两者被绑在一起;
  DoRA 让低秩更新 ΔV 只管方向, 另设一个可训标量 m 只管长度 —— 沿径向拖 ΔV 时 DoRA 的结果纹丝不动。
  这里画的是权重矩阵里的一个权重向量 (2 维); 真实 DoRA 对每个向量各存一个 m (本仓库按输出行)。
-->
<template>
  <LabFrame
    title="DoRA — 把权重向量拆成「长度」和「方向」, 各给一个旋钮"
    sub="- 模拟什么: 权重矩阵的每一行 (或每一列) 均可分解为幅度 (模长) 与方向两个几何分量。标准 LoRA 将低秩增量 $\Delta V$ 直接叠加至原权重向量上, 单一增量同时改变模长与方向, 使二者强耦合。DoRA 将二者显式解耦: 低秩增量仅负责调整方向, 同时为每个输出通道独立学习标量幅度参数 $m$, 组合权重写作 $W^\prime = m\cdot V/\|V\|$ (先将 $V = w_0 + \Delta V$ 归一化为单位方向向量, 再乘以幅度标量 $m$)。本实验台抽取单个权重向量并投影至 2 维平面以直观展示其几何解耦关系。
- 图上是什么: 平面坐标系内从原点出发的多支向量箭头及辅助几何参考线:
  - 灰色箭头: 冻结的预训练初始权重向量 $w_0$。
  - 黄色 ★: 全参微调期望收敛的目标权重位置。
  - 紫色圆点: 叠加增量后的中间向量 $w_0 + \Delta V$; 连接 $w_0$ 箭头末端与紫点的紫色线段即代表低秩更新量 $\Delta V$。
  - 粉色箭头: LoRA 的最终输出权重向量, 末端直接与紫色圆点重合。
  - 绿色箭头: DoRA 的最终输出权重向量, 方向沿紫色圆点所在的虚线射线, 模长完全由滑杆 $m$ 决定。
  - 两个虚线同心圆: 灰色圆半径等于初始模长 $\|w_0\|$, 绿色圆半径等于当前幅度 $m$, 绿色箭头末端始终落在绿色圆周上。
- 可以做什么: 支持以下交互操作:
  - 点击三个「目标」预设按钮: 切换黄色目标星号位置, 分别对应纯旋转 35°、纯放大 1.4 倍, 以及旋转 25° 同时缩放至 0.7 倍三种典型微调模式。
  - 拖动紫色圆点 (或聚焦后使用方向键): 改变低秩更新向量 $\Delta V$。
  - 拖动「DoRA 幅度 m」滑杆: 独立调节绿色结果向量的模长。
  - 点击「复位」按钮: 将 $\Delta V$ 归零并将 $m$ 恢复为 $\|w_0\|$, 使三支权重向量重新重合。
- 应该看到: 沿原点出发的虚线射线往复拖动紫色圆点时, 粉色箭头 (LoRA) 随之同步伸缩, 而绿色箭头 (DoRA) 保持静止不变, 因为单位方向归一化消除了径向模长变化。右侧对比表格各项含义如下:
  - 「长度变化 ΔM」: 当前结果向量相对初始权重 $w_0$ 的模长增减量。
  - 「方向变化 ΔD」: 当前结果向量相对初始权重 $w_0$ 的旋转角度。
  - LoRA 的 ΔM 与 ΔD 由同一个紫色圆点位置共同决定; DoRA 的 ΔD 仅取决于紫色圆点所在射线方向, ΔM 仅取决于幅度滑杆 $m$。
  - 「距目标」: 结果向量末端与黄色目标星号之间的欧氏距离, 小于 0.05 时显示为绿色。"
    module="llm_finetune/methods/dora.py"
    run="python -m llm_finetune.run_finetune.dora.train_dora"
    :challenge="{
      ask: '选择「目标: 只转 35°」, 按顺序完成以下探究:\n1. 仅拖动紫色圆点使 LoRA 距目标误差小于 0.05: 紫色圆点必须落在什么位置?\n2. 观察 DoRA 的绿色箭头: 当紫色圆点沿原点出发的目标射线往复移动时, 绿色箭头是否发生变化?\n3. 这一几何现象说明 DoRA 中损失函数对低秩增量 $\\Delta V$ 的梯度具有什么性质?',
      answer: '结论:\n1. 要使 LoRA 的距目标误差小于 0.05, 紫色圆点必须精确落在黄色目标星号邻域内。\n2. 沿原点射线拖动紫色圆点时, DoRA 的绿色箭头完全保持静止。\n3. 这说明损失函数对低秩更新量 $\\Delta V$ 的梯度严格正交于当前向量 $V$, 不含任何径向分量。\n机制解析:\n- LoRA 的耦合约束: LoRA 输出即为 $w_0 + \\Delta V$ 本身, 要实现「纯旋转而不改变模长」, 紫色圆点必须同时满足角度与半径约束而精确命中目标点; 稍有偏移即会引起模长波动, 使幅度变化 $\\Delta M$ 与方向变化 $\\Delta D$ 强耦合。DoRA 原论文实证指出, 标准 LoRA 训练中权重幅度变化与方向变化呈强正相关, 而全参微调中二者近似正交甚至呈负相关。\n- DoRA 的归一化不变量: DoRA 将权重参数化为 $W^\\prime = m\\cdot V/\\|V\\|$, 先对 $V = w_0 + \\Delta V$ 做单位范数归一化再乘以独立标量 $m$。紫色圆点沿射线移动仅改变范数 $\\|V\\|$, 在归一化步骤中被精确抵消, 因此绿色向量保持不变。\n- 梯度的正交投影性质: 对归一化算子求导可知, 损失函数对 $V$ 的梯度会自动投影至与 $V$ 正交的切空间上并乘以缩放因子 $m/\\|V\\|$。这使得受限的低秩参数容量能够全部用于学习方向旋转, 而模长伸缩交由各通道独立的标量 $m$ 承担。\n- 参数与推理开销: 每个通道仅新增 1 个幅度标量, 单层共增加 $d$ 个可训练参数, 相对 LoRA 原有的 $2dr$ 参数量仅增加 $1/(2r)$; 推理部署前同样可将幅度与低秩项完整合并回单一稠密矩阵, 实现零额外推理延迟。本仓库在 $r = 8$ 配置下取 3 个随机种子平均实测: LoRA 留出集 EM 为 0.382, DoRA 提升至 0.522。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(t, k) in TARGETS" :key="k" type="button" :class="{ active: tgt === k }" @click="tgt = k">{{ t.name }}</button>
        <button type="button" @click="reset">复位 (ΔV = 0, m = ‖w₀‖)</button>
      </div>
      <LabSlider v-model="m" label="DoRA 幅度 m" :min="0.5" :max="4.5" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 380" role="group" aria-label="二维权重向量的幅度与方向分解">
      <defs>
        <marker v-for="c in ['text-dim', 'right', 'left']" :id="'ah-' + c" :key="c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" :fill="`var(--${c})`" />
        </marker>
      </defs>
      <circle :cx="OX" :cy="OY" :r="N0 * SC" fill="none" stroke="var(--border-strong)" stroke-dasharray="3 4" />
      <circle :cx="OX" :cy="OY" :r="m * SC" fill="none" stroke="var(--left)" stroke-dasharray="2 5" opacity="0.7" />
      <text :x="OX + 4" :y="OY - m * SC - 4" class="t" fill="var(--left)">半径 m</text>
      <!-- 从原点过紫点的射线: DoRA 只看这条线的方向 -->
      <line :x1="OX" :y1="OY" :x2="px(ray)[0]" :y2="px(ray)[1]" stroke="var(--accent)" stroke-width="0.8" stroke-dasharray="2 3" />
      <line v-for="a in arrows" :key="a.id" :x1="OX" :y1="OY" :x2="px(a.v)[0]" :y2="px(a.v)[1]" :stroke="`var(--${a.c})`" :stroke-width="a.w" :marker-end="`url(#ah-${a.c})`" />
      <line :x1="px(W0)[0]" :y1="px(W0)[1]" :x2="px(v)[0]" :y2="px(v)[1]" stroke="var(--accent)" stroke-width="1.5" />
      <text v-for="a in arrows" :key="'l' + a.id" :x="px(a.v)[0] + 8" :y="px(a.v)[1] + a.dy" class="t" :fill="`var(--${a.c})`">{{ a.id }}</text>
      <text :x="px(target)[0]" :y="px(target)[1] + 6" text-anchor="middle" class="star">★</text>
      <!-- ★ 手放在 ΔV 上 -->
      <circle
        class="draggable" :cx="px(v)[0]" :cy="px(v)[1]" r="9" fill="var(--accent)" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="w₀+ΔV 的位置, 方向键移动" :aria-valuenow="v[0]" aria-valuemin="-1" aria-valuemax="7"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x, y }) => setV((x - OX) / SC, (OY - y) / SC) })"
        @keydown.right.prevent="setV(v[0] + 0.1, v[1])" @keydown.left.prevent="setV(v[0] - 0.1, v[1])"
        @keydown.up.prevent="setV(v[0], v[1] + 0.1)" @keydown.down.prevent="setV(v[0], v[1] - 0.1)"
      />
      <text :x="px(v)[0] + 12" :y="px(v)[1] - 10" class="t" fill="var(--accent)">w₀+ΔV</text>
    </svg>

    <template #stats>
      <table class="cmp">
        <thead><tr><th></th><th>LoRA</th><th>DoRA</th></tr></thead>
        <tbody>
          <tr><td><Tex text="长度变化 $\Delta M$" /></td><td class="mono">{{ sg(lora.dM) }}</td><td class="mono">{{ sg(dora.dM) }}</td></tr>
          <tr><td><Tex text="方向变化 $\Delta D$" /></td><td class="mono">{{ lora.dD.toFixed(1) }}°</td><td class="mono">{{ dora.dD.toFixed(1) }}°</td></tr>
        </tbody>
      </table>
      <div class="kv"><span>LoRA 距目标 (粉箭头尖到黄星)</span><b :class="lora.err < 0.05 ? 'good' : 'bad'">{{ lora.err.toFixed(3) }}</b></div>
      <div class="kv"><span>DoRA 距目标 (绿箭头尖到黄星)</span><b :class="dora.err < 0.05 ? 'good' : 'bad'">{{ dora.err.toFixed(3) }}</b></div>
      <div class="lab-note">
        <ul class="pts">
          <li><b>LoRA:</b> 两行数字由同一个紫点决定, 改一个另一个必然跟着动, 想「只转不伸」就得把紫点放得分毫不差。</li>
          <li><b>DoRA:</b> <Tex text="幅度与方向解耦后, 方向旋转角 $\Delta D$ 仅由紫色圆点所在射线决定, 幅度增量 $\Delta M = m - \|w_0\|$ 仅由幅度滑杆 $m$ 决定, 两组参数互不干扰。" /></li>
        </ul>
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
import { clamp } from '@/utils/labmath.js'

const OX = 90, OY = 330, SC = 72
const W0 = [2.6, 1.0], N0 = Math.hypot(...W0)
const rot = ([x, y], deg) => { const a = deg * Math.PI / 180; return [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)] }
const TARGETS = [
  { name: '目标: 只转 35°', v: rot(W0, 35) },
  { name: '目标: 只放大 1.4×', v: W0.map((x) => x * 1.4) },
  { name: '目标: 转 25° 且缩到 0.7×', v: rot(W0, 25).map((x) => x * 0.7) },
]
const px = ([x, y]) => [OX + x * SC, OY - y * SC]

const V_INIT = [3.2, 1.9]       // 初始就带一点 ΔV: 三支箭头分得开, 一打开就看得到 LoRA 和 DoRA 的差别
const v = ref([...V_INIT])      // w₀ + ΔV
const m = ref(Math.round(N0 * 20) / 20)
const tgt = ref(0)
const svgEl = ref(null)
const { start } = useDrag()

const setV = (x, y) => { const p = [clamp(x, -1, 7.4), clamp(y, -0.5, 4.4)]; if (Math.hypot(...p) > 0.2) v.value = p.map((c) => Math.round(c * 50) / 50) }
const reset = () => { v.value = [...W0]; m.value = Math.round(N0 * 20) / 20 }

const target = computed(() => TARGETS[tgt.value].v)
const unit = computed(() => { const n = Math.hypot(...v.value); return v.value.map((c) => c / n) })
const doraW = computed(() => unit.value.map((c) => c * m.value))      // ★ W′ = m · V/‖V‖
const ray = computed(() => unit.value.map((c) => c * 7.6))

const measure = (w) => ({
  dM: Math.hypot(...w) - N0,
  dD: Math.acos(clamp((w[0] * W0[0] + w[1] * W0[1]) / (Math.hypot(...w) * N0), -1, 1)) * 180 / Math.PI,
  err: Math.hypot(w[0] - target.value[0], w[1] - target.value[1]),
})
const lora = computed(() => measure(v.value))                          // LoRA: W′ = w₀ + ΔV
const dora = computed(() => measure(doraW.value))
const arrows = computed(() => [
  // dy 错开: 复位后三支箭头重合, 标签仍然一行一个 (紫点自己的标签在 −10)
  { id: 'w₀', v: W0, c: 'text-dim', w: 2, dy: 30 },
  { id: 'LoRA', v: v.value, c: 'right', w: 2, dy: 16 },
  { id: 'DoRA', v: doraW.value, c: 'left', w: 3, dy: -24 },
])
const sg = (x) => (x >= 0 ? '+' : '') + x.toFixed(2)
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.star { font-size: 18px; fill: var(--warn); }
.cmp { width: 100%; border-collapse: collapse; font-size: 12px; }
.cmp th, .cmp td { border-bottom: 1px dashed var(--border); padding: 4px 6px; text-align: right; color: var(--text); }
.cmp th:first-child, .cmp td:first-child { text-align: left; color: var(--text-muted); }
.cmp th { color: var(--text-muted); font-weight: 500; }
</style>
