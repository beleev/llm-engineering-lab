<!--
  可学 attention sink 实验台 (对应 llm_models/layers/core/attention.py:GroupedQueryAttention 的 use_sink)。
  只讲一件事: softmax 强迫概率和为 1, 一个 head 即使 “没什么可看” 也得把 1 分完;
  GPT-OSS 给每个 head 加一个可学 logit 参与 softmax、算完就丢 —— 给 head 一个 “谁都不看” 的出口。
-->
<template>
  <LabFrame
    title="Attention sink — 让一个注意力头可以「谁都不看」"
    sub="- 模拟什么: 单个注意力 head 在当前位置对 6 个历史 key 分别计算未归一化相关性得分 (logit), 经 softmax 归一化为概率权重后再对各 key 的 value 加权求和得到该 head 的输出。由于标准 softmax 严格要求所有真实 key 的概率之和恒等于 1, 即便当前上下文中 6 个历史 key 均与查询无关, 该 head 仍不得不将 100% 的权重强行分配给它们, 导致向残差流注入无关 value 的加权噪声。GPT-OSS 引入的可学习 attention sink 机制对此进行了修正: 为每个注意力 head 额外配置一个可学习的标量偏置 logit (即 sink 槽位), 该标量一同参与 softmax 分母归一化但不关联任何 value 向量; 分配给 sink 的概率质量直接丢弃不参与加权求和, 从而允许真实历史 key 的有效概率之和小于 1 (实现近似「本步静默」)。
- 图上是什么: 上下两排对齐的柱状图, 开启 sink 时最右侧额外显示一列:
  - 上排柱图: 6 根柱子分别代表历史位置 k0 至 k5 的注意力 logit, 零刻度线居中, 向上为正、向下为负, 柱顶圆点支持上下拖动。
  - 下排柱图: 经 softmax 归一化后各真实 key 实际分得的注意力概率, 柱顶标注对应百分比。
  - 最右侧 sink 专列: 仅在开启 use_sink 时显示; 上方为其标量 sink logit, 下方虚线空心柱代表被 sink 吸收并直接丢弃的概率占比。
- 可以做什么: 通过拖拽手柄、模式按钮与场景预设进行交互对照:
  - 拖动圆点: 上下拖动上排任意圆点修改对应 key 的 logit (聚焦后按键盘上下方向键每次增减 0.5)。
  - 切换机制按钮: 在「use_sink = True」与「标准 softmax」之间切换是否启用可学习 sink 槽位。
  - 点击预设「有一个相关 key」: 将 k2 的 logit 设为 4、其余位置设为负分。
  - 点击预设「全都不相关」: 将全部 6 个历史 key 的 logit 统一压低至 −3 附近。
  - 点击预设「Python demo」: 复现 infer_gpt_oss 脚本中的验证算例, 其中 3 个 key 得分为 0、3 个 key 得分为 −6, sink 标量设为 $\ln 3 \approx 1.1$。
  - 拖动「sink logit」滑杆: 开启 sink 后出现 (调节范围 −4 至 4), 用于模拟该 head 在训练收敛后学到的静态 sink 阈值。
- 应该看到: 在「标准 softmax」模式下将 6 根 logit 柱同步向下压低时, 下排概率分布保持不变, 因为标准 softmax 具有平移不变性、仅取决于各 logit 之间的相对差值。开启 sink 后再同步压低 6 个真实 key 的 logit, 由于固定参考点 sink logit 未变, 概率质量便持续转移至右侧 sink 槽位。右侧读数含义如下:
  - 「真实 key 的概率和」代表该 head 实际输出的加权幅值比例, 100% 为满幅输出, 数值越小表示该 head 在当前步越趋于静默。
  - 「被 sink 吸走」表示分配给虚拟槽位并直接丢弃的概率占比, 与上一项之和恒为 100%。
  - 「最大单 key 概率」衡量注意力分布的峰值集中程度。
  - 「最大 key logit − sink logit」反映最高真实相关性与静默阈值的差值: 差值为较大正数时 sink 吸收率极低、几乎不干预正常关注; 差值为负数时则由 sink 吸收主要概率质量。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.moe.gpt_oss.infer_gpt_oss"
    :challenge="{
      ask: '先点击「预设: 全都不相关」, 再点击「标准 softmax」关闭 sink。请先推演此时 6 个历史 key 各自的概率与总和分别是多少, 该 head 的实际输出具有什么特征? 推演后再点击「use_sink = True」开启 sink 观察变化。',
      answer: '在标准 softmax 下, 每个历史 key 分得的概率均约为 1/6 (约 17%), 概率总和恒为 100%; 此时该 head 的输出为 6 个无关 value 向量的等权平均, 相当于向残差流注入满幅度的无关噪声。\n- 平移不变性限制: 6 个历史 key 的 logit 均处于 −3 附近。由于标准 softmax 计算式 $e^{s_j} / \\sum_k e^{s_k}$ 对全体输入同时加减任意常数保持结果不变, 它仅能反映各 logit 之间的相对差值而无法表达「全体绝对相关性均很低」; 6 个同样低但彼此接近的负分经归一化后仍会得到均匀平分的概率。\n- 被迫满幅输出: 由于概率总和恒等于 1, 加权输出 $\\sum_j p_j v_j$ 的组合权重规模与命中强相关 token 时完全相同, 导致无关上下文噪声被全额叠加进残差流。\n- 隐式首 token 汇聚的副作用: 未设显式 sink 槽位的常规 Transformer 在预训练中会自发学会将无处安放的冗余注意力权重倾倒在序列第 1 个 token 上 (即 StreamingLLM 论文揭示的隐式 attention sink 现象); 这导致滑动窗口注意力一旦将序列首部 token 移出 KV cache, 模型的生成质量便会严重退化。\n- 开启显式 sink (设标量 logit 为 0) 后的效果: softmax 分母新增固定参考项 $e^0 = 1$, 而全部 6 个无关历史 key 的指数项之和仅为 $6 \\cdot e^{-3} \\approx 0.3$。由此 sink 槽位吸收并丢弃约 77% 的概率质量, 真实历史 key 的有效概率总和压缩至约 23%, 该 head 的噪声输出幅度随之大幅衰减。\n- 强相关场景对照: 点击「预设: 有一个相关 key」时, k2 的 logit 为 4, 其指数项 $e^4 \\approx 55$ 远大于 sink 参考项 $e^0 = 1$, 此时 sink 吸收占比极低, 几乎不干扰有效信息提取。\n借助可学习的显式 sink 标量, 模型无需再依赖序列起始位置的真实 token 充当概率垃圾桶, 因此即便采用纯滑动窗口截断驱逐首部 token 也不会破坏推理稳定性。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: useSink }" @click="useSink = true">use_sink = True (GPT-OSS)</button>
        <button type="button" :class="{ active: !useSink }" @click="useSink = false">标准 softmax</button>
        <button type="button" @click="logits = [...PRESET.hit]">预设: 有一个相关 key</button>
        <button type="button" @click="logits = [...PRESET.none]">预设: 全都不相关</button>
        <button type="button" @click="demo">预设: Python demo (3 个 0 分 key + sink = ln 3)</button>
      </div>
      <LabSlider v-if="useSink" v-model="sink" label="sink logit (可学)" :min="-4" :max="4" :step="0.1" :format="(t) => t.toFixed(1)" />
    </template>

    <svg ref="svg" viewBox="0 0 520 300" role="group" aria-label="logit 与概率">
      <text x="4" y="12" class="sec">logit (可拖)</text><text x="4" y="192" class="sec">softmax 概率</text>
      <line :x1="X0" :x2="X1" :y1="ly(0)" :y2="ly(0)" class="base" /><line :x1="X0" :x2="X1" :y1="PY1" :y2="PY1" class="base" />
      <g v-for="(l, j) in logits" :key="j">
        <rect :x="cx(j) - 18" :y="Math.min(ly(l), ly(0))" width="36" :height="Math.abs(ly(l) - ly(0))" class="lbar" />
        <text :x="cx(j)" :y="l >= 0 ? ly(l) - 10 : ly(l) + 18" class="num">{{ l.toFixed(1) }}</text>
        <g class="draggable" tabindex="0" role="slider" :aria-label="`key ${j} 的 logit`" :aria-valuenow="l"
          @pointerdown="start($event, { svg, onMove: ({ y }) => setL(j, y) })" @keydown.up.prevent="bump(j, 0.5)" @keydown.down.prevent="bump(j, -0.5)">
          <rect :x="cx(j) - 22" :y="LY0 - 4" width="44" :height="LY1 - LY0 + 8" fill="transparent" />
          <circle :cx="cx(j)" :cy="ly(l)" r="6" class="handle" />
        </g>
        <rect :x="cx(j) - 18" :y="pyy(probs[j])" width="36" :height="PY1 - pyy(probs[j])" class="pbar" />
        <text :x="cx(j)" :y="pyy(probs[j]) - 4" class="num">{{ (probs[j] * 100).toFixed(0) }}%</text>
        <text :x="cx(j)" y="292" class="tok">k{{ j }}</text>
      </g>
      <g v-if="useSink">
        <rect :x="cx(6) - 18" :y="Math.min(ly(sink), ly(0))" width="36" :height="Math.abs(ly(sink) - ly(0))" class="sbar" />
        <text :x="cx(6)" :y="sink >= 0 ? ly(sink) - 5 : ly(sink) + 12" class="num">{{ sink.toFixed(1) }}</text>
        <rect :x="cx(6) - 18" :y="pyy(sinkP)" width="36" :height="PY1 - pyy(sinkP)" class="sbar ghost" />
        <text :x="cx(6)" :y="pyy(sinkP) - 4" class="num">{{ (sinkP * 100).toFixed(0) }}% 丢弃</text>
        <text :x="cx(6)" y="292" class="tok s">sink</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>真实 key 的概率和 (= 这个 head 的输出幅度)</span><b :class="keySum < 0.999 ? 'good' : ''">{{ (keySum * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>被 sink 吸走并丢弃的概率</span><b>{{ (sinkP * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>最大单 key 概率 (注意力有多集中)</span><b>{{ (Math.max(...probs) * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>最大 key logit − sink logit (越正 sink 越隐身)</span><b>{{ useSink ? (Math.max(...logits) - sink).toFixed(1) : '—' }}</b></div>
      <div class="lab-note">
        <template v-if="useSink">
          <p>可学习 sink 等效于在注意力分母中引入一个不携带 value 的虚拟基准槽位:</p>
          <ul class="pts">
            <li>某个 key 的 logit 明显高于它时, 它分不到什么概率, 等于隐身。</li>
            <li>所有 key 都低于它时, 它把大部分概率吸走, 这个 head 的输出就变小。</li>
          </ul>
          <p>该标量参数在初始化时设为 0, 并在训练过程中由每个注意力 head 独立学习各自的静默阈值。</p>
        </template>
        <p v-else>标准 softmax 具有平移不变性: 将所有 key 的 logit 同步压低时, 输出概率分布完全不变。由于它仅对各候选项之间的相对差值敏感而无法表达全局拒绝关注, 6 个均不相关的历史 key 依然会分满 100% 的概率权重。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, softmax, sum } from '@/utils/labmath.js'

const PRESET = { hit: [-2, -1, 4, -1.5, -2, -1], none: [-3, -3.2, -2.8, -3.1, -2.9, -3] }
const useSink = ref(true), sink = ref(0), logits = ref([...PRESET.none])
const svg = ref(null)
const { start } = useDrag()

// ★ 与 Python 同: softmax(cat([scores, sink]))[..., :-1] —— sink 只进分母, 行和 < 1
const all = computed(() => softmax(useSink.value ? [...logits.value, sink.value] : logits.value))
const probs = computed(() => all.value.slice(0, 6))
const sinkP = computed(() => (useSink.value ? all.value[6] : 0))
const keySum = computed(() => sum(probs.value))

const X0 = 50, X1 = 515, LY0 = 22, LY1 = 170, PY0 = 204, PY1 = 274, LMAX = 6
const cx = (j) => X0 + 36 + j * 66
const ly = (v) => (LY0 + LY1) / 2 - (v / LMAX) * ((LY1 - LY0) / 2)
const pyy = (p) => PY1 - p * (PY1 - PY0)
const setL = (j, y) => { logits.value = logits.value.map((v, i) => (i === j ? clamp(Math.round((((LY0 + LY1) / 2 - y) / ((LY1 - LY0) / 2)) * LMAX * 10) / 10, -LMAX, LMAX) : v)) }
// infer_gpt_oss 里的算例: 3 个分数为 0 的 key + sink = ln 3 → 真实 key 合计正好一半 (另 3 个 key 压到 −6 ≈ 屏蔽)
const demo = () => { useSink.value = true; sink.value = Math.log(3); logits.value = [0, 0, 0, -6, -6, -6] }
const bump = (j, by) => { logits.value = logits.value.map((v, i) => (i === j ? clamp(v + by, -LMAX, LMAX) : v)) }
</script>

<style scoped>
.sec { font-size: 10px; fill: var(--text-dim); }
.base { stroke: var(--border-strong); }
.lbar { fill: color-mix(in srgb, var(--accent) 35%, transparent); stroke: var(--accent); }
.pbar { fill: color-mix(in srgb, var(--left) 45%, transparent); stroke: var(--left); }
.sbar { fill: color-mix(in srgb, var(--eye) 40%, transparent); stroke: var(--eye); }
.sbar.ghost { stroke-dasharray: 4 3; fill: color-mix(in srgb, var(--eye) 15%, transparent); }
.handle { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.draggable:focus-visible .handle { stroke: var(--text); }
.num { text-anchor: middle; font-size: 10px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.tok { text-anchor: middle; font-size: 11px; fill: var(--text-muted); }
.tok.s { fill: var(--eye); font-weight: 700; }
</style>
