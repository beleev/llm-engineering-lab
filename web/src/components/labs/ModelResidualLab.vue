<!--
  残差流实验台: 把 "Pre-LN 把 norm 挪进分支" 这件事画出来, 并让它的后果 (各层梯度尺度) 可测。
  上面是一个 block 的真实接线, 下面是 L 层摞起来之后每层的读数。切换 Pre/Post 时 norm 方块会真的搬家。
  右侧三个读数永远按反传算; 「看前向」只换下面那张柱图, 并多出一行主干方差。
-->
<template>
  <LabFrame
    title="残差流 — norm 挪一个位置, 几十层的网络才训得动"
    sub="- 模拟什么: Transformer 的每个 block 并非直接替换输入表示, 而是在一条恒等主干上累加增量: 输入张量 x (形状为 [B,T,D]) 进入后, attention 子层计算一个修正项加回主干, FFN 子层再计算一个修正项加回主干, 输出形状仍保持 [B,T,D]。这条贯穿全网的加法主干称为残差流, attention 与 FFN 均为从主干引出又汇回主干的残差分支。归一化层 norm 的安放位置直接决定了反向传播时上游梯度能否无衰减地穿过数十层传回第 0 层。对应实现见 llm_models/layers/core/blocks.py 中的 PreLNBlock。norm 主要有两种连接方式:
  - Post-LN: 将 norm 串联在残差主干上, 每次分支相加之后再做归一化。
  - Pre-LN: 将 norm 放在残差分支入口, 仅对送入子层的那份副本做归一化, 主干保持纯加法通路。
- 图上是什么: 上方为单层 block 拓扑接线图, 下方为多层堆叠后的逐层指标柱状图:
  - 上方接线图: 展示单个 block 内部的数据流向。粗线代表残差主干 x, 方块分别代表 norm、attention (GQA) 与 FFN (SwiGLU), 「+」代表分支汇回主干的加法节点; 切换 Pre-LN 与 Post-LN 时, norm 方块会在分支入口与主干之间移动。
  - 下方柱状图: 展示堆叠 L 层后各层的读数, 最左侧第 0 层紧邻输入端, 最右侧第 L−1 层紧邻输出端。
  - 默认反传视图: 每根柱子表示该层接收到的梯度尺度 (以顶层为基准 1), 采用对数纵轴, 刻度标示 1、1e-5、1e-10。
  - 切换至「看前向」视图: 柱高表示每层出口处残差主干的信号方差, 采用线性纵轴。
- 可以做什么: 按界面控件顺序操作:
  - 「Pre-LN / Post-LN」: 切换归一化层的连接拓扑。
  - 「柱图看反传 / 看前向」: 切换下方柱状图展示反向梯度尺度还是前向主干方差。
  - 拖动「堆多少层」: 调整模型总层数 L。
  - 拖动「分支输出方差 v」: 调整每个子层分支汇回主干信号的相对方差大小。
  - 悬停接线图方块: 将鼠标移至任意模块方块上 (键盘可通过 Tab 键切换焦点), 右侧说明区会显示该模块在当前计算流中的具体作用。
- 应该看到: 依次切换连接方式与观测视图对比读数:
  - Pre-LN 模式下: 反传柱状图各层全部满格, 「第 0 层梯度尺度」恒为 1.00, 因为残差主干上仅有加法运算, 梯度可沿恒等通路无损直达底层。
  - 切换至 Post-LN 模式: 反传柱高从右向左逐层指数衰减; 梯度每穿过一次主干上的 norm 就要乘以缩放因子 $1/\sqrt{1+v}$ ($v = 1$ 时约为 0.707), 每层含两个子层需穿过两次, 在默认 24 层配置下第 0 层梯度仅剩约 1e-7。
  - 「顶层 / 底层梯度之比」: 量化首尾两层梯度幅值相差的倍数。
  - 切换至「看前向」: Pre-LN 模式下主干方差逐层线性累加为 $1 + v \cdot (l + 1)$, 因此模型出口处必须额外补一层 final norm; Post-LN 模式下每层主干均被重归一化, 出口方差恒为 1。"
    module="llm_models/layers/core/blocks.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '将模型层数拖至 48 层, Pre-LN 与 Post-LN 两种接法下第 0 层的梯度尺度分别是多少? 为什么早期采用 Post-LN 架构的 Transformer 论文普遍高度依赖学习率 warmup?',
      answer: '答案: 48 层配置下, Pre-LN 的第 0 层梯度尺度恒为 1.00, 而 Post-LN 仅剩约 7e-15。\n- Pre-LN: 底层梯度尺度恒为 1.00。由于残差主干由纯加法节点串联而成, 反向传播经过加法节点时会将上游梯度原样分配给主干与分支两条路径, 沿主干回传的分量不受层数影响, 直达第 0 层。\n- Post-LN: 每个 block 包含两个子层, 梯度回传每经过一层就要穿过两次主干上的 norm, 每次乘以因子 $1/\\sqrt{1+v}$ (当 $v = 1$ 时为 0.707)。在 48 层网络中, 回传至第 0 层需连乘 $2 \\times 47 = 94$ 次该因子, 梯度幅值衰减至约 7e-15。\nPost-LN 造成了严重的层间梯度失衡: 靠近输出端的顶层梯度尺度正常, 而靠近输入端的底层梯度几乎为零。若使用全局统一的学习率, 适配顶层的学习率会让底层几乎无法更新, 适配底层的大学习率则会令顶层更新发散。学习率 warmup (在训练前几千步将学习率从极小值逐步线性抬升) 的作用正是在初始阶段避免顶层崩溃, 等待各层参数尺度逐步收敛协调。\n改用 Pre-LN 后, 底层梯度不再随网络深度指数衰减, 训练稳定性显著提升, 对 warmup 的依赖大幅减弱 (现代大模型训练虽仍普遍保留 warmup 以平抑优化器初期动量波动, 但已不再靠它解决底层梯度消失); 相应代价是主干方差逐层增长, 需在网络末尾补充一层 final norm。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: pre }" @click="pre = true">Pre-LN (norm 在分支上)</button>
        <button type="button" :class="{ active: !pre }" @click="pre = false">Post-LN (norm 在主干上)</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: !flow }" @click="flow = false">柱图看反传 (每层拿到的梯度)</button>
        <button type="button" :class="{ active: flow }" @click="flow = true">柱图看前向 (每层出口的主干方差)</button>
      </div>
      <LabSlider v-model="L" label="堆多少层 L" :min="2" :max="48" />
      <LabSlider v-model="v" label="分支输出方差 v (相对主干)" :min="0.1" :max="3" :step="0.1"
                 :format="(x) => x.toFixed(1)" />
    </template>

    <DagView v-bind="wiring" @node-hover="hover = $event" />
    <p :class="['verdict mono', gradCls]">
      {{ pre ? '主干从头到尾只有加法' : `norm 压在主干上, 梯度回到第 0 层要过 ${2 * (L - 1)} 次` }} → 第 0 层梯度 {{ fmt(grad[0]) }}
    </p>

    <!-- L 层摞起来: 每层一根柱。反传用对数高度 (1 到 1e-15), 前向用线性高度 -->
    <div class="depth">
      <div class="depth-head mono">
        {{ flow ? `前向: 第 l 层出口的主干方差 (线性高度, 满格 = ${fmt(fwdMax)})`
                : '反传: 第 l 层拿到的梯度尺度 (顶层 = 1, 对数高度)' }}
      </div>
      <div class="bars" :style="{ gridTemplateColumns: `repeat(${L}, 1fr)` }">
        <template v-if="!flow">
          <div v-for="t in TICKS" :key="t.label" class="tick mono" :style="{ bottom: `calc(6px + ${t.at} * (100% - 12px))` }">
            <span>{{ t.label }}</span>
          </div>
        </template>
        <div
          v-for="(g, l) in series" :key="l" class="bar-slot"
          :title="`第 ${l} 层: ${fmt(g)}`"
        >
          <div class="bar" :style="{ height: `${barH(g)}%`, background: barColor(g) }" />
        </div>
      </div>
      <div class="axis mono"><span>第 0 层 (最靠近输入)</span><span>第 {{ L - 1 }} 层 (最靠近输出)</span></div>
    </div>

    <template #stats>
      <div class="kv"><span>反传: 第 0 层 (最靠近输入) 拿到的梯度尺度</span><b :class="gradCls">{{ fmt(grad[0]) }}</b></div>
      <div class="kv"><span>反传: 顶层 / 底层梯度之比 (两头差几倍)</span><b :class="gradCls">{{ fmt(1 / grad[0]) }}</b></div>
      <div class="kv"><span>反传: 每过一次主干上的 norm, 梯度乘</span><b>{{ pre ? '1.000 (主干上没有 norm)' : perNorm.toFixed(3) }}</b></div>
      <div v-if="flow" class="kv"><span>前向: 顶层出口的主干方差</span><b>{{ fmt(fwd[L - 1]) }}</b></div>
      <div v-if="hovered" class="kv-note">
        <b>{{ hovered.title }}</b>
        <p>{{ hovered.desc }}</p>
      </div>
      <p v-else class="lab-note">将鼠标悬停于接线图中的任意模块方块上 (或按 Tab 键选中), 此处将显示该模块在前向与反向计算流中的具体作用。</p>
      <div class="lab-note">
        <p><Tex text="此处采用简化方差传播模型计算梯度尺度: 设分支输出相对主干的方差为 $v$, 前向相加后总方差变为 $1+v$, 主干上的 norm 将其重缩放回单位方差 1, 反向传播穿过该 norm 时梯度便按相同系数乘上 $1/\sqrt{1+v}$。" /></p>
        <p><Tex text="前向用同一个 $v$:" /></p>
        <ul class="pts">
          <li><b>Pre-LN:</b> <Tex text="Pre-LN 模式下每层向主干独立叠加方差 $v$, 第 $l$ 层出口处的主干方差累积为 $1+v(l+1)$ 并随层数线性增长, 因此网络末端必须接入最终归一化层 (对应代码中的 ln_f)。" /></li>
          <li><b>Post-LN:</b> 每层都归一化, 恒为 1。</li>
        </ul>
        <p>柱状图对数纵轴的显示范围为 1 至 1e-15, 低于 1e-15 的数值将截断显示在坐标轴底部。</p>
        <p><Tex text="真实训练网络中各层的分支相对方差 $v$ 并非完全相等, 但「Post-LN 反向梯度需连乘 $L$ 层小于 1 的缩放因子、Pre-LN 始终保有恒等残差通路」这一结构性质完全一致。" /></p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import DagView from '@/components/dag/DagView.vue'
import { range } from '@/utils/labmath.js'

const pre = ref(true)
const flow = ref(false)
const L = ref(24)
const v = ref(1)
const hover = ref(null)

const SUBLAYERS = ['attn', 'ffn']
const META = {
  attn: { pre: 'GQA', post: 'GQA', cls: 'attn', title: 'attention',
    desc: '一个 block 里唯一让不同位置互相交换信息的地方。进去 [B,T,D], 出来还是 [B,T,D]。' },
  ffn: { pre: 'SwiGLU', post: 'SwiGLU', cls: 'ffn', title: 'FFN',
    desc: '逐位置独立的两层 MLP, 中间宽到 d_ff。一个 block 的参数大头在这里, LLaMA 里约占三分之二。' },
}
const NORM_DESC = {
  pre: { title: 'norm 在分支上 (Pre-LN)', desc: '只归一化送进子层的那一份副本。主干上的 x 不变, 直接往下走。' },
  post: { title: 'norm 在主干上 (Post-LN)', desc: '先把子层输出加回主干, 再归一化整条主干。主干从此不是纯加法, 反传要过这一道。' },
}

// 接线图。坐标交给 DagView 里的 dagre, 这里只声明"谁接谁" ——
// 切 Pre/Post 时 norm 方块真的搬家 (分支上 ↔ 主干上), 版式自己重排。
const boxes = computed(() => {
  const nd = NORM_DESC[pre.value ? 'pre' : 'post']
  return SUBLAYERS.flatMap((op, i) => {
    const m = META[op]
    return [
      { id: `n${i}`, label: pre.value ? 'RMSNorm' : 'LayerNorm', kind: pre.value ? 'norm' : 'bad', ...nd },
      { id: op, label: m[pre.value ? 'pre' : 'post'], kind: m.cls, title: m.title, desc: m.desc },
    ]
  })
})

const wiring = computed(() => {
  const lit = (b) => ({ ...b, aria: b.title, active: hover.value === b.id })
  const nodes = boxes.value.map(lit)
  const adds = SUBLAYERS.map((op, i) => lit({ id: `add${i}`, label: '+', kind: 'trunk', title: '加回主干',
    desc: `把 ${META[op].title} 的输出加到残差流上, 形状还是 [B,T,D]。${pre.value ? '加完直接往下走。' : '加完还要过主干上的 norm。'}` }))
  const io = [
    { id: 'x', label: 'x (in)', sub: '[B, T, D]', kind: 'io', title: '残差流入口',
      desc: '上一个 block 的输出 (第一层是 embedding)。它分成两路: 一路留在主干, 一路送进子层。' },
    { id: 'y', label: 'y (out)', sub: '[B, T, D]', kind: 'io', title: '残差流出口',
      desc: '交给下一个 block。形状和入口一样, 所以 block 可以一层层摞。' },
  ].map(lit)
  const trunk = (from, to, label) => ({ from, to, kind: 'main', label })
  const edges = []
  // 主干依次穿过每个子层的加号; Post-LN 下加号后面还挂着一个 norm
  let cur = 'x'
  SUBLAYERS.forEach((op, i) => {
    if (pre.value) {
      // norm 和零件都在分支上, 主干只是一路加过去
      edges.push(trunk(cur, `add${i}`, '残差流 x'), { from: cur, to: `n${i}` },
        { from: `n${i}`, to: op }, { from: op, to: `add${i}` })
      cur = `add${i}`
    } else {
      // 零件直接挂分支, norm 压在主干上 —— 主干从此不是纯加法
      edges.push(trunk(cur, `add${i}`, '残差流 x'), { from: cur, to: op },
        { from: op, to: `add${i}` }, trunk(`add${i}`, `n${i}`))
      cur = `n${i}`
    }
  })
  edges.push(trunk(cur, 'y'))
  return { dir: 'TB', nodes: [io[0], ...nodes, ...adds, io[1]], edges }
})

// 没有说明的节点返回 undefined, 说明框就不显示
const hovered = computed(() => wiring.value.nodes.find((n) => n.id === hover.value && n.desc))

// ── 反传: Pre-LN 主干是恒等; Post-LN 每层过两次 norm。右侧读数只用这一条序列 ──
const perNorm = computed(() => 1 / Math.sqrt(1 + v.value))
const grad = computed(() =>
  range(L.value).map((l) => (pre.value ? 1 : perNorm.value ** (2 * (L.value - 1 - l)))))   // ★ 上面还有 L-1-l 层, 每层两次 norm
// ── 前向: Pre-LN 主干方差逐层累加, Post-LN 每层被归一化回 1 ──
const fwd = computed(() => range(L.value).map((l) => (pre.value ? 1 + v.value * (l + 1) : 1)))
const fwdMax = computed(() => Math.max(...fwd.value))
const series = computed(() => (flow.value ? fwd.value : grad.value))

// 颜色跟数值走: 梯度剩两成以上算好, 不到千分之一算坏
const tone = (g) => (g > 0.2 ? 'good' : g < 1e-3 ? 'bad' : '')
const gradCls = computed(() => tone(grad.value[0]))

const fmt = (x) =>
  x === 0 ? '0' : x >= 1000 || x < 0.001 ? x.toExponential(1) : x.toFixed(x < 1 ? 3 : 2)
// 反传用固定的对数轴 1 → 1e-15: 线性画的话 1e-8 和 0 看不出区别; 轴固定, 拖层数时斜坡才有可比性
const FLOOR = -15
const TICKS = [0, -5, -10].map((e) => ({ label: e ? `1e${e}` : '1', at: 1 - e / FLOOR }))
const barH = (g) =>
  flow.value ? (g / fwdMax.value) * 100 : Math.max(2, (1 - Math.log10(g) / FLOOR) * 100)
const barColor = (g) => {
  if (flow.value) return 'var(--accent)'
  return { good: 'var(--left)', bad: 'var(--danger)', '': 'var(--warn)' }[tone(g)]
}
</script>

<style scoped>
.verdict { font-size: 11.5px; text-align: center; margin-top: 8px; }
.verdict.good { color: var(--left); }
.verdict.bad { color: var(--danger); }

.depth { margin-top: 14px; }
.depth-head { font-size: 11px; color: var(--text-dim); margin-bottom: 6px; }
.bars { position: relative; display: grid; gap: 1px; height: 96px; align-items: end; background: var(--code-bg); padding: 6px 6px 6px 40px; border-radius: var(--radius-sm); }
.bar-slot { height: 100%; display: flex; align-items: flex-end; }
/* 刻度线: 一条横贯的细线, 左端写数值 */
.tick { position: absolute; left: 0; right: 6px; border-top: 1px dashed var(--border-strong); pointer-events: none; }
.tick span { position: absolute; left: 4px; top: -7px; font-size: 9.5px; color: var(--text-dim); background: var(--code-bg); padding-right: 3px; }
.bar { width: 100%; border-radius: 1px 1px 0 0; }
.axis { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim); margin-top: 4px; }
.kv-note { border-left: 2px solid var(--accent); padding-left: 10px; }
.kv-note b { font-size: 13px; color: var(--text); }
.kv-note p { font-size: 12px; color: var(--text-muted); line-height: 1.6; margin-top: 3px; }
</style>
