<!--
  残差流实验台: 把 "Pre-LN 把 norm 挪进分支" 这件事画出来, 并让它的后果 (各层梯度尺度) 可测。
  上面是一个 block 的真实接线, 下面是 L 层摞起来之后每层的读数。切换 Pre/Post 时 norm 方块会真的搬家。
  右侧三个读数永远按反传算; 「看前向」只换下面那张柱图, 并多出一行主干方差。
-->
<template>
  <LabFrame
    title="残差流 — norm 挪一个位置, 几十层的网络才训得动"
    sub="- 模拟什么: Transformer 的每一层 (block) 不是把输入整个变换掉, 而是在一条主干上做加法: x 进来, attention 算出一个修正量加回去, FFN 再算一个加回去, 出去的还是 [B,T,D]。这条一路加下去的主干叫残差流, attention 和 FFN 只是从旁边接出去算、再加回来的分支。归一化 norm (把向量缩放到固定尺度) 放在哪, 有两种接法: Post-LN 放在主干上, 加完再归一化; Pre-LN 放在分支上, 只归一化送进子层的那份副本。这个位置决定了反向传播时, 梯度能不能原样穿过几十层回到第 0 层。对应 llm_models/layers/core/blocks.py 的 PreLNBlock。
      - 图上是什么: 上面是一个 block 的接线图: 粗线是主干 x, 方块是 norm、attention (GQA) 和 FFN (SwiGLU), 「+」是加回主干的地方。切 Pre / Post 时 norm 方块会真的从分支搬到主干上。下面的柱图是把 L 层摞起来之后每一层的读数: 默认看反传, 每根柱是这一层拿到的梯度尺度, 顶层记为 1, 对数高度, 刻度 1、1e-5、1e-10; 切到「看前向」则是每层出口处主干的方差, 线性高度。左边第 0 层最靠近输入, 右边第 L−1 层最靠近输出。
      - 可以做什么: 「Pre-LN / Post-LN」切接法。「柱图看反传 / 看前向」切柱图内容。拖「堆多少层」改 L。拖「分支输出方差 v」改每个子层往主干上加的东西有多大 (相对主干)。把鼠标放到接线图任意方块上 (键盘: Tab 选中), 右侧说明它在干什么。
      - 应该看到: Pre-LN 时反传柱图全是满格, 「第 0 层梯度尺度」恒为 1.00: 主干只有加法, 梯度原样传到底。切到 Post-LN, 柱子从右往左一路塌下去: 每过一次主干上的 norm, 梯度乘 $1/\sqrt{1+v}$ (v = 1 时是 0.707), 一层两个子层要过两次, 默认 24 层时第 0 层只剩约 1e-7。「顶层 / 底层梯度之比」是两头差了多少倍。切「看前向」: Pre-LN 的主干方差逐层累加 (1 + v·(l+1)), 所以出口要再过一次 final norm; Post-LN 每层都归一化, 恒为 1。"
    module="llm_models/layers/core/blocks.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '把层数拖到 48, 两种接法第 0 层的梯度尺度各是多少? 为什么 Post-LN 那个年代的论文都在讲 warmup?',
      answer: 'Pre-LN 是 1.00, Post-LN 只剩约 7e-15。\n- Pre-LN: 恒为 1.00。主干是纯加法, 反传时加法节点把上游梯度原样发给两条路, 走主干的那条一路送到底, 与层数无关。\n- Post-LN: 一层里有两个子层, 梯度回去要过两次主干上的 norm, 每次乘 $1/\\sqrt{1+v}$, $v = 1$ 时是 0.707。48 层要连乘 2 × 47 = 94 次, 第 0 层只剩约 7e-15。\nPost-LN 的后果是顶层梯度正常、底层几乎为零: 同一个学习率, 对顶层合适就对底层太小, 对底层合适就把顶层炸了。warmup (前几千步把学习率从很小慢慢升上去) 就是在等各层的尺度自己长匀, 否则一开始就不稳。\nPre-LN 直接把这个问题消掉了, 所以现在没人再为它调 warmup, 只剩出口处要补一次 final norm。',
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
      <p v-else class="lab-note">把鼠标放到接线图任意一个方块上 (或用 Tab 选中), 这里会说明它在这条流水线里干什么。</p>
      <div class="lab-note">
        <p><Tex text="这里的梯度用一个简化模型: 每过一次主干上的 norm, 梯度乘 $1/\sqrt{1+v}$, $v$ 是分支输出相对主干的方差 (分支加回来的东西有多大)。norm 把 $1+v$ 的方差压回 1, 反向时梯度就按同样的比例缩。" /></p>
        <p><Tex text="前向用同一个 $v$: Pre-LN 每层往主干上加方差 $v$, 第 $l$ 层出口的方差是 $1+v(l+1)$, 越往上越大, 所以出口要再过一次 norm (代码里的 ln_f); Post-LN 每层都归一化, 恒为 1。" /></p>
        <p>柱图的对数轴从 1 画到 1e-15, 更小的值贴底。</p>
        <p><Tex text="真实网络里每层的 $v$ 不一样, 但「Post-LN 要连乘 $L$ 个小于 1 的因子、Pre-LN 有一条恒等通路」这一点是一样的。" /></p>
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
    desc: '一个 block 里唯一让不同位置互相说话的地方。进去 [B,T,D], 出来还是 [B,T,D]。' },
  ffn: { pre: 'SwiGLU', post: 'SwiGLU', cls: 'ffn', title: 'FFN',
    desc: '逐位置独立的两层 MLP, 中间宽到 d_ff。一个 block 的参数大头在这里, LLaMA 里约占三分之二。' },
}
const NORM_DESC = {
  pre: { title: 'norm 在分支上 (Pre-LN)', desc: '只归一化送进子层的那一份副本。主干上的 x 一个字没动, 直接往下走。' },
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
      desc: '上一个 block 的输出 (第一层是 embedding)。它兵分两路: 一路留在主干, 一路送进子层。' },
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
