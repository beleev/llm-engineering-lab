<!--
  残差流实验台: 把 "Pre-LN 把 norm 挪进分支" 这件事画出来, 并让它的后果 (各层梯度尺度) 可测。
  左边是一个 block 的真实接线, 右边是 L 层摞起来之后反传回来的梯度。切换 Pre/Post 时 norm 方块会真的搬家。
-->
<template>
  <LabFrame
    title="残差流 — norm 挪一个位置, 深层网络就能开训"
    sub="主干那条粗带子是残差流 x [B,T,D], 从头贯到尾。attention 和 FFN 都不在主干上, 从旁边接出去、算完再加回来。
      norm 放在哪, 决定了主干还是不是一条干净的加法通路。这也就决定了梯度能不能原样回到第 0 层。"
    module="llm_models/layers/core/blocks.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '把层数拖到 48, 两种接法第 0 层的梯度尺度各是多少? 为什么 Post-LN 那个年代的论文都在讲 warmup?',
      answer: '- Pre-LN: 恒为 1.00。主干是纯加法, 反传时恒等项把梯度原样送到底, 与层数无关。\n- Post-LN: 一层里有两个子层, 要过两次 norm, 每次乘 $1/\\sqrt{1+v}$, $v=1$ 时是 0.707。48 层连乘 94 次, 第 0 层只剩 7e-15。\nPost-LN 顶层梯度正常、底层几乎为零, 同一个学习率没法同时喂饱两头。warmup 就是拿前几千步把学习率压住, 等各层尺度自己长匀。\nPre-LN 直接把这个问题消掉了, 所以现在没人再为它调 warmup。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: pre }" @click="pre = true">Pre-LN (norm 在分支上)</button>
        <button type="button" :class="{ active: !pre }" @click="pre = false">Post-LN (norm 在主干上)</button>
        <button type="button" @click="flow = !flow">{{ flow ? '看前向' : '看反传' }}</button>
      </div>
      <LabSlider v-model="L" label="堆多少层" :min="2" :max="48" />
      <LabSlider v-model="v" label="分支输出方差 v" :min="0.1" :max="3" :step="0.1"
                 :format="(x) => x.toFixed(1)" />
    </template>

    <DagView v-bind="wiring" @node-hover="hover = $event" />
    <p :class="['verdict mono', pre ? 'good' : 'bad']">
      {{ pre ? '主干从头到尾只有加法 → 反传有一条恒等通路' : '主干被 norm 截断 2L 次 → 梯度每次都要被缩放' }}
    </p>

    <!-- L 层摞起来: 每层一根柱, 高度 = 那一层拿到的梯度尺度 -->
    <div class="depth">
      <div class="depth-head mono">{{ flow ? '前向: 主干方差逐层怎么走' : '反传: 第 l 层拿到的梯度尺度 (顶层 = 1)' }}</div>
      <div class="bars" :style="{ gridTemplateColumns: `repeat(${L}, 1fr)` }">
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
      <div class="kv"><span>第 0 层梯度尺度</span><b :class="pre ? 'good' : 'bad'">{{ fmt(series[0]) }}</b></div>
      <div class="kv"><span>顶层 / 底层之比</span><b :class="pre ? 'good' : 'bad'">{{ fmt(1 / series[0]) }}</b></div>
      <div class="kv"><span>每层缩放因子</span><b>{{ pre ? '1.000 (恒等)' : perLayer.toFixed(3) }}</b></div>
      <div v-if="hover" class="kv-note">
        <b>{{ nodeBy(hover).title }}</b>
        <p>{{ nodeBy(hover).desc }}</p>
      </div>
      <p v-else class="lab-note">把鼠标放到任意一个方块上, 看它在这条流水线里干什么。</p>
      <div class="lab-note">
        <p><Tex text="这里的梯度用一个简化模型: 每过一次主干上的 norm, 梯度乘 $1/\sqrt{1+v}$, $v$ 是分支输出相对主干的方差。" /></p>
        <p><Tex text="真实网络里每层的 $v$ 不一样, 但 “Post-LN 要连乘 $L$ 个因子、Pre-LN 有一条恒等通路” 这一点是一样的。" /></p>
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
  const nodes = boxes.value.map((b) => ({ ...b, aria: b.title, active: hover.value === b.id }))
  const adds = SUBLAYERS.map((_, i) => ({ id: `add${i}`, label: '+', kind: 'trunk' }))
  const io = [
    { id: 'x', label: 'x (in)', sub: '[B, T, D]', kind: 'io' },
    { id: 'y', label: 'y (out)', sub: '[B, T, D]', kind: 'io' },
  ]
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

const nodeBy = (id) => boxes.value.find((n) => n.id === id) || {}

// ── 梯度尺度: Pre-LN 主干是恒等; Post-LN 每层过两次 norm ──────────────
const perLayer = computed(() => 1 / Math.sqrt(1 + v.value))
const series = computed(() =>
  range(L.value).map((l) => {
    const above = L.value - 1 - l                 // 它上面还有多少层要穿过
    if (flow.value) return pre.value ? 1 + v.value * (l + 1) : 1   // 前向: Pre-LN 主干方差逐层累加
    return pre.value ? 1 : perLayer.value ** (2 * above)           // 每层两个子层 = 两次 norm
  }))

const fmt = (x) =>
  x === 0 ? '0' : x >= 1000 || x < 0.001 ? x.toExponential(1) : x.toFixed(x < 1 ? 3 : 2)
// 对数标高: 线性画的话 1e-8 和 0 看不出区别
const barH = (g) => {
  const vals = series.value.filter((x) => x > 0)
  const lo = Math.min(...vals), hi = Math.max(...vals)
  if (hi / lo < 1.01) return 100                    // 各层一样高 → 满格, 一眼看出"没有失衡"
  return Math.max(2, (Math.log(g) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * 100)
}
const barColor = (g) => (flow.value ? 'var(--eye)' : g > 0.2 ? 'var(--left)' : g > 1e-3 ? 'var(--warn)' : 'var(--danger)')
</script>

<style scoped>
.verdict { font-size: 11.5px; text-align: center; margin-top: 8px; }
.verdict.good { color: var(--left); }
.verdict.bad { color: var(--danger); }

.depth { margin-top: 14px; }
.depth-head { font-size: 11px; color: var(--text-dim); margin-bottom: 6px; }
.bars { display: grid; gap: 1px; height: 84px; align-items: end; background: var(--code-bg); padding: 6px; border-radius: var(--radius-sm); }
.bar-slot { height: 100%; display: flex; align-items: flex-end; }
.bar { width: 100%; border-radius: 1px 1px 0 0; }
.axis { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim); margin-top: 4px; }
.kv-note { border-left: 2px solid var(--accent); padding-left: 10px; }
.kv-note b { font-size: 13px; color: var(--text); }
.kv-note p { font-size: 12px; color: var(--text-muted); line-height: 1.6; margin-top: 3px; }
</style>
