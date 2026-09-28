<!--
  反向传播实验台 (对应 llm_basic/model.py 的 linear_backward / cross_entropy_forward_backward)。
  一条最短的链: x, W → z = W·x → p = softmax(z) → L = -log p[y]。
  点节点看 "上游梯度 × 本地导数 = 本节点梯度", 链式法则只有这一句话。
-->
<template>
  <LabFrame
    title="反向传播 — 点一个节点, 看梯度怎么传到它"
    sub="- 前向: 从左到右算出 loss (灰色数字)。
      - 反向: 从右到左把 $\partial L/\partial\,\cdot$ 传回来 (粉色数字)。
      点任意节点, 右侧会把它的梯度拆成 '上游梯度 × 本地导数'。拖 x、换目标类、走几步 SGD, 看每个数怎么跟着变。"
    module="llm_basic/model.py"
    run="python llm_basic/gradcheck.py"
    :challenge="{
      ask: '把目标类 y 换成模型当前概率最低的那一类。dz (logits 的梯度) 里哪一项绝对值最大? 为什么 softmax+CE 的梯度永远是 p − onehot 这么干净?',
      answer: '$dz = p - \\mathrm{onehot}(y)$。目标类那一项是 $p_y - 1$: $p_y$ 越小它越接近 $-1$, 绝对值最大。错得越离谱, 推得越狠。\n- 单看 CE 的本地导数: $-1/p_y$, $p_y$ 小时会爆炸。\n- 单看 softmax 的雅可比: $\\mathrm{diag}(p) - pp^\\top$, 很麻烦。\n两者一乘, $1/p_y$ 正好被约掉。所以 llm_basic 把 softmax 和 CE 合并成一个 forward_backward 函数: 又简单又数值稳定。',
    }"
  >
    <template #controls>
      <LabSlider v-model="x1" label="输入 x₁" :min="-2" :max="2" :step="0.1" />
      <LabSlider v-model="x2" label="输入 x₂" :min="-2" :max="2" :step="0.1" />
      <div class="row">
        <span class="hint">目标类 y:</span>
        <button v-for="c in 3" :key="c" type="button" :class="{ active: y === c - 1 }" @click="y = c - 1">类 {{ c - 1 }}</button>
        <button type="button" @click="sgd">走一步 SGD (lr = 0.5)</button>
        <button type="button" @click="reset">重置 W</button>
        <span class="hint mono">已走 {{ steps }} 步</span>
      </div>
    </template>

    <DagView v-bind="graph" @node-click="sel = $event">
      <template #node="{ node }">
        <div class="nm">{{ node.label }}</div>
        <div class="fw mono">{{ node.fw }}</div>
        <div class="bw mono">{{ node.bw }}</div>
      </template>
    </DagView>
    <p class="cap"><Tex text="灰 = 前向值 · 粉 = $\partial L/\partial(\text{该节点})$ · 粉色边 = 梯度传到所选节点走过的路" /></p>

    <template #stats>
      <div class="kv"><span><Tex text="loss $= -\log p[y]$" /></span><b :class="g.L < 0.3 ? 'good' : ''">{{ g.L.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="目标类概率 $p[y]$" /></span><b>{{ g.p[y].toFixed(3) }}</b></div>
      <div class="kv"><span>dW[0][0] 解析 / 数值</span><b :class="relErr < 1e-6 ? 'good' : 'bad'">{{ g.dW[0][0].toFixed(4) }} / {{ numeric.toFixed(4) }}</b></div>
      <div class="detail">
        <b>{{ info.title }}</b>
        <p><span class="k">上游梯度</span><span class="mono"><Tex :text="info.up" /></span></p>
        <p><span class="k">本地导数</span><span class="mono"><Tex :text="info.local" /></span></p>
        <p><span class="k">相乘得到</span><span class="mono res"><Tex :text="info.out" /></span></p>
      </div>
      <p class="lab-note"><Tex :text="info.note" /></p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import DagView from '@/components/dag/DagView.vue'
import { softmax } from '@/utils/labmath.js'

const W0 = [[0.6, -0.4], [-0.3, 0.8], [0.2, 0.1]]
const W = ref(W0.map((r) => [...r]))
const x1 = ref(1), x2 = ref(-0.5), y = ref(1), sel = ref('z'), steps = ref(0)

// 前向 + 反向一次算完, 和 llm_basic 的 forward/backward 成对函数是同一件事
const run = (Wm, x, yi) => {
  const z = Wm.map((r) => r[0] * x[0] + r[1] * x[1])
  const p = softmax(z)
  const L = -Math.log(p[yi])
  const dp = p.map((_, i) => (i === yi ? -1 / p[yi] : 0))
  const dz = p.map((pi, i) => pi - (i === yi ? 1 : 0))         // ★ softmax+CE 合并后的梯度: p − onehot
  const dW = dz.map((d) => [d * x[0], d * x[1]])               // 外积 dz ⊗ x
  const dx = [0, 1].map((j) => Wm.reduce((s, r, i) => s + r[j] * dz[i], 0)) // Wᵀ dz
  return { z, p, L, dp, dz, dW, dx }
}
const x = computed(() => [x1.value, x2.value])
const g = computed(() => run(W.value, x.value, y.value))

// 中心差分验一个元素 —— gradcheck 的最小版本
const numeric = computed(() => {
  const eps = 1e-5
  const at = (d) => run(W.value.map((r, i) => (i === 0 ? [r[0] + d, r[1]] : r)), x.value, y.value).L
  return (at(eps) - at(-eps)) / (2 * eps)
})
const relErr = computed(() => Math.abs(numeric.value - g.value.dW[0][0]) / Math.max(1e-12, Math.abs(numeric.value) + Math.abs(g.value.dW[0][0])))

const sgd = () => { const d = g.value.dW; W.value = W.value.map((r, i) => r.map((w, j) => w - 0.5 * d[i][j])); steps.value++ }
const reset = () => { W.value = W0.map((r) => [...r]); steps.value = 0 }

const v = (a) => '[' + a.map((t) => t.toFixed(2)).join(', ') + ']'
const nodes = computed(() => [
  { id: 'x', label: '输入 x', fw: v(x.value), bw: v(g.value.dx) },
  { id: 'W', label: '权重 W (3×2)', fw: '第0行 ' + v(W.value[0]), bw: '第0行 ' + v(g.value.dW[0]) },
  { id: 'z', label: 'z = W·x (logits)', fw: v(g.value.z), bw: v(g.value.dz) },
  { id: 'p', label: 'p = softmax(z)', fw: v(g.value.p), bw: v(g.value.dp) },
  { id: 'L', label: 'L = −log p[y]', fw: g.value.L.toFixed(3), bw: '1' },
])
// 梯度从 L 传到所选节点经过的边 —— 选中谁, 就把那条回传路径点亮
const PATH = { L: [], p: ['pL'], z: ['pL', 'zp'], x: ['pL', 'zp', 'xz'], W: ['pL', 'zp', 'Wz'] }
const WIRES = [
  { id: 'xz', from: 'x', to: 'z' }, { id: 'Wz', from: 'W', to: 'z' },
  { id: 'zp', from: 'z', to: 'p' }, { id: 'pL', from: 'p', to: 'L' },
]
// 图是左右向的: 前向 x/W → z → p → L, 坐标交给 dagre
const graph = computed(() => ({
  dir: 'LR',
  nodes: nodes.value.map((n) => ({ ...n, kind: 'io', aria: `节点 ${n.label}`, active: sel.value === n.id })),
  // 回传路径画成粉色, 和节点里的 ∂L/∂· 数字同色
  edges: WIRES.map((e) => {
    const on = PATH[sel.value].includes(e.id)
    return { ...e, active: on, color: on ? 'var(--right)' : undefined }
  }),
}))

const info = computed(() => {
  const G = g.value
  return {
    L: { title: 'L: 反向的起点', up: '—', local: '$\\partial L/\\partial L = 1$', out: '1', note: '反向传播永远从标量 loss 对自己的导数 1 出发。' },
    p: { title: 'p ← L', up: '1', local: `$\\partial L/\\partial p[y] = -1/p[y]$ = ${(-1 / G.p[y.value]).toFixed(2)}`, out: v(G.dp), note: '只有目标类那一项有梯度; $p[y]$ 越小, $-1/p[y]$ 越大。单独算这一步数值上很危险。' },
    z: { title: 'z ← p', up: v(G.dp), local: '$J^\\top$, $J = \\mathrm{diag}(p) - pp^\\top$', out: v(G.dz) + ' = $p - \\mathrm{onehot}$', note: 'softmax 的每个输出依赖所有输入, 所以本地导数是一个矩阵 (耦合项)。乘上上游后化简成 $p - \\mathrm{onehot}$, $1/p[y]$ 被约掉了。' },
    W: { title: 'W ← z', up: v(G.dz), local: `$\\partial z_i/\\partial W_{ij} = x_j$ = ${v(x.value)}`, out: '$dz \\otimes x$, 第0行 ' + v(G.dW[0]), note: '$dW = dz \\otimes x$: 这就是 linear_backward 里的 x.T @ dout。把 $x$ 拖到 0, $dW$ 整个变 0: 输入为 0 的权重学不到东西。' },
    x: { title: 'x ← z', up: v(G.dz), local: '$\\partial z/\\partial x = W$', out: '$W^\\top\\cdot dz$ = ' + v(G.dx), note: '$dx = W^\\top dz$: 这份梯度会继续往更前面的层传, 多层网络就是把这一步重复 n_layer 次。' },
  }[sel.value]
})
</script>

<style scoped>
.hint { font-size: 12px; color: var(--text-muted); }
/* 节点内容走 DagView 的 #node 插槽, 这几条作用在插槽里 */
.nm { font-size: 11.5px; font-weight: 600; color: var(--text); }
.fw { font-size: 10.5px; color: var(--text-muted); margin-top: 2px; }
.bw { font-size: 10.5px; color: var(--right); }
.cap { text-align: center; font-size: 11px; color: var(--text-dim); margin-top: 8px; }
.detail { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; display: grid; gap: 4px; }
.detail b { color: var(--text); font-size: 13px; }
.detail p { display: grid; grid-template-columns: 64px 1fr; gap: 6px; }
.detail .k { color: var(--text-dim); }
.detail .res { color: var(--right); }
</style>
