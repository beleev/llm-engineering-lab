<!--
  反向传播实验台 (对应 llm_basic/model.py 的 linear_backward / cross_entropy_forward_backward)。
  一条最短的链: x, W → z = W·x → p = softmax(z) → L = -log p[y]。
  点节点看 "上游梯度 × 本地导数 = 本节点梯度", 链式法则只有这一句话。
-->
<template>
  <LabFrame
    title="反向传播 — 点一个节点, 看 loss 的梯度怎么一步步传回它"
    sub="- 模拟什么: 最小三分类线性网络的前向推理、反向传播求导与单步参数更新全过程:
  1. 线性投影: 二维输入向量 $x$ (含 2 个分量) 与权重矩阵 $W$ (维度为 $3 \times 2$) 相乘, 输出 3 个未归一化得分 $z$ (即 logits, 每个候选类别对应一个分值)。
  2. 概率归一化: 经由 softmax 算子将得分映射为总和等于 1 的类别概率分布 $p$。
  3. 交叉熵损失计算: 由 $L = -\log p[y]$ 度量真实目标类 $y$ 的预测负对数似然, 目标类分配到的概率越低则损失 $L$ 越大。
  4. 链式法则反向传播: 为降低损失 $L$, 需计算损失对各权重参数的偏导数即梯度 $\partial L / \partial W$, 以确定参数更新的方向与步幅。反向传播从标量损失节点出发沿前向计算图逆序回传, 每经过一个算子节点便将上游传回的梯度与该节点的本地偏导数相乘。源码 llm_basic/model.py 中各算子的 backward 函数实现的正是这一局部链式求导过程。
- 图上是什么: 从左至右展开的前向计算图, 数据流依次由 $x$ 与 $W$ 汇入 $z$, 再经 $p$ 最终算出标量损失 $L$:
  - 灰色数值: 前向传播阶段在各节点计算出的激活值或损失值。
  - 粉色数值: 反向传播阶段算出的标量损失对该节点的梯度 $\partial L / \partial(\text{节点})$。
  - 粉色高亮连线: 点击选中某节点后, 标示梯度信号从末端 $L$ 逆向回传至该节点所流经的依赖路径。
- 可以做什么: 按界面从上到下依次交互:
  - 拖动滑杆 x₁、x₂: 调整二维输入特征的数值。
  - 点击「类 0 / 1 / 2」: 切换真实目标类别标签 $y$。
  - 点击「走一步 SGD」: 按当前梯度执行一次随机梯度下降更新 (学习率取 0.5, 即令 $W$ 减去 0.5 倍梯度), 观察损失下降过程; 点击「重置 W」可恢复初始权重矩阵。
  - 点击图上任意计算节点: 右侧面板将该节点梯度拆解为「上游梯度 × 本地导数」的显式乘积, 其中上游梯度为损失对紧邻下游节点的已知梯度, 本地导数为当前节点输出对输入的局部导数。
- 应该看到: 在默认输入 $x = (1, -0.5)$ 且目标类别 $y = 1$ 时, 预测概率约为 $p \approx (0.57, 0.13, 0.30)$, 目标类概率仅为 0.128, 对应损失值为 2.057。连续点击数次「走一步 SGD」后, 损失值单调下降且目标类概率 $p[y]$ 稳步上升。此外可重点核对两处核心性质:
  - logits 层的梯度满足 $dz = p - \mathrm{onehot}(y)$: 目标类分量为 $p[y] - 1$ (恒为负值, 经负梯度更新后将推高该类得分), 其余非目标类分量直接等于 $p$ 本身 (恒为正值, 更新时压低错误类别得分)。
  - 右侧「dW[0][0] 反向算的 / 差商验的」两项读数应当高度吻合: 前者由手写反向传播解析算出, 后者通过对权重元素 W[0][0] 施加微小扰动并重算前向损失由有限差商近似得出 (即下一实验台 gradcheck 所演示的数值梯度校验), 两者一致即证明反向求导逻辑正确无误。"
    module="llm_basic/model.py"
    run="python llm_basic/gradcheck.py"
    :challenge="{
      ask: '将目标类别 $y$ 切换为模型当前预测概率最低的那一类。此时 logits 梯度 $dz$ 中哪一分量的绝对值最大? 为什么 softmax 与交叉熵 (CE) 组合后的反向梯度总能化简为 $p - \\mathrm{onehot}(y)$ 这一简洁形式?',
      answer: '真实目标类对应的那一分量绝对值最大。\n- 幅度对比: 由 $dz = p - \\mathrm{onehot}(y)$ 可知, 目标类分量为 $p_y - 1$, 当预测概率 $p_y$ 极低时该项逼近 $-1$; 而其余所有非目标类分量分别为各自的预测概率 $p_i$, 它们之和等于 $1 - p_y$ (小于 1), 单项绝对值必然小于 $|p_y - 1|$。模型对正确类别漏判得越严重, 反向传播将该类得分向上拉升的梯度幅值就越大。\n- 代数相消原理: 若将两步拆开独立求导, 末端负对数似然损失的本地导数为 $-1 / p_y$, 在 $p_y$ 趋近于 0 时数值会急剧放大甚至上溢; 而前置 softmax 算子的每个输出分量均依赖全部输入分量, 其本地导数是一个稠密的雅可比 (Jacobian) 矩阵 $\\mathrm{diag}(p) - p p^\\top$。\n- 算子融合优势: 按链式法则将上述两步的局部导数相乘后, 分母中的 $p_y$ 因子恰好与雅可比矩阵中的对应项代数约消, 最终化简为极其规整的逐元素减法 $p - \\mathrm{onehot}(y)$。因此, 源码 llm_basic 将 softmax 与交叉熵封装为单一融合算子 cross_entropy_forward_backward, 在反向阶段直接按 $(p - \\mathrm{onehot}(y)) / N$ 计算梯度, 既省去了显式构造雅可比矩阵的开销, 又彻底消除了除以极小概率 $p_y$ 引发的浮点溢出风险。',
    }"
  >
    <template #controls>
      <LabSlider v-model="x1" label="输入 x₁" :min="-2" :max="2" :step="0.1" />
      <LabSlider v-model="x2" label="输入 x₂" :min="-2" :max="2" :step="0.1" />
      <div class="row">
        <span class="hint">目标类 y (正确答案):</span>
        <button v-for="c in 3" :key="c" type="button" :class="{ active: y === c - 1 }" @click="y = c - 1">类 {{ c - 1 }}</button>
        <button type="button" @click="sgd">走一步 SGD (W 减去 0.5 × 梯度)</button>
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
    <p class="cap"><Tex text="灰色数字 = 前向算出的值 · 粉色数字 = loss 对该节点的梯度 $\partial L/\partial(\text{该节点})$ · 粉色边 = 梯度从 L 传到所选节点走过的路" /></p>

    <template #stats>
      <div class="kv"><span><Tex text="loss $= -\log p[y]$ (越小越好)" /></span><b :class="g.L < 0.3 ? 'good' : ''">{{ g.L.toFixed(3) }}</b></div>
      <div class="kv"><span><Tex text="正确答案的概率 $p[y]$" /></span><b>{{ g.p[y].toFixed(3) }}</b></div>
      <div class="kv"><span>dW[0][0] 反向算的 / 差商验的</span><b :class="relErr < 1e-6 ? 'good' : 'bad'">{{ g.dW[0][0].toFixed(4) }} / {{ numeric.toFixed(4) }}</b></div>
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
    p: { title: 'p ← L', up: '1', local: `$\\partial L/\\partial p[y] = -1/p[y]$ = ${(-1 / G.p[y.value]).toFixed(2)}`, out: v(G.dp), note: '只有目标类那一项有梯度; $p[y]$ 越小, $-1/p[y]$ 越大。单独算这一步容易数值溢出。' },
    z: { title: 'z ← p', up: v(G.dp), local: '$J^\\top$, $J = \\mathrm{diag}(p) - pp^\\top$', out: v(G.dz) + ' = $p - \\mathrm{onehot}$', note: 'softmax 的每个输出依赖所有输入, 所以本地导数是一个矩阵 (耦合项)。乘上上游后化简成 $p - \\mathrm{onehot}$, $1/p[y]$ 被约掉了。' },
    W: { title: 'W ← z', up: v(G.dz), local: `$\\partial z_i/\\partial W_{ij} = x_j$ = ${v(x.value)}`, out: '$dz \\otimes x$, 第0行 ' + v(G.dW[0]), note: '$dW = dz \\otimes x$, 对应 linear_backward 里的 x.T @ dout。把 $x$ 拖到 0, $dW$ 整个变 0: 输入为 0 的权重学不到东西。' },
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
