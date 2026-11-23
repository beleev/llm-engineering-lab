<!--
  树形投机实验台 (对应 llm_infer/m19_tree_speculation/tree_spec.py)。
  一件事: 每个节点留 draft 的 top-w 候选长成树, tree attention mask (只看祖先 + 自己) 让 target 一次 forward 验完所有分支。
-->
<template>
  <LabFrame
    title="树形投机 — draft 每层留几个候选长成树, target 用 tree mask 一次验完所有分支"
    sub="- 模拟什么: 链式投机解码在每个未来步仅保留单一候选, 一旦首步猜错后续猜测便全部作废, 而实际模型推理中目标 token 往往就落在草稿模型的第 2、第 3 高分候选内。本实验采用简化的独立命中概率模型: 目标 token 命中草稿第 1 名的概率为 $p_1$, 命中第 2、3 名的概率依次乘以衰减系数 0.4, 且各层相互独立。树形投机解码 (SpecInfer、Medusa、EAGLE 等方法的核心框架) 的执行机制如下:
  1. 草稿模型在每一层每个节点处展开保留 top-$w$ 个分支候选, 逐层生长为一棵候选树。
  2. 目标模型通过单次并行 forward 一次性验证整棵树的所有分支。单次前向得以等价验证多条分支的关键在于构造专用的树形注意力掩码 (tree attention mask): 树中每个节点仅被允许关注上文前缀、其路径上的各级祖先节点以及自身, 同层及异枝的兄弟节点之间彼此不可见, 从而保证任一分支路径算出的 logits 与单独串行解码该路径的结果严格一致。
  3. 验证后采纳从根出发与 target 逐层匹配的最长分支路径。
- 图上是什么: 左右两幅联动视图:
  - 左侧视图: 草稿候选树, 节点按广度优先 (BFS, 逐层从左至右) 编号, 0 号根节点代表已确定的末尾上下文 token, 每向下延伸一层对应未来下一个生成位置的候选分支; 绿色加粗的边与节点标示本轮验证被 target 采纳的最长匹配路径。
  - 右侧视图: 对应的 tree attention mask 矩阵, 每一行对应一个树节点; 最左侧前 3 列代表上文历史前缀 (灰色, 对所有树节点完全可见), 后续各列依次对应各个树节点, 高亮单元格表示该行节点在注意力计算中允许看见对应列节点 (每行高亮位置恰好为其全部祖先与自身)。
  - 联动高亮: 悬停或点击左侧任一树节点或右侧掩码矩阵的任一行, 两侧视图会同步高亮对应元素, 并在下方显示该节点的树深度、共享的 RoPE 位置索引及可见节点列表。
- 可以做什么: 调整候选树拓扑结构与草稿模型准确度:
  - 点击各层对应的「+ / −」按钮修改该层每个父节点展开保留的子分支数 (取值 1 至 3)。
  - 点击「加一层」或「删掉这层」增减候选树最大深度 (最多 3 层), 树拓扑与右侧掩码矩阵会实时重绘。
  - 拖动「draft top-1 命中率 p1」调整草稿模型首选候选的基准准确率。
  - 点击「换一组」重新随机采样本轮 target 实际命中的分支路径。
- 应该看到: 当 $p_1$ 较低时树形结构显著占优, 因为首层保留多个并行分支将首步命中概率从单一的 $p_1$ 大幅抬升; 当将 $p_1$ 拖至 0.9 以上时, 同等验证节点预算下的深层单链反而实现反超, 说明草稿准确率极高时纵向延伸深度比横向铺开宽度收益更大。右侧「树: 期望接受 (公式 / 模拟)」并列展示解析期望值与随机模拟 20000 轮统计的样本均值。其余读数含义如下:
  - 「1 次 target 调用验证节点数」统计整棵候选树包含的节点总数, 即单次 target 并行前向实际打包计算的 token 数量。
  - 「本轮接受」显示当前绿色采纳路径的深度, 额外加 1 代表 target 在匹配路径末端直接产出的纠错或 bonus token。
  - 「链」一栏并列提供两条单链基线对照: 同深度链 (草稿串行步数相同) 与同验证量链 (target 单次验证的节点总数相同), 期望产出更高的一方标绿。"
    module="llm_infer/m19"
    run="python -m llm_infer.m19_tree_speculation.demo"
    :challenge="{
      ask: '默认候选树 [3, 2, 1] 共包含 16 个节点。若与单次同样验证 16 个 token 的长链 ($K = 15$) 对比, 哪一种结构每轮期望接受的 token 更多?',
      answer: '结论取决于草稿模型的基准准确率 $p_1$, 在高低两种准确率区间下优劣次序恰好相反:\n- 在默认中等准确率 ($p_1 = 0.5$) 下: 树 [3, 2, 1] 每轮期望接受约 1.60 个 token, 显著优于长链 $K = 15$ 的约 1.00 个。单链结构缺乏备选分支, 一旦首步猜错后续 14 个候选便全部作废; 首步命中率仅为 $p_1 = 0.5$, 此后每加深一层存活率再乘 0.5, 级数求和 $\\sum_i 0.5^i$ 的期望接受数仅约为 1。而候选树通过横向分支有效抬升了前序关键层的通过率:\n  - 第一层: 最易发生截断的瓶颈层保留 3 个候选, 层通过率从 0.5 提升至 $0.5 + 0.2 + 0.08 = 0.78$。\n  - 第二层: 保留 2 个候选, 层通过率为 0.70。\n  - 第三层: 保留 1 个候选, 层通过率为 0.5。\n  - 累计期望接受数为 $0.78 + 0.78 \\times 0.70 + 0.78 \\times 0.70 \\times 0.5 \\approx 1.60$。\n- 将「draft top-1 命中率 p1」拖至 0.9 的高准确率区间时: 长链 $K = 15$ 的期望接受数跃升至约 7.1, 大幅反超候选树的 2.90。当草稿首选准确率本身已极高时, 同层额外分支的边际收益很小, 而深度仅为 3 层的树单轮最多只能接受 3 个草稿节点; 深度达 15 步的长链则能连续命中十余个位置。这一对比表明高置信度时纵向深度优于横向宽度, 低置信度时横向宽度优于纵向深度, 这也正是 EAGLE-2 根据草稿模型局部置信度动态决定树生长拓扑的原因。\n单次 target 前向能够无损并行验证整棵候选树依赖两项关键对齐规则:\n- 共享位置编码: 同一树深度的所有兄弟节点实为同一未来生成位置上的互斥候选 (验证后至多保留其一), 因此共享完全相同的 RoPE 位置索引 $n_{\\text{ctx}} + \\text{depth}$。\n- 树形因果掩码: 每个节点仅能关注上下文前缀、其祖先节点及自身, 兄弟分支之间严格屏蔽。由此保证矩阵中每一行算出的 logits 均与单独将「上下文 + 根到该节点的单链」送入模型串行解码的结果逐位相等。\n在真实模型权重的 Python demo (以注入 10% 权重噪声的 target 作为 draft) 中: 树 [3, 2, 1] 每次 target 调用平均产出 2.58 个 token, 优于同深度链 $K = 3$ 的 1.86 个及同验证量长链 $K = 15$ 的 2.13 个。',
    }"
  >
    <template #controls>
      <div v-for="(w, d) in widths" :key="d" class="row">
        <span class="wl">第 {{ d + 1 }} 层每节点孩子数</span>
        <button type="button" :disabled="w <= 1" :aria-label="`第 ${d + 1} 层减少`" @click="widths[d]--">−</button>
        <b class="mono wv">{{ w }}</b>
        <button type="button" :disabled="w >= 3" :aria-label="`第 ${d + 1} 层增加`" @click="widths[d]++">+</button>
        <button v-if="d === widths.length - 1 && d > 0" type="button" @click="widths.pop()">删掉这层</button>
      </div>
      <div class="row">
        <button type="button" :disabled="widths.length >= 3" @click="widths.push(1)">加一层</button>
        <button type="button" @click="seed++">换一组 (重新抽 target 的选择)</button>
      </div>
      <LabSlider v-model="p1" label="draft top-1 命中率 p1" :min="0.1" :max="0.95" :step="0.05" :format="(t) => t.toFixed(2)" />
    </template>

    <div class="pair">
      <svg :viewBox="`0 0 360 ${60 + widths.length * 56}`" class="tree" role="group" aria-label="draft token 树: 绿色路径是本轮被 target 接受的">
        <line v-for="i in tree.n - 1" :key="`e${i}`" :x1="pos[tree.parents[i]].x" :y1="pos[tree.parents[i]].y" :x2="pos[i].x" :y2="pos[i].y"
          :class="['edge', { ok: onPath(i) }]" />
        <g v-for="(p, i) in pos" :key="i" tabindex="0" role="button" :aria-label="`节点 ${i}`" class="node"
          :class="{ ok: onPath(i), anc: focus >= 0 && tree.anc[focus][i], me: focus === i }"
          @mouseenter="hover = i" @mouseleave="hover = -1" @focus="hover = i" @blur="hover = -1"
          @click="pin = pin === i ? -1 : i" @keydown.enter="pin = pin === i ? -1 : i">
          <circle :cx="p.x" :cy="p.y" :r="R" />
          <text v-if="tree.n <= 22" :x="p.x" :y="p.y + 3">{{ i }}</text>
        </g>
      </svg>
      <svg :viewBox="`0 0 ${(NCTX + tree.n) * C + 4} ${tree.n * C + 4}`" class="mask" role="group" aria-label="tree attention mask: 一行一个树节点, 亮格表示这一行能看到那一列">
        <g v-for="i in tree.n" :key="i" @mouseenter="hover = i - 1" @mouseleave="hover = -1" @click="pin = pin === i - 1 ? -1 : i - 1">
          <rect v-for="j in NCTX" :key="`c${j}`" :x="(j - 1) * C + 2" :y="(i - 1) * C + 2" :width="C - 1" :height="C - 1" class="m ctx" :class="{ row: focus === i - 1 }" />
          <rect v-for="j in tree.n" :key="j" :x="(NCTX + j - 1) * C + 2" :y="(i - 1) * C + 2" :width="C - 1" :height="C - 1"
            class="m" :class="{ vis: tree.anc[i - 1][j - 1], row: focus === i - 1 }" />
        </g>
      </svg>
    </div>
    <p class="read mono">{{ readout }}</p>

    <template #stats>
      <div class="kv"><span>1 次 target 调用一起验证的节点数 (树有多少个节点)</span><b>{{ tree.n }}</b></div>
      <div class="kv"><span>本轮接受的 draft token (+1 个 target 给的纠错或 bonus)</span><b>{{ path.length - 1 }} + 1</b></div>
      <div class="kv"><span>树: 每轮期望接受数 (公式 / 模拟 {{ ROUNDS }} 轮)</span><b :class="eTree >= chain(tree.n - 1) ? 'good' : ''">{{ eTree.toFixed(2) }} / {{ emp.toFixed(2) }}</b></div>
      <div class="kv"><span>链的对照: 同深度 K={{ widths.length }} (draft 调用一样多) / 同验证量 K={{ tree.n - 1 }} (验证 token 一样多)</span><b :class="chain(tree.n - 1) > eTree ? 'good' : ''">{{ chain(widths.length).toFixed(2) }} / {{ chain(tree.n - 1).toFixed(2) }}</b></div>
      <div class="lab-note">
        <p>玩具命中模型 (每层独立):</p>
        <ul class="pts">
          <li><Tex text="目标 token 命中草稿第 $r$ 候选的概率设为 $p_1 \cdot 0.4^{r-1}$ (前三名概率之和超过 1 时截断至 1)。" /></li>
          <li><Tex text="单个父节点保留 $w$ 个子分支时, 该层的单步命中概率 $H(w)$ 等于前 $w$ 名候选的命中概率之和。" /></li>
          <li><Tex text="匹配路径延伸至第 $d$ 层要求前序每一层均命中, 因此总期望接受数为 $\sum_d \prod_{i \le d} H(w_i)$; 链式投机即为每层分支数恒取 $w = 1$ 的特例, 此时 $H(1) = p_1$。" /></li>
        </ul>
        <p>同验证量的链要 {{ tree.n - 1 }} 次串行的 draft 调用, 树只要 {{ widths.length }} 次 (一层一次, 每次同时出几个候选)。</p>
        <p>多验的 token 在 decode 带宽受限时几乎免费 (读一遍权重顺便多算几个 token), 大 batch 算力受限时就不免费了。真实 demo (draft 是加噪 10% 的 target): 树 [3,2,1] 每次 target 调用 2.58 个 token, 链 <Tex text="$K=3$" /> 1.86, 链 <Tex text="$K=15$" /> 2.13。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import Tex from '@/components/Tex.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const NCTX = 3, ROUNDS = 20000, RHO = 0.4
const widths = reactive([3, 2, 1])
const p1 = ref(0.5), seed = ref(1), hover = ref(-1), pin = ref(-1)

// 与 tree_shape / ancestor_matrix 逐行对应: BFS 编号, 节点 0 = 根 (已确定的 out[-1])
const tree = computed(() => {
  const parents = [-1], depth = [0]
  let level = [0]
  widths.forEach((w, d) => {
    const nxt = []
    for (const p of level) for (let c = 0; c < w; c++) { nxt.push(parents.length); parents.push(p); depth.push(d + 1) }
    level = nxt
  })
  const n = parents.length
  const anc = range(n).map((i) => range(n).map((j) => i === j))
  for (let i = 1; i < n; i++) anc[parents[i]].forEach((v, j) => { if (v) anc[i][j] = true }) // ★ anc[i] |= anc[parent]: 父先于子, 一遍即闭包
  const kids = range(n).map((i) => range(n).filter((j) => parents[j] === i))
  return { parents, depth, n, anc, kids }
})
const focus = computed(() => { const f = hover.value >= 0 ? hover.value : pin.value; return f < tree.value.n ? f : -1 })

// 命中率: 第 r 名 = p1·ρ^(r−1), 总和截到 1
const hits = computed(() => { let left = 1; return range(3).map((r) => { const h = Math.min(p1.value * RHO ** r, left); left -= h; return h }) })
const H = (w) => sum(hits.value.slice(0, w))
const eTree = computed(() => { let prod = 1, e = 0; for (const w of widths) { prod *= H(w); e += prod } return e })
const chain = (k) => sum(range(k).map((d) => p1.value ** (d + 1)))

// 一轮: 从根往下, target 的 token 是第几名 → 有这个孩子就进去, 否则断 (同 accept_tree)
const walk = (rand) => {
  const path = [0]
  for (const w of widths) {
    const u = rand()
    let r = 0, cum = hits.value[0]
    while (r < 3 && u >= cum) { r++; cum += hits.value[r] ?? 0 }
    if (r >= w) break
    path.push(tree.value.kids[path[path.length - 1]][r])
  }
  return path
}
const path = computed(() => walk(mulberry32(seed.value * 101 + 7)))
const emp = computed(() => { const rand = mulberry32(seed.value * 977 + 3); let t = 0; for (let i = 0; i < ROUNDS; i++) t += walk(rand).length - 1; return t / ROUNDS })
const onPath = (i) => path.value.includes(i)

// 画树: 叶子均匀铺开, 内部节点取孩子中点
const R = computed(() => (tree.value.n > 22 ? 5 : 9)), C = computed(() => (tree.value.n > 22 ? 8 : 14))
const pos = computed(() => {
  const { n, kids, depth } = tree.value, xs = Array(n).fill(0)
  const leaves = range(n).filter((i) => !kids[i].length)
  leaves.forEach((i, k) => { xs[i] = 20 + (320 * (k + 0.5)) / leaves.length })
  for (let i = n - 1; i >= 0; i--) if (kids[i].length) xs[i] = sum(kids[i].map((k) => xs[k])) / kids[i].length
  return range(n).map((i) => ({ x: xs[i], y: 24 + depth[i] * 56 }))
})
const readout = computed(() => {
  const i = focus.value, t = tree.value
  if (i < 0) return `悬停或点一个节点 (或 mask 的一行): 看它的深度、RoPE 位置和能看见谁。mask 前 ${NCTX} 列是上下文 (全可见), 后 ${t.n} 列是树节点。`
  const a = range(t.n).filter((j) => t.anc[i][j] && j !== i)
  return `节点 ${i}: 深度 ${t.depth[i]} → RoPE 位置 = n_ctx + depth = ${NCTX} + ${t.depth[i]} = ${NCTX + t.depth[i]}; 可见 = 上下文 + 祖先 [${a.join(', ')}] + 自己 (共 ${NCTX + a.length + 1} 列)`
})
</script>

<style scoped>
.wl { font-size: 12px; color: var(--text-muted); min-width: 130px; }
.wv { min-width: 16px; text-align: center; color: var(--accent); }
.pair { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.tree { flex: 1 1 260px; min-width: 220px; }
.mask { flex: 0 1 250px; min-width: 180px; }
.edge { stroke: var(--border-strong); stroke-width: 1.2; }
.edge.ok { stroke: var(--left); stroke-width: 2.5; }
.node { cursor: pointer; outline: none; }
.node circle { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 1.2; }
.node.ok circle { fill: color-mix(in srgb, var(--left) 30%, transparent); stroke: var(--left); }
.node.anc circle { stroke: var(--accent); stroke-width: 2.5; }
.node.me circle, .node:focus-visible circle { stroke: var(--warn); stroke-width: 3; }
.node text { font-size: 9px; fill: var(--text); text-anchor: middle; pointer-events: none; }
.m { fill: var(--bg-elev); stroke: var(--border); stroke-width: 0.5; }
.m.ctx { fill: color-mix(in srgb, var(--text-dim) 35%, transparent); }
.m.vis { fill: var(--accent); }
.m.row { stroke: var(--warn); stroke-width: 1.5; }
.read { font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.6; }
</style>
