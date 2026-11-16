<!--
  树形投机实验台 (对应 llm_infer/m19_tree_speculation/tree_spec.py)。
  一件事: 每个节点留 draft 的 top-w 候选长成树, tree attention mask (只看祖先 + 自己) 让 target 一次 forward 验完所有分支。
-->
<template>
  <LabFrame
    title="树形投机 — draft 每层留几个候选长成树, target 用 tree mask 一次验完所有分支"
    sub="- 模拟什么: 链式投机解码 (draft 连猜 $K$ 个) 的弱点是第一个猜错后面全废, 而 draft 的第 2、第 3 候选经常是对的。树形投机 (SpecInfer、Medusa、EAGLE 都是这个套路) 让 draft 在每个节点保留 top-$w$ 个候选, 一层层长成一棵树, target 用一次 forward 把整棵树所有分支一起验证, 接受和 target 一致的最长路径。能一次验整棵树靠的是 tree attention mask: 树里每个节点只能看到上下文、自己的祖先和自己, 兄弟之间互相看不见, 于是每条路径算出来的 logits 和单独顺序跑这条路径完全一样。这里用一个玩具命中模型: target 的 token 落在 draft 第 1 名的概率是 $p_1$, 第 2、3 名依次乘 0.4, 每层独立。
      - 图上是什么: 左边是 draft 树, 节点按 BFS 编号 (一层一层从左到右), 0 号是根 (已经确定的上一个 token), 每往下一层是下一个位置的候选; 绿色加粗的边和节点是本轮被 target 接受的路径。右边是 tree attention mask, 一行一个树节点, 前 3 列是上下文 (灰色, 全可见), 后面每列对应一个树节点, 亮的格子表示「这一行的节点能看到这一列的节点」, 每行亮的恰好是它的祖先和自己。悬停或点左边任一节点, 或 mask 的任一行, 两边同步高亮, 下面一行写出它的深度、RoPE 位置和能看见谁。
      - 可以做什么: 每一层一行「+ / −」改这一层每个节点留几个孩子 (1 到 3), 「加一层」「删掉这层」改树的深度 (最多 3 层), 树和 mask 立刻重画。拖「draft top-1 命中率 p1」改 draft 的准确度。「换一组」重抽本轮 target 实际选了哪条路径。
      - 应该看到: 右侧「1 次 target 调用验证节点数」是树的节点数, 也就是这次 forward 一起算了多少个 token; 「本轮接受」是绿色路径的长度, 加 1 是 target 顺手给出的纠错或 bonus token; 「树: 期望接受 (公式 / 模拟)」是按玩具模型算的每轮平均接受数和模拟 20000 轮的均值; 「链」给出两条对照: 同深度的链 (draft 调用次数相同) 和同验证量的链 (验证的 token 数相同), 哪边高哪边标绿。$p_1$ 低时树明显赢, 因为第一层多留的备胎把「首个 token 猜对」的概率从 $p_1$ 抬高了; $p_1$ 拖到 0.9 以上时同验证量的长链反超, 因为 draft 够准时深度比宽度值钱。"
    module="llm_infer/m19"
    run="python -m llm_infer.m19_tree_speculation.demo"
    :challenge="{
      ask: '默认树 [3,2,1] 有 16 个节点。同样验 16 个 token 的链 (K=15) 和它比, 谁每轮接受得多?',
      answer: '看 draft 准不准, 两种情况答案相反。\n- 默认 $p_1 = 0.5$: 树每轮约接受 1.60 个, 链 $K = 15$ 约 1.00 个。链是一条路走到黑, 第一个猜错后面 14 个全废; 首个 token 只有 $p_1 = 0.5$ 的命中率, 之后每深一层再乘 0.5, 期望接受数 $\\sum_i 0.5^i \\approx 1$。树在最容易错的第一层留了 3 个候选, 命中率从 0.5 抬到 $0.5 + 0.2 + 0.08 = 0.78$; 第二层 2 个候选 0.70, 第三层 1 个 0.5; 期望接受数 $0.78 + 0.78 \\times 0.70 + 0.78 \\times 0.70 \\times 0.5 \\approx 1.60$。\n- 把「draft top-1 命中率」拖到 0.9: 反过来, 链 $K = 15$ 约 7.1, 远超树的 2.90。draft 本来就很准时, 第一层的备胎几乎用不上, 而树最深只有 3 层, 每轮最多接受 3 个; 链能一路接受十几个。draft 足够准时深度比宽度值钱, 所以 EAGLE-2 按 draft 的置信度动态决定往哪个方向长。\n树能一次验完靠两条规则:\n- 位置: 同一层的兄弟节点是同一个位置的不同候选, 最终只有一个会留下, 所以它们的 RoPE 位置相同, 都是 $n_{\\text{ctx}} + \\text{depth}$。\n- mask: 每个节点只看到上下文、祖先和自己, 兄弟之间互相看不见。这样每一行算出的 logits 恰好等于「上下文 + 从根到它这条链」顺序跑一遍的结果, 一次 forward 等价于每条路径各跑一次顺序 decode。\n真实 demo (draft 是加噪 10% 的 target): 树 [3,2,1] 每次 target 调用 2.58 个 token, 链 $K = 3$ 1.86, 链 $K = 15$ 2.13。',
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
          <li><Tex text="target 的 token 落在 draft 第 $r$ 名的概率 $= p_1 \cdot 0.4^{r-1}$ (三名之和超过 1 就截到 1)。" /></li>
          <li><Tex text="一个节点留 $w$ 个孩子, 这一层猜中的概率 $H(w) =$ 前 $w$ 名概率之和。" /></li>
          <li><Tex text="要走到第 $d$ 层, 前面每一层都得猜中, 所以期望接受数 $= \sum_d \prod_{i \le d} H(w_i)$。链就是每层 $w = 1$ 的特例, $H(1) = p_1$。" /></li>
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
