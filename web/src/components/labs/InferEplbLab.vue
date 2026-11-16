<!-- EPLB 实验台 (llm_infer/m21): 路由倾斜 → rank 负载不均 → 冗余专家。放置算法与 moe.py:eplb_placement 相同。 -->
<template>
  <LabFrame
    title="MoE 专家并行 — 热点专家把一张卡堵死, EPLB 用冗余副本把它拆开"
    sub="- 模拟什么: MoE (mixture of experts, 混合专家) 模型的一层有 16 个专家 (各自是一个小前馈网络), 每个 token 由路由器选 2 个专家去算 (top-2)。专家并行 (EP) 把 16 个专家分到 4 张卡 (rank) 上, token 要先发到专家所在的卡 (dispatch), 算完再发回来 (combine)。combine 要等所有卡算完, 所以一步的耗时由最忙的那张卡决定。路由不均匀时, 热门专家所在的卡堵死, 其他卡闲着。EPLB (expert parallelism load balancer) 的做法: 给热门专家多放几个副本 (冗余 slot), 把它的 token 拆给各副本, 再把所有 slot 按负载从重到轻依次放到当前最轻的卡上。这里 2048 个 token, 路由用带 Zipf 偏置的随机 logits (前几个专家天生更热)。
      - 图上是什么: 每根柱子是一张卡 (rank 0 到 3), 柱子里每一段是这张卡上的一个专家 slot, 段的高度是它收到的 token 数, 段上写着专家编号 (E3 ×2 表示这个专家有 2 个副本), 虚线边框的段是有副本的专家。柱子下面是这张卡的总负载, 以及它相对平均值的倍数, 最高的那根标红。横向虚线是平均每卡负载 (2048 × 2 ÷ 4 = 1024)。
      - 可以做什么: 点「连续放置」(E0 到 E3 放 rank 0, E4 到 E7 放 rank 1, 以此类推, 每个专家 1 个 slot) 或「EPLB 贪心放置」切换放置算法; 点「换一组 token」换一组随机路由; 拖「路由倾斜度 s」改路由有多偏 (0 是均匀, 越大越集中在前几个专家); 拖「冗余副本数」改 EPLB 额外加几个副本 (连续放置下不起作用, 滑杆置灰); 点或悬停一段, 高亮这个专家的所有副本, 右侧显示它的数字。
      - 应该看到: 倾斜度 1 时连续放置的 rank 0 一柱独高 (热门的 E0、E1 都在它上面); 切到 EPLB 不加副本, 柱子平了一些但还是不齐, 因为最热的专家一个就超过了平均每卡负载; 加 4 个副本后四根柱子几乎一样高。右侧读数: 「max / mean rank 负载」是最忙的卡相对平均的倍数, 一步耗时正比于它, 越接近 1 越好 (1.1 以下绿, 1.5 以上红); 「平均利用率」是它的倒数, 表示其他卡平均有多忙; 「最热专家 ÷ 平均 rank 负载」大于 1 就意味着不复制专家就不可能均衡; 「物理 slot 数」是 16 加副本数。"
    module="llm_infer/m21"
    run="python -m llm_infer.m21_moe_serving.demo"
    :challenge="{
      ask: '倾斜度 s = 1 时, 只用「贪心放置、0 个副本」能把 max/mean 压到 1.1 以下吗? 看右边「最热专家 ÷ 平均 rank 负载」。',
      answer: '不能。倾斜度 s = 1 时最热的专家 E0 一个就收到约 1.4 倍「平均每卡负载」的 token (右侧「最热专家 ÷ 平均 rank 负载」约 1.4)。专家是不可分的: 无论把它放到哪张卡, 那张卡都至少超载 40%, 所以 max/mean 最好也只能到 1.4 以上 (demo 实测贪心放置 1.48×)。\n重新摆放解决不了「一个专家比一张卡的平均负载还大」的热点。只有复制它, 把它的 token 拆给多个副本 (EPLB 的冗余 slot), 才能把热点切小。\n副本的权重完全相同, 一个 token 走哪个副本结果都一样, 所以模型输出逐位不变。代价是多占几个 slot 的显存, 以及重新放置时要搬专家权重, 真实系统每隔几分钟才做一次。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.name }}</button>
        <button type="button" @click="seed++">换一组 token</button>
      </div>
      <LabSlider v-model="skew" label="路由倾斜度 s (Zipf)" :min="0" :max="2" :step="0.1" />
      <!-- 连续放置下每个专家固定 1 个 slot, 副本数不起作用: 置灰, 不让拖 -->
      <div :inert="mode === 'contig' || null" :class="{ off: mode === 'contig' }">
        <LabSlider v-model="redIdx" label="冗余副本数 (EPLB)" :min="0" :max="2" :format="(v) => RED[v]" />
      </div>
      <p v-if="mode === 'contig'" class="tip">连续放置不复制专家, 副本数滑杆不起作用。先切到「EPLB 贪心放置」。</p>
    </template>

    <svg :viewBox="`0 0 ${W} ${H + 30}`" role="group" aria-label="每张卡上的专家 slot 负载">
      <g v-for="(col, r) in view.cols" :key="r">
        <g
          v-for="s in col.slots" :key="s.slot" role="button" tabindex="0" :aria-label="`专家 ${s.e}, ${s.load} token`"
          class="slot" @click="sel = s.e" @keydown.enter="sel = s.e" @mouseenter="sel = s.e"
        >
          <rect
            :x="r * CW + 14" :y="H - (s.y0 + s.load) * view.sy" :width="CW - 28" :height="Math.max(0.5, s.load * view.sy)"
            :fill="fill(s.e)" :class="{ dim: sel >= 0 && sel !== s.e, rep: s.nRep > 1 }"
          />
          <text v-if="s.load * view.sy > 11" :x="r * CW + CW / 2" :y="H - (s.y0 + s.load / 2) * view.sy + 4" text-anchor="middle" class="lbl in">
            E{{ s.e }}{{ s.nRep > 1 ? ` ×${s.nRep}` : '' }}
          </text>
        </g>
        <text :x="r * CW + CW / 2" :y="H + 14" text-anchor="middle" class="lbl">rank {{ r }} · {{ col.load }}</text>
        <text :x="r * CW + CW / 2" :y="H + 26" text-anchor="middle" class="lbl" :class="{ bad: col.load === view.max }">{{ (col.load / view.mean).toFixed(2) }}×</text>
      </g>
      <line x1="0" :x2="W" :y1="H - view.mean * view.sy" :y2="H - view.mean * view.sy" class="mean" />
      <text :x="W - 2" :y="H - view.mean * view.sy - 4" text-anchor="end" class="lbl">平均 {{ view.mean }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>max / mean rank 负载 (最忙的卡 ÷ 平均, 一步耗时正比于它)</span><b :class="view.ratio > 1.5 ? 'bad' : view.ratio < 1.1 ? 'good' : ''">{{ view.ratio.toFixed(2) }}×</b></div>
      <div class="kv"><span>平均利用率 (mean/max)</span><b :class="view.ratio > 1.5 ? 'bad' : view.ratio < 1.1 ? 'good' : ''">{{ (100 / view.ratio).toFixed(0) }}%</b></div>
      <div class="kv"><span>最热专家 ÷ 平均 rank 负载 (大于 1 就必须复制)</span><b :class="view.hot > 1 ? 'bad' : ''">{{ view.hot.toFixed(2) }}</b></div>
      <div class="kv"><span>物理 slot 数 (16 个专家 + 副本)</span><b>{{ view.nSlot }}</b></div>
      <p class="lab-note">
        <template v-if="sel >= 0">专家 E{{ sel }}: 这一批收到 {{ counts[sel] }} 个 token, 有 {{ view.nRep[sel] }} 个副本, 每个副本分到约 {{ Math.round(counts[sel] / view.nRep[sel]) }} 个。</template>
        <template v-else>点或悬停柱子里的一段, 高亮这个专家的所有副本。</template>
      </p>
      <div class="lab-note">
        <p>为什么看 max 不看 mean: EP 是同步的, 每层 combine 都要等所有 rank 算完, 其它卡都在等最满的那张。平均负载再低也没用。</p>
        <p>放置用的负载统计来自上一批 token (历史), 在新一批上评测 (同 Python demo): 真实系统里放置要搬专家权重, 必须在 batch 到来之前定好。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, argmax, sum } from '@/utils/labmath.js'

const E = 16, R = 4, K = 2, T = 2048, W = 480, H = 240, CW = W / R
const RED = [0, 4, 8]
const MODES = [{ id: 'contig', name: '连续放置 (E0–3 在 rank 0 …)' }, { id: 'eplb', name: 'EPLB 贪心放置' }]
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)']
const fill = (e) => `color-mix(in srgb, ${PALETTE[e % 6]} ${e < 6 ? 75 : e < 12 ? 50 : 30}%, transparent)`

const mode = ref('contig')
const skew = ref(1)
const redIdx = ref(1)
const seed = ref(1)
const sel = ref(-1)

// 路由 logits 的噪声部分 ~ N(0,1), 只跟 seed 走; 两批: [0] 历史 (用来定放置), [1] 新 batch (用来评测)
const noise = computed(() => {
  const rand = mulberry32(seed.value * 104729)
  return range(2).map(() => Float32Array.from({ length: T * E }, () => randn(rand)))
})
const route = (z) => {
  const bias = range(E).map((e) => -skew.value * Math.log(e + 1)) // 专家 e 的先验 ∝ (e+1)^-s, 同 make_layer
  const cnt = Array(E).fill(0)
  for (let t = 0; t < T; t++) {
    const l = range(E).map((e) => z[t * E + e] + bias[e])
    for (let k = 0; k < K; k++) { const e = argmax(l); cnt[e]++; l[e] = -Infinity }
  }
  return cnt
}
const hist = computed(() => route(noise.value[0]))
const counts = computed(() => route(noise.value[1]))

// 与 moe.py:eplb_placement 相同的两步贪心
const placement = computed(() => {
  if (mode.value === 'contig') return { slotExpert: range(E), slotRank: range(E).map((e) => Math.floor(e / (E / R))), nRep: Array(E).fill(1) }
  const load = hist.value, nRep = Array(E).fill(1)
  // ★ ① 副本数: 反复给「每副本负载」最大的专家加一个副本
  for (let i = 0; i < RED[redIdx.value]; i++) nRep[argmax(load.map((l, e) => l / nRep[e]))]++
  const slotExpert = range(E).flatMap((e) => Array(nRep[e]).fill(e))
  const slotLoad = slotExpert.map((e) => load[e] / nRep[e])
  const cap = slotExpert.length / R, rankLoad = Array(R).fill(0), rankFree = Array(R).fill(cap), slotRank = []
  // ② 放置: slot 从重到轻, 依次放到当前最轻且还有空位的 rank
  for (const s of range(slotExpert.length).sort((a, b) => slotLoad[b] - slotLoad[a] || a - b)) {
    const r = argmax(rankLoad.map((l, i) => (rankFree[i] > 0 ? -l : -Infinity)))
    slotRank[s] = r; rankLoad[r] += slotLoad[s]; rankFree[r]--
  }
  return { slotExpert, slotRank, nRep }
})

const view = computed(() => {
  const { slotExpert, slotRank, nRep } = placement.value
  const seen = Array(E).fill(0)
  const cols = range(R).map(() => ({ load: 0, slots: [] }))
  slotExpert.forEach((e, slot) => {
    const j = seen[e]++, c = counts.value[e]
    const load = Math.floor(c / nRep[e]) + (j < c % nRep[e] ? 1 : 0) // 副本间 round-robin 拆 token, 同 ep_forward 的 slot_of
    const col = cols[slotRank[slot]]
    col.slots.push({ slot, e, load, y0: col.load, nRep: nRep[e] })
    col.load += load
  })
  const mean = (T * K) / R, max = Math.max(...cols.map((c) => c.load))
  return { cols, nRep, mean, max, ratio: max / mean, hot: Math.max(...counts.value) / mean, sy: (H - 10) / max, nSlot: slotExpert.length, total: sum(counts.value) }
})
</script>

<style scoped>
.off { opacity: 0.45; }
.tip { font-size: 11px; color: var(--text-dim); }
.slot { cursor: pointer; outline: none; }
.slot rect { stroke: var(--bg-card); stroke-width: 1; }
.slot rect.rep { stroke: var(--text); stroke-dasharray: 3 2; }
.slot rect.dim { opacity: 0.25; }
.slot:focus-visible rect { stroke: var(--accent); stroke-width: 2; }
.mean { stroke: var(--text-muted); stroke-dasharray: 5 4; }
.lbl { font-size: 10px; fill: var(--text-muted); }
.lbl.in { fill: var(--text); pointer-events: none; }
.lbl.bad { fill: var(--danger); }
</style>
