<!--
  RAG 检索路线对照 (对应 llm_agent/m16_rag/demo.py 的 BM25 / DenseIndex / rrf / reranked / metrics)。
  只讲一件事: 两路召回各有盲区, RRF 融合会把"只有一路召回"的答案压后, rerank 只能在它拿到的前 N 名里把名次排对。
-->
<template>
  <LabFrame
    title="RAG — 20 道题, 四条检索路线"
    sub="每一行是一道标注问答, 格子是名次 1–10, 亮的那格是第一个相关块排在哪。
      切换检索路线, 拖 rerank 范围, 看 recall@1 和 MRR 怎么变。点任意一行, 看这道题的前 5 名和四条路线的名次。"
    module="llm_agent/m16"
    run="python -m llm_agent.m16_rag.demo"
    :challenge="{
      ask: '选「RRF + rerank」, 把 rerank 范围从 10 拖到 5。先猜: 哪道题会掉? 它掉到第几名?',
      answer: '「Can an interrupted upload resume?」直接变成未召回。\n- 融合后排第 8: BM25 没召回它, 只从 dense 一路拿到 1/61。几个两路都排 3–6 名的块各拿两份分数, 反超到前面。\n- rerank 只精读前 N 名: N=5 时第 8 名不在候选里, 打分器再准也看不见。\nrerank 改的是名次, 不是召回。召回阶段的盲区, 要靠多一路召回或调融合权重来补。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="r in ROUTES" :key="r.id" type="button" :class="{ active: route === r.id }" @click="route = r.id">{{ r.label }}</button>
      </div>
      <LabSlider v-model="top" label="rerank 范围 (前 N 名)" :min="1" :max="15" unit=" 名" />
    </template>

    <div class="rows">
      <button v-for="(q, i) in DATA" :key="i" type="button" class="qrow" :class="{ sel: sel === i }" @click="sel = i">
        <span class="qt">{{ q.q }}</span>
        <span class="pips">
          <span v-for="k in 10" :key="k" class="pip" :class="k === ranks[i] ? tone(ranks[i]) : ''" />
        </span>
        <span class="mono rk" :class="{ miss: !ranks[i] }">{{ ranks[i] ? '#' + ranks[i] : '✘' }}</span>
      </button>
    </div>

    <div class="detail">
      <p><b>{{ DATA[sel].q }}</b></p>
      <p class="mono small">前 5 名:
        <span v-for="c in lists[sel].slice(0, 5)" :key="c" class="chip" :class="{ hit: DATA[sel].rel.includes(c) }">{{ IDS[c] }}</span>
        <span v-if="!lists[sel].length" class="miss">BM25 没有任何词命中, 返回空列表</span>
      </p>
      <p class="mono small">四条路线的名次: <span v-for="r in ROUTES" :key="r.id" class="rt">{{ r.label }} {{ rankOf(sel, r.id) || '✘' }}</span></p>
    </div>

    <template #stats>
      <div class="kv"><span>recall@1</span><b :class="cmp(m[0], base[0])">{{ m[0].toFixed(2) }}</b></div>
      <div class="kv"><span>recall@3</span><b :class="cmp(m[1], base[1])">{{ m[1].toFixed(2) }}</b></div>
      <div class="kv"><span>MRR</span><b :class="cmp(m[2], base[2])">{{ m[2].toFixed(2) }}</b></div>
      <div class="kv"><span>未召回</span><b :class="misses ? 'bad' : 'good'">{{ misses }} 题</b></div>
      <div class="lab-note">
        <p>颜色和 BM25 单路比: 绿 = 更好, 红 = 更差。</p>
        <p>名次在前端现算: RRF 用两路的真实排序, rerank 用 Python 算好的 rerank_score。rerank 范围 10 时四条路线和 demo [2] 逐题一致。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const ROUTES = [{ id: 'bm25', label: 'BM25' }, { id: 'dense', label: 'dense' }, { id: 'rrf', label: 'RRF 融合' }, { id: 'rerank', label: 'RRF + rerank' }]
const route = ref('rerank')
const top = ref(10)
const sel = ref(5)

// 数据来自 python -m llm_agent.m16_rag.demo 的同一套结构切块 (15 块) 与 QA:
// bm25 / dense = BM25.rank / DenseIndex.rank 的完整排序 (块下标), rr = rerank_score(q, 每块) 保留 4 位, rel = 相关块。
const IDS = ['uploads#0', 'uploads#1', 'uploads#2', 'sharing#0', 'sharing#1', 'sharing#2', 'security#0', 'security#1', 'security#2', 'billing#0', 'billing#1', 'billing#2', 'sync#0', 'sync#1', 'sync#2']
const DATA = [
  { q: 'What is the maximum file size on a free account?', rel: [0], bm25: [11, 0, 13, 3, 12, 2, 4, 5, 1, 9, 14, 6], dense: [0, 10, 11, 6, 5, 4, 8, 1, 14, 3, 9, 13, 7, 2, 12], rr: [1.175, 0, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0.4, 0, 0.2, 0] },
  { q: 'Why do I get error E429?', rel: [2], bm25: [2, 0], dense: [2, 0, 14, 13, 4, 1, 10, 8, 11, 9, 5, 3, 6, 7, 12], rr: [0.5, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'How are uploads throttled?', rel: [2], bm25: [1, 0, 2, 6, 10, 5, 7], dense: [2, 1, 12, 0, 13, 9, 6, 14, 5, 8, 11, 3, 10, 7, 4], rr: [0.5, 0.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5] },
  { q: 'What does error E413 mean?', rel: [0], bm25: [0, 2], dense: [2, 8, 0, 7, 9, 5, 1, 11, 6, 13, 4, 3, 12, 10, 14], rr: [0.9167, 0, 0.3333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'How big is each part of a resumable upload?', rel: [1], bm25: [1, 13, 11, 0, 10, 14, 3, 5, 4, 2], dense: [1, 0, 13, 14, 2, 6, 12, 11, 10, 8, 9, 3, 7, 5, 4], rr: [0.25, 0.9167, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25] },
  { q: 'Can an interrupted upload resume?', rel: [1], bm25: [7, 14, 0, 13, 3, 10, 4], dense: [1, 0, 14, 8, 13, 3, 2, 5, 6, 4, 9, 11, 12, 7, 10], rr: [0.3333, 0.9167, 0.3333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3333, 0.3333] },
  { q: 'When do shared links expire?', rel: [4], bm25: [4, 3, 13, 2], dense: [4, 3, 6, 5, 11, 1, 0, 13, 2, 9, 8, 10, 14, 7, 12], rr: [0, 0, 0, 0.9167, 1.5, 0.3333, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'How do I revoke a link?', rel: [5], bm25: [3, 5, 4, 11, 13, 2], dense: [5, 0, 11, 3, 7, 2, 1, 9, 13, 6, 8, 4, 14, 12, 10], rr: [0, 0, 0, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'Can I password protect a share link?', rel: [3], bm25: [3, 4, 5, 11, 10, 7, 14, 13, 2, 0], dense: [3, 5, 11, 6, 8, 0, 4, 2, 7, 9, 10, 1, 13, 14, 12], rr: [0, 0, 0, 1.1667, 0.5, 0.6667, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'How often are encryption keys rotated?', rel: [6], bm25: [6, 10, 5, 7, 1, 0], dense: [6, 7, 0, 11, 8, 2, 9, 3, 12, 10, 4, 13, 1, 5, 14], rr: [0, 0, 0, 0, 0, 0, 1.0833, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'Is my data encrypted?', rel: [6], bm25: [6, 11, 13], dense: [6, 9, 11, 8, 12, 3, 0, 4, 7, 13, 14, 10, 1, 5, 2], rr: [0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0] },
  { q: 'How do I set up two-factor authentication?', rel: [7], bm25: [7, 4, 0, 13], dense: [7, 10, 8, 11, 6, 12, 4, 0, 2, 13, 14, 9, 1, 5, 3], rr: [0.2, 0, 0, 0, 0.2, 0, 0, 0.85, 0, 0, 0, 0, 0, 0.2, 0] },
  { q: 'When does an idle session time out?', rel: [8], bm25: [8, 7, 13, 2], dense: [8, 2, 10, 1, 5, 6, 4, 7, 11, 12, 13, 0, 3, 14, 9], rr: [0, 0, 0, 0, 0, 0, 0, 0, 0.9167, 0, 0, 0, 0, 0, 0] },
  { q: 'Can I get a refund on a monthly subscription?', rel: [10], bm25: [10, 3, 12, 4, 11, 5, 7, 14, 13, 2, 0], dense: [10, 6, 4, 14, 2, 7, 5, 8, 12, 0, 9, 13, 3, 11, 1], rr: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3333, 1.25, 0, 0, 0, 0] },
  { q: 'What happens if my payment fails?', rel: [11], bm25: [11, 1], dense: [11, 7, 8, 6, 12, 3, 4, 1, 2, 14, 5, 0, 9, 10, 13], rr: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9167, 0, 0, 0] },
  { q: 'How much does the Pro plan cost per year?', rel: [9], bm25: [9, 0, 4, 2, 13, 1, 3, 12, 14, 11, 6], dense: [9, 12, 0, 6, 7, 13, 3, 2, 5, 4, 10, 11, 1, 14, 8], rr: [0.3333, 0, 0.1667, 0, 0.1667, 0, 0, 0, 0, 0.8667, 0, 0, 0, 0, 0] },
  { q: 'What if two devices edit the same file?', rel: [13], bm25: [13, 7, 0, 1, 11, 3, 9, 12, 14, 4, 6, 2], dense: [13, 5, 0, 7, 11, 14, 6, 1, 8, 10, 3, 4, 2, 12, 9], rr: [0.2, 0, 0, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 1.5, 0] },
  { q: 'How fast can the desktop client upload?', rel: [14], bm25: [14, 12, 13, 0, 1, 3, 4, 10, 7, 9, 11, 6, 2], dense: [12, 14, 1, 0, 13, 7, 6, 10, 11, 9, 2, 4, 8, 5, 3], rr: [0.25, 0.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6667, 0.5, 1.0833] },
  { q: 'Can I keep some folders online only?', rel: [12], bm25: [12, 3, 10, 7, 14, 4, 0], dense: [12, 8, 11, 0, 1, 7, 3, 2, 13, 10, 9, 6, 5, 4, 14], rr: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.85, 0, 0] },
  { q: 'How do I get my money back?', rel: [10], bm25: [], dense: [13, 11, 14, 7, 10, 4, 6, 1, 0, 2, 5, 12, 9, 3, 8], rr: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
]

// ★ RRF: 只用名次; 同分时保留插入顺序 (先 BM25 后 dense), 与 Python Counter.most_common 一致
function rrf(lists, k = 60) {
  const s = new Map()
  for (const l of lists) l.forEach((c, i) => s.set(c, (s.get(c) ?? 0) + 1 / (k + i + 1)))
  return [...s.entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0])
}
function rankList(q, r, n) {
  if (r === 'bm25' || r === 'dense') return q[r]
  const fused = rrf([q.bm25, q.dense])
  if (r === 'rrf') return fused
  return fused.slice(0, n).sort((a, b) => q.rr[b] - q.rr[a]) // ★ 只重排前 n 名, 稳定排序; 后面的直接丢掉
}
const firstHit = (q, list) => { const i = list.findIndex((c) => q.rel.includes(c)); return i < 0 ? 0 : i + 1 }
const metrics = (rs) => [rs.filter((r) => r === 1).length / rs.length, rs.filter((r) => r > 0 && r <= 3).length / rs.length, rs.reduce((a, r) => a + (r ? 1 / r : 0), 0) / rs.length]

const lists = computed(() => DATA.map((q) => rankList(q, route.value, top.value)))
const ranks = computed(() => DATA.map((q, i) => firstHit(q, lists.value[i])))
const m = computed(() => metrics(ranks.value))
const base = metrics(DATA.map((q) => firstHit(q, q.bm25)))
const misses = computed(() => ranks.value.filter((r) => !r).length)
const rankOf = (i, r) => firstHit(DATA[i], rankList(DATA[i], r, top.value))
const tone = (r) => (r === 1 ? 'ok' : r <= 3 ? 'hot' : 'bad')
const cmp = (a, b) => (a > b + 1e-9 ? 'good' : a < b - 1e-9 ? 'bad' : '')
</script>

<style scoped>
.rows { display: flex; flex-direction: column; gap: 2px; }
.qrow { display: flex; align-items: center; gap: 8px; min-height: 0; padding: 3px 6px; border: 1px solid transparent; border-radius: var(--radius-sm); background: none; color: var(--text); text-align: left; }
.qrow.sel { border-color: var(--accent); background: var(--accent-soft); }
.qt { flex: 1; min-width: 0; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pips { display: flex; gap: 2px; flex-shrink: 0; }
.pip { width: 12px; height: 12px; border-radius: 2px; border: 1px solid var(--border); }
.pip.ok { background: var(--left); border-color: var(--left); }
.pip.hot { background: var(--warn); border-color: var(--warn); }
.pip.bad { background: var(--danger); border-color: var(--danger); }
.rk { width: 28px; flex-shrink: 0; font-size: 12px; text-align: right; }
.miss { color: var(--danger); }
.detail { margin-top: 10px; font-size: 12px; line-height: 1.8; color: var(--text-muted); }
.small { font-size: 11px; }
.chip { display: inline-block; margin: 0 4px 2px 0; padding: 0 5px; border: 1px solid var(--border); border-radius: 3px; }
.chip.hit { border-color: var(--left); color: var(--left); }
.rt { display: inline-block; margin-right: 10px; }
</style>
