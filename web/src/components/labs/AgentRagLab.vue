<!--
  RAG 检索路线对照 (对应 llm_agent/m16_rag/demo.py 的 BM25 / DenseIndex / rrf / reranked / metrics)。
  只讲一件事: 两路召回各有盲区, RRF 融合会把"只有一路召回"的答案压后, rerank 只能在它拿到的前 N 名里把名次排对。
-->
<template>
  <LabFrame
    title="RAG — 同样 20 道题, 四条检索路线各把正确答案排到第几名"
    sub="- 模拟什么: RAG (检索增强生成) 管线中的候选检索与重排阶段。系统将一份产品文档切分为 15 个文本块 (chunk), 在收到用户提问时先从中检索出最相关的若干 chunk 传入模型上下文。评测集包含 20 道人工标注了标准答案 chunk 的测试问题, 横向对比四条检索技术路线:
  - BM25: 基于词面精确匹配与词频统计打分, 对 E413 等低频精确错误码极为敏锐, 但无法关联 resume 与 resumable 等词形变体。
  - dense: 将文本映射为稠密向量并计算余弦相似度; 本 demo 采用基于字符 n-gram 构造的轻量级玩具向量, 能够容忍拼写与词缀变体, 但不具备跨词汇的语义同义理解能力。
  - RRF 融合: 倒数排名融合 (Reciprocal Rank Fusion), 忽略两路打分的绝对量纲, 仅按各自排序名次合并, 单路对某候选块的贡献分为 $1 / (60 + \text{名次})$。
  - RRF + rerank: 在双路融合排序的基础上, 截取前 $N$ 名候选交由计算开销更大但判别更精细的重排器重新打分排序。
- 图上是什么: 每行对应评测集中的一道测试问题:
  - 右侧 10 个方格: 依次对应排名第 1 至第 10 位, 高亮方格指示该题首个标准相关块在当前路线下的实际排位。绿色表示排在第 1 名, 黄色表示排在第 2 至 3 名, 红色表示落入第 4 至 10 名。
  - 行末标记: 直接显示该相关块的具体名次数字, ✘ 表示前 10 名候选内未包含相关块 (即召回失败)。
  - 点击任意问题行: 下方面板列出该题在当前路线下的前 5 名召回片段 (绿框标出真实相关块), 并并列展示四条路线各自给出的排位以便逐题比对。
- 可以做什么: 操作顶部路线切换按钮与重排截断滑杆:
  - 第一排按钮: 在四条检索路线之间切换。
  - 「rerank 范围」滑杆: 仅在选中「RRF + rerank」路线时显示。重排器仅对 RRF 融合结果的前 $N$ 名候选执行精排并直接丢弃第 $N$ 名之后的块, 拖动滑杆可观察截断窗口 $N$ 过小时引发的漏召回现象; 其余三条路线不涉及重排, 该滑杆自动隐藏。
- 应该看到: 融合与重排前后的排位变化及右侧评测读数:
  - 「RRF + rerank」(当 $N$ 取 10 时): 20 道题中有 19 题将标准相关块排在第 1 位, 唯独同义改写问句「How do I get my money back?」排在第 5 位, 原因是原文表述为 refund, 而当前四条无神经网络语义编码的路线均无法识别该组同义映射。
  - 单独观察「RRF 融合」: 问句「Can an interrupted upload resume?」的相关块跌至第 8 位。由于 BM25 未能召回该块, 它仅获得 dense 单路赋予的最高分, 在求和时反而被两路排名均居中游的无关块反超; 这种单路强信号被双路平庸信号稀释的现象是名次融合的典型副作用, 需依赖后续 rerank 重新提回首位。
  - 右侧 recall@1: 标准相关块排在第 1 名的题目占比。
  - recall@3: 标准相关块进入前 3 名的题目占比。
  - MRR: 平均倒数排名, 即所有题目「1 / 名次」的算术均值 (前 10 名未召回记为 0)。
  - 指标着色规则: 上述三项指标均为数值越大越优, 颜色以单路 BM25 为基线对比, 优于基线标绿, 劣于基线标红。
  - 「未召回」: 前 10 名结果中完全缺失相关块的题目数量。"
    module="llm_agent/m16"
    run="python -m llm_agent.m16_rag.demo"
    :challenge="{
      ask: '选择「RRF + rerank」路线, 将「rerank 范围」滑杆从 10 调小至 5。先预测: 哪道测试题的排名会发生跌落? 它的最终排位会变为多少?',
      answer: '问句「Can an interrupted upload resume?」会直接变为未召回状态 (✘)。\n- 单路偏科原因: 该题中查询词 resume 与文档词 resumable 词面不一致, 导致 BM25 完全未能命中目标块; 目标块仅在 dense 单路排在第 1 位, 获得的 RRF 融合分数仅为单路贡献 $1 / (60 + 1)$。\n- 融合排名稀释: 另有若干非相关块在 BM25 与 dense 两路中均排在第 3 至第 6 名之间, 两路倒数名次分累加后反超单路满分, 将真实相关块在 RRF 融合列表中挤到了第 8 位。\n- 截断窗口限制: 重排器仅读取融合列表的前 $N$ 名候选并直接丢弃其余部分。当 $N$ 为 10 时, 排在第 8 位的相关块处于候选池内, 被重排器准确识别并提至第 1 名; 当 $N$ 缩减至 5 时, 第 8 位在进入重排前即被截断丢弃, 重排器精度再高也无从挽回。\n这印证了多阶段检索的核心边界: rerank 只能调整候选池内部的相对名次, 无法弥补上游的召回缺失; 凡未能进入截断窗口前 $N$ 名的文档块, 在后续环节中均无法恢复。要消除此类盲区, 必须扩充上游召回通道、放宽截断窗口 $N$ 或调整各路融合权重。点击该问题行查看底部「四条路线的名次」, 即可直观核对 BM25 ✘、dense 1、RRF 融合 8、RRF + rerank (窗口为 10 时) 1 的完整演化链。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="r in ROUTES" :key="r.id" type="button" :class="{ active: route === r.id }" @click="route = r.id">{{ r.label }}</button>
      </div>
      <LabSlider v-if="route === 'rerank'" v-model="top" label="rerank 范围 (只重排融合结果的前 N 名)" :min="1" :max="15" unit=" 名" />
      <p v-else class="tip">「rerank 范围」已隐藏: 这条路线不做 rerank。切到「RRF + rerank」再拖。</p>
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
      <div class="kv"><span>recall@1 (相关块排第 1 的题占比)</span><b :class="cmp(m[0], base[0])">{{ m[0].toFixed(2) }}</b></div>
      <div class="kv"><span>recall@3 (相关块进前 3 的题占比)</span><b :class="cmp(m[1], base[1])">{{ m[1].toFixed(2) }}</b></div>
      <div class="kv"><span>MRR (每题 1/名次 的平均, 没召回记 0)</span><b :class="cmp(m[2], base[2])">{{ m[2].toFixed(2) }}</b></div>
      <div class="kv"><span>未召回 (前 10 名里没有相关块)</span><b :class="misses ? 'bad' : 'good'">{{ misses }} 题</b></div>
      <div class="lab-note">
        <p>三项检索指标均为数值越大越优, 满分为 1.00。读数颜色以单路 BM25 为对比基准: 绿色表示优于 BM25, 红色表示劣于 BM25, 默认无着色表示与基线持平。</p>
        <p>各路线名次均由前端实时计算: RRF 融合基于两路真实排序结果合成, 重排阶段则使用离线 Python 脚本预计算的 rerank_score。当 rerank 范围设为 10 时, 四条路线的逐题排位与代码 demo [2] 完全一致。</p>
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
.tip { font-size: 12px; color: var(--text-muted); }
</style>
