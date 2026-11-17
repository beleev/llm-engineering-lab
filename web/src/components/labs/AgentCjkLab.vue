<!--
  中文检索: 字符 bigram 分词 + BM25 式 idf (对应 llm_agent/core/utils.py:tokenize 与 core/retrieval.py:TfidfIndex)。
  只讲一件事: 分词器只认 [a-z0-9]+ 时, 中文查询的得分恒为 0 —— 检索质量先死在分词上。
-->
<template>
  <LabFrame
    title="中文检索 — 分词器不认中文时, 检索为什么整个失效"
    sub="- 模拟什么: agent 的 search_docs 工具: 知识库放在上下文外面, 每次按问题取最相关的几篇。语料是 8 篇一句话的文档 (6 篇英文 + 2 篇中文)。问题出在分词: 分词器只认 [a-z0-9]+ 的话, 中文被整段丢掉, 中文查询什么也检索不到。这里用的是 TF-IDF 余弦检索, 和 m08 demo 完全相同, 分三步:
        1. 先把文本切成一个个 token (分词)。
        2. 给每个 token 一个权重 idf (它在越少的文档里出现, 权重越高)。
        3. 算查询和每篇文档的余弦相似度来排序。
      - 图上是什么: 上下两部分:
        - 上面一排小方块: 查询被切出来的每个 token, 方块下面的小数字是这个 token 的 idf; 灰色、写着「语料外」的 token 在 8 篇文档里一次都没出现过, 算分时会被直接丢掉。
        - 下面每一行: 一篇文档, 排名、标题、得分条和得分 (0 到 1 的余弦相似度)。得分最高且大于 0 的那行有底色; 带黄框的行表示它含有你正在悬停或点选的那个 token。
      - 可以做什么: 按界面顺序:
        - 查询: 在输入框里改查询, 或点三个预设查询之一。
        - 「英文按词 + 中文 bigram」: 正确的分词器。英文按空格切词, 连续的中文每相邻两个字切成一个 token (bigram)。
        - 「只认 [a-z0-9]+」: 反例, 中文全部丢掉。
        - 在两种 idf 公式 (BM25 式和平滑式) 之间切换的是「idf: …」按钮。公式里 N 是文档总数 8, df 是含这个 token 的文档数。
        - 悬停或点击任一 token: 右侧会显示它的 df, 含它的文档会加黄框。
      - 应该看到: 两种分词器的对照和右侧读数:
        - 默认查询「显存碎片怎么解决」: 在 bigram 分词下切成 7 个 token, 「分页注意力」那篇得分最高。
        - 切到「只认 [a-z0-9]+」: token 行变成「分词结果为空」, 所有文档得分 0, 右侧 top-1 变成 no matches。
        - 右侧「top-1」: 排第一的文档。
        - 「top-1 的余弦得分」: 只用来在同一个查询下比大小, 不是跨查询可比的置信度。
        - 「得分大于 0 的文档」: 至少命中一个 token 的文档数。
        - 「df」: 当前盯住的 token 出现在几篇文档里, df 越大 idf 越小。"
    module="llm_agent/m08"
    run="python -m llm_agent.m08_retrieval.demo"
    :challenge="{
      ask: '用「只认英文」分词器查「显存碎片怎么解决」。top-1 得分是多少? 换成 bigram 之后, 为什么「显存」这个 token 的 idf 比「碎片」低?',
      answer: 'top-1 得分是 0: 只认英文的正则 [a-z0-9]+ 把整句中文丢光了, 查询一个 token 都没有, 查询向量是零向量, 和任何文档的余弦都是 0。调用它的 agent 拿到的是 no matches, 只能凭记忆编答案。\n换成 bigram 后不需要词典: 「显存碎片」切成 显存 / 存碎 / 碎片。「存碎」是跨词边界切出来的噪声 token, 但它几乎不会出现在别的文档里 (多半是「语料外」, 直接被丢掉), 所以影响很小。\n「显存」的 idf 比「碎片」低, 是因为它出现在更多文档里:\n- 「显存」: 两篇中文文档都有 (分页注意力那篇说「显存碎片」, 低秩微调那篇说「显存占用」), df=2。\n- 「碎片」: 只在分页注意力那篇出现, df=1。\n两种 idf 公式都是 df 越大值越小。以 BM25 式 $\\mathrm{idf} = \\ln\\big(1 + (N-\\mathrm{df}+0.5)/(\\mathrm{df}+0.5)\\big)$ 为例, N=8 时 df=1 得 1.79, df=2 得 1.28。几乎每篇都有的词 (比如英文的 the, df=6, idf 只有 0.33) 分不出哪篇相关, 权重趋近 0; 只在一两篇出现的词才有区分力。',
    }"
  >
    <template #controls>
      <div class="ctl">
        <label for="cjk-q">查询 query</label>
        <input id="cjk-q" v-model="query" type="text" class="mono q" spellcheck="false" />
        <span class="val mono">{{ qTokens.length }} token</span>
      </div>
      <div class="row">
        <button v-for="q in QUERIES" :key="q" type="button" :class="{ active: query === q }" @click="query = q">{{ q }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: bigram }" @click="bigram = true">英文按词 + 中文 bigram</button>
        <button type="button" :class="{ active: !bigram }" @click="bigram = false">只认 [a-z0-9]+ (反例)</button>
        <button type="button" :class="{ active: bm25 }" @click="bm25 = !bm25">idf: {{ bm25 ? 'BM25 式 ln(1+(N−df+.5)/(df+.5))' : '平滑式 ln(N/(1+df))+1' }}</button>
      </div>
    </template>

    <div class="tokens">
      <button
        v-for="(w, i) in qTokens" :key="i" type="button" class="tok mono" :class="{ oov: !(w in index.idf), on: focus === w }"
        @mouseenter="hover = w" @mouseleave="hover = ''" @focus="hover = w" @blur="hover = ''" @click="pin = w"
      >{{ w }}<em>{{ w in index.idf ? index.idf[w].toFixed(2) : '语料外' }}</em></button>
      <span v-if="!qTokens.length" class="empty">分词结果为空: 查询向量是零向量, 所有文档得分都是 0</span>
    </div>
    <div v-for="(d, i) in ranked" :key="d.title" class="doc" :class="{ top: i === 0 && d.score > 0, hit: focus && d.tf[focus] }">
      <span class="mono rank">#{{ i + 1 }}</span>
      <span class="mono title">{{ d.title }}</span>
      <span class="track"><span class="fill" :style="{ width: d.score * 100 + '%' }" /></span>
      <span class="mono score">{{ d.score.toFixed(2) }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>top-1 (排第一的文档)</span><b class="small" :class="ranked[0].score > 0 ? 'good' : 'bad'">{{ ranked[0].score > 0 ? ranked[0].title : 'no matches' }}</b></div>
      <div class="kv"><span>top-1 的余弦得分</span><b :class="ranked[0].score > 0 ? '' : 'bad'">{{ ranked[0].score.toFixed(2) }}</b></div>
      <div class="kv"><span>得分大于 0 的文档</span><b>{{ ranked.filter((d) => d.score > 0).length }} / {{ ranked.length }}</b></div>
      <div class="kv"><span>「{{ focus || '—' }}」的 df (出现在几篇里)</span><b>{{ focus ? (index.df[focus] || 0) : '—' }}</b></div>
      <div class="lab-note">
        <p>这里算的仍是 TF-IDF 余弦相似度, 只是 idf 这一项借了 BM25 的公式。完整的 BM25 还有两个这里没有的部分: 词频饱和参数 k1 (同一个词重复很多次, 分数不再线性增加) 和文档长度归一化 b。</p>
        <p>这种检索只做字面匹配: 同义词、改写、跨语言都匹配不上。要升级成语义检索, 把 embed() 换成神经网络 embedding 就行; 工具名 search_docs 和参数不变, agent loop 一行不用改。</p>
      </div>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

// 与 llm_agent/m08_retrieval/demo.py 的 DOCS 相同
const DOCS = {
  paged_attention: 'paged attention fixes memory fragmentation in the kv cache with block tables, like virtual memory pages for the model.',
  kv_cache: 'how does decode stay fast: the kv cache is the model memory of past keys and values.',
  faq: 'how does the model work, how does the team manage the model, and how does the budget get set.',
  lora: 'how does the team manage finetuning cost: lora trains low rank adapters for the model.',
  dpo: 'how does the team manage alignment: dpo tunes the model on preference pairs, no reward model.',
  sampling: 'how does the model pick the next token: temperature and top k control randomness.',
  分页注意力: '分页注意力用块表管理 KV 缓存, 解决显存碎片问题, 思路类似操作系统的虚拟内存分页。',
  低秩微调: 'LoRA 只训练低秩适配器, 可训练参数不到百分之一, 显存占用大幅下降。',
}
const QUERIES = ['显存碎片怎么解决', 'how does the model manage memory fragmentation', 'LoRA 显存占用']
const query = ref(QUERIES[0]), bigram = ref(true), bm25 = ref(true), hover = ref(''), pin = ref('')

// ★ utils.py:tokenize —— 英文按词; 中文连续段切成相邻两字的 bigram (单字段保留该字)
const tokenize = (text) => {
  const out = []
  for (const run of text.toLowerCase().match(/[a-z0-9]+|[一-鿿]+/g) || []) {
    if (/[a-z0-9]/.test(run[0])) out.push(run)
    else if (bigram.value) for (let i = 0; i < Math.max(1, run.length - 1); i++) out.push(run.slice(i, i + 2))
  }
  return out
}
const count = (tokens) => tokens.reduce((tf, w) => ((tf[w] = (tf[w] || 0) + 1), tf), {})

const index = computed(() => {
  const N = Object.keys(DOCS).length, df = {}
  const tfs = Object.entries(DOCS).map(([title, body]) => ({ title, tf: count(tokenize(`${title} ${body}`)) }))
  tfs.forEach(({ tf }) => Object.keys(tf).forEach((w) => (df[w] = (df[w] || 0) + 1)))
  const idf = Object.fromEntries(Object.entries(df).map(([w, c]) => [w, bm25.value ? Math.log(1 + (N - c + 0.5) / (c + 0.5)) : Math.log(N / (1 + c)) + 1]))
  return { tfs, df, idf }
})
// 归一化的稀疏向量; 权重 (1+ln tf)·idf: 同一个词重复 3 次不该有 3 倍话语权; 语料外的词直接丢
const vec = (tf, idf) => {
  const v = Object.fromEntries(Object.entries(tf).filter(([w]) => w in idf).map(([w, c]) => [w, (1 + Math.log(c)) * idf[w]]))
  const norm = Math.sqrt(Object.values(v).reduce((s, x) => s + x * x, 0)) || 1
  return Object.fromEntries(Object.entries(v).map(([w, x]) => [w, x / norm]))
}
const qTokens = computed(() => tokenize(query.value))
// 读数盯住的 token: 悬停的优先, 其次是点过的, 都没有就取查询的第一个。不悬停时 df 也有数
const focus = computed(() => hover.value || (qTokens.value.includes(pin.value) ? pin.value : qTokens.value[0] || ''))
const ranked = computed(() => {
  const { tfs, idf } = index.value
  const q = vec(count(qTokens.value), idf)
  return tfs.map(({ title, tf }) => {
    const d = vec(tf, idf)
    return { title, tf, score: Object.keys(q).reduce((s, w) => s + q[w] * (d[w] || 0), 0) }
  }).sort((a, b) => b.score - a.score)
})
</script>

<style scoped>
.q { min-height: 32px; min-width: 0; background: var(--bg-elev); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0 8px; font-size: 12px; }
.tokens { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; min-height: 44px; }
.tok { display: flex; flex-direction: column; align-items: center; min-height: 0; padding: 3px 8px; font-size: 12px; border: 1px solid var(--accent); border-radius: 4px; background: var(--accent-soft); color: var(--text); }
.tok em { font-style: normal; font-size: 9px; color: var(--text-dim); }
.tok.oov { border-color: var(--border); background: none; color: var(--text-dim); }
.tok.on { outline: 2px solid var(--warn); }
.empty { font-size: 12px; color: var(--danger); align-self: center; }
.doc { display: grid; grid-template-columns: 26px 120px 1fr 40px; gap: 8px; align-items: center; padding: 4px 6px; border-radius: 4px; font-size: 12px; color: var(--text-muted); }
.doc.top { color: var(--text); background: var(--accent-soft); }
.doc.hit { outline: 1px solid var(--warn); }
.rank { color: var(--text-dim); font-size: 11px; } .title { font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.track { height: 8px; border-radius: 4px; background: var(--border); overflow: hidden; } .fill { display: block; height: 100%; background: var(--accent); }
.score { text-align: right; }
.small { font-size: 12px !important; text-align: right; }
</style>
