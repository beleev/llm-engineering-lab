<!--
  中文检索: 字符 bigram 分词 + BM25 式 idf (对应 llm_agent/core/utils.py:tokenize 与 core/retrieval.py:TfidfIndex)。
  只讲一件事: 分词器只认 [a-z0-9]+ 时, 中文查询的得分恒为 0 —— 检索质量先死在分词上。
-->
<template>
  <LabFrame
    title="中文检索 — 分词器不认中文时, 检索为什么整个失效"
    sub="- 模拟什么: Agent 的 search_docs 知识库检索工具。外部语料库包含 8 篇单句文档 (6 篇英文与 2 篇中文), 每次按用户查询召回最相关的文档。实验重点展示分词器对跨语言检索的决定性影响: 若分词正则仅匹配 [a-z0-9]+, 中文字符会被全部丢弃, 导致中文查询零召回。底层采用与 m08 demo 完全一致的 TF-IDF 余弦检索流程, 共分三步:
  1. 将查询与文档切分为离散 token 序列 (分词)。
  2. 为每个 token 计算逆文档频率权重 idf (出现的文档越少, 权重越高)。
  3. 计算查询向量与各篇文档向量的余弦相似度并按得分降序排列。
- 图上是什么: 界面分为上下两个可视化区域:
  - 顶部 token 序列行: 展示当前查询切分出的各个 token, 方块下方标注该 token 的 idf 值; 标灰且注明「语料外」的 token 表示在全部 8 篇文档中均未出现, 不参与相似度计分。
  - 下方文档排名列表: 每行对应一篇文档, 依次展示排名、标题、相似度条形图与具体得分 (取值范围 0 至 1)。最高分且得分大于 0 的文档行带高亮底色; 悬停或选中某个查询 token 时, 包含该 token 的文档行会高亮黄色边框。
- 可以做什么: 按界面从上至下可执行五项操作:
  - 修改查询词: 在输入框中编辑文本, 或直接点击三个预设查询按钮。
  - 选择「英文按词 + 中文 bigram」: 启用兼容中英文的分词器, 英文按空白切词, 连续中文字符按相邻双字滑动切分为 bigram token。
  - 选择「只认 [a-z0-9]+」: 切换至仅识别英文字符与数字的反例分词器, 中文字符将被全部过滤。
  - 点击「idf: …」按钮: 在 BM25 式与平滑式两种 idf 公式之间切换, 公式中 N 为文档总数 8, df 为包含该 token 的文档篇数。
  - 悬停或点击任一查询 token: 在右侧面板查看其 df 统计, 并高亮命中的文档行。
- 应该看到: 对比两种分词器的检索表现与右侧读数:
  - 默认查询「显存碎片怎么解决」配合 bigram 分词: 查询被切分为 7 个 token, 「分页注意力」文档得分位居第一。
  - 切换至「只认 [a-z0-9]+」: 顶部显示「分词结果为空」, 全部文档得分归 0, 右侧 top-1 变为 no matches。
  - 右侧「top-1」: 当前排序第一的命中文档标题。
  - 「top-1 的余弦得分」: 仅用于同一查询内部候选文档的相对排序, 不能作为跨查询比较的绝对置信度。
  - 「得分大于 0 的文档」: 至少命中一个查询 token 的候选文档数量。
  - 「df」: 当前选中 token 在语料库中出现的文档篇数, df 越大则对应的 idf 权重越低。"
    module="llm_agent/m08"
    run="python -m llm_agent.m08_retrieval.demo"
    :challenge="{
      ask: '选用「只认 [a-z0-9]+」分词器检索「显存碎片怎么解决」时, top-1 文档的得分是多少? 切换为中文 bigram 分词器后, 为什么「显存」这一 token 的 idf 权重低于「碎片」?',
      answer: '仅识别英文字符时 top-1 得分为 0; 切换为 bigram 后, 「显存」因在更多文档中出现 (df=2 &gt; df=1) 而具有更低的 idf 权重。\n- 英文正则零召回原因: 正则表达式 [a-z0-9]+ 会过滤掉全部中文字符, 查询切分结果为空集, 对应的查询向量为零向量, 与任意文档向量的余弦相似度均为 0; 下游 Agent 收到 no matches 后将失去外部知识支撑。\n- 中文 bigram 切分机制: 无需维护外部中文词表即可将「显存碎片」切分为「显存」「存碎」「碎片」三个双字 token。其中跨词边界产生的「存碎」虽属切分噪声, 但通常属于「语料外」低频串而不会误命中无关文档, 对排序干扰极小。\n- idf 差异来源: 「显存」同时出现在两篇中文文档中 (分页注意力文档提及「显存碎片」, 低秩微调文档提及「显存占用」), 故 df=2; 而「碎片」仅出现在分页注意力文档中, 故 df=1。\n- 数值验算: 两种 idf 公式均随 df 增大而单调递减。以 BM25 式 $\\mathrm{idf} = \\ln\\big(1 + (N-\\mathrm{df}+0.5)/(\\mathrm{df}+0.5)\\big)$ 为例, 当 N=8 时, df=1 对应 idf 为 1.79, df=2 对应 idf 降至 1.28; 高频通用词 (如英文中的 the, df=6, idf 仅为 0.33) 缺乏区分文档的能力, 稀有词才主导相关性排序。',
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
        <p>本实验台计算的仍是 TF-IDF 向量余弦相似度, 仅在逆文档频率项上复用了 BM25 的平滑 idf 公式; 完整的 BM25 排序函数还包含此处未引入的两项机制: 控制词频边际收益递减的饱和参数 k1, 以及消除长短文档偏置的长度归一化参数 b。</p>
        <p>基于词频的稀疏检索仅能完成字面匹配, 无法处理同义替换、句式改写或跨语言查询。若需升级为稠密语义检索, 只需将底层向量化函数 embed() 替换为神经网络 embedding 模型; 上层工具签名 search_docs 与参数协议保持不变, Agent 主循环代码无需任何改动。</p>
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
