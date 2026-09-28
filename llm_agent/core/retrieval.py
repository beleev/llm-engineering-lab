"""TF-IDF 向量检索: 稀疏向量 + 余弦相似度, 支持中文。

没有它: 关键词计数把 "the / model" 和 "fragmentation" 等权对待, 凑满常见词的文档能挤掉正确答案;
只认 [a-z0-9]+ 的分词器会让中文查询得分恒为 0。
关键设计:
  - idf 用 BM25 同款 ln(1 + (N-df+0.5)/(df+0.5)): 几乎每篇都有的词权重趋近 0, 罕见词主导排序。
  - 分词: 英文按词, 中文按字符 bigram (utils.tokenize)。
  - 换成神经 embedding (稠密语义检索) 要动三处: embed()、__init__ 里建文档向量的那一行、
    search() 的点积 (稀疏 dict → 稠密向量)。工具接口 search_docs(query) 不变。
VectorSearchTool 标了 untrusted_output: 语料是别人写的, 检索结果和抓回来的网页一样包装、置污点。
对应: RAG 的检索层; Claude Code 的 grep / glob 走的是同一思路 (按需取回, 而非预先塞满上下文)。
差异: grep / glob 按正则和文件名做精确匹配, 不打分。这里对内存里的语料算相似度, 按分数排序。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool, _obj
from llm_agent.core.utils import tokenize


class TfidfIndex:
    """内存里的 TF-IDF 索引。docs 是 {标题: 正文}, 构造时一次建好, 之后不能增删文档。

    向量是稀疏的 dict {词: 权重}, 已归一化到长度 1。
    """

    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs
        n = len(docs)  # 文档总数 N
        df: Counter = Counter()  # df[词] = 有几篇文档出现过它 (document frequency)
        tfs = {}  # tfs[标题] = 这篇文档里每个词出现几次 (term frequency)
        for title, body in docs.items():
            tokens = tokenize(f"{title} {body}")
            tfs[title] = Counter(tokens)
            df.update(set(tokens))  # 先去重: 一篇文档里出现多少次都只算 1
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self._doc_vec = {t: self._vectorize(tf) for t, tf in tfs.items()}

    def _vectorize(self, tf: Counter) -> Dict[str, float]:
        # 1+ln(tf): 同一个词重复 3 次不该有 3 倍话语权; 语料外的词没有方向, 直接丢
        vec = {w: (1 + math.log(c)) * self.idf[w] for w, c in tf.items() if w in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0  # 空向量的模是 0, 换成 1 防除零
        return {w: v / norm for w, v in vec.items()}

    def embed(self, text: str) -> Dict[str, float]:
        """查询文本 → 稀疏向量。用的是建索引时算好的 idf; 语料里没出现过的词被丢掉。"""
        return self._vectorize(Counter(tokenize(text)))

    def search(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        """返回得分最高的 k 个 (余弦相似度, 标题)。逐篇算, 文档多了要换倒排索引或向量库。"""
        q = self.embed(query)
        # 只遍历查询里的词: 查询没有的词对点积的贡献是 0
        scored = [(sum(q[w] * vec.get(w, 0.0) for w in q), title) for title, vec in self._doc_vec.items()]
        scored.sort(reverse=True)  # 两边都已归一化, 稀疏点积 = 余弦
        return scored[:k]


class VectorSearchTool(Tool):
    """与关键词版 SearchDocsTool 同名同 schema: 升级检索质量不用动 agent loop 和模型侧任何东西。"""

    name = "search_docs"
    description = "Search docs by TF-IDF cosine similarity (top-k with scores)."
    parameters = _obj(["query"], query={"type": "string"})
    untrusted_output = True  # 文档正文可能夹带注入; 配了 guardrails 时结果被包装, 并置污点

    def __init__(self, index: TfidfIndex, k: int = 3) -> None:
        self.index, self.k = index, k

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        hits = [(s, t) for s, t in self.index.search(args["query"], self.k) if s > 0]
        if not hits:
            return ToolResult(self.name, "no matches")
        return ToolResult(self.name, "\n".join(f"[{s:.2f}] {t}: {self.index.docs[t]}" for s, t in hits))
