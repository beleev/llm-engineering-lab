"""M16 — RAG: 切块 → 稀疏 (BM25) + 稠密两路召回 → RRF 融合 → rerank, 用带标注的问答集量 recall@k / MRR。

没有它: m08 一篇文档就是一句话, 整篇返回; 真实文档要切块。只用词面匹配, "throttled" 找不到 "throttling";
只用向量, 错误码 E413 这种罕见精确词又容易被稀释。没有评测, 所有"升级"都只是感觉。
关键设计:
  - 切块: 定长 + 重叠 vs 按标题 / 段落。后者每块带上 "文档 > 小节" 标题路径, 问题里的词常常只出现在标题里。
  - 稠密向量 (纯 stdlib): 字符 3-gram → 哈希到 4096 桶 → 固定种子的 ±1 随机投影到 128 维。
    它只懂"拼写像不像", 不懂"意思像不像" —— 同义改写照样找不到 (demo [3] 当场演示)。
  - BM25: 在 m08 的 idf 上加 tf 饱和 (k1) 和长度归一化 (b)。
  - RRF: 只看名次不看分数, 两路分数尺度不同也能融合。
  - rerank: 只对融合后的 top-10 逐对打分 (覆盖率 + 词序短语匹配), 贵但准, 所以只用在少数候选上。
  - 接进 agent: 工具名仍是 search_docs, loop 零改动; 检索回来的语料标记为不可信 (m12)。
对应: 生产 RAG 的 chunking / hybrid search (BM25 + embedding) / reciprocal rank fusion / cross-encoder rerank。
差异: 这里的"稠密向量"是字符 n-gram 的随机投影, 生产系统用神经 embedding;
  这里的 rerank 是词面规则 (覆盖率 + 短语匹配), 生产系统用 cross-encoder 模型。
  评测集只有 20 道题、5 篇文档, 数字只能说明方向。
"""

from __future__ import annotations

import math
import random
import re
import zlib
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from llm_agent.core import Agent, PermissionGate, RuleBasedLLM, TfidfIndex, Tool, ToolRegistry
from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import _obj
from llm_agent.core.utils import banner, kv, tokenize

DOCS = {
    "uploads": """# Uploads

## File size limits
Free accounts can upload files up to 2 GB each. Pro accounts raise the per-file limit to 50 GB.
Files larger than the limit are rejected with error E413.

## Resumable uploads
Large uploads are split into 8 MB parts. If the connection drops, the client resumes from the last confirmed part instead of starting over.

## Throttling
When one account sends more than 300 requests per minute, the API starts throttling and returns error E429 with a Retry-After header.""",
    "sharing": """# Sharing

## Share links
Anyone with a share link can view the file. The owner can protect a link with a password.

## Link expiration
Share links expire after 30 days by default. The owner can set a custom expiry between 1 hour and 1 year.

## Revoking access
Deleting a share link revokes access immediately, and cached previews are purged within 10 minutes.""",
    "security": """# Security

## Encryption at rest
All stored files are encrypted with AES-256. Encryption keys are rotated every 90 days by the key management service.

## Two-factor authentication
Accounts can enable two-factor authentication with an authenticator app. Recovery codes are shown once at setup.

## Session timeout
Idle web sessions sign out automatically after 12 hours.""",
    "billing": """# Billing

## Plans
The Pro plan costs 8 dollars per month or 80 dollars per year.

## Refunds
Annual subscriptions can be refunded within 14 days of purchase. Monthly subscriptions are not refundable.

## Failed payments
If a card payment fails, the account enters a 7 day grace period before it is downgraded to Free.""",
    "sync": """# Desktop sync

## Selective sync
Choose which folders the desktop client keeps on disk; unselected folders stay online only.

## Conflicts
When two devices edit the same file offline, the later upload is saved as a conflicted copy next to the original.

## Bandwidth
The desktop client limits upload bandwidth to 5 MB/s by default; this can be changed in settings.""",
}

# (问题, 标准答案里的一段原文): 块里完整包含这段原文才算"相关" —— 与切块方式无关, 两种切法可以同尺比较
QA = [
    ("What is the maximum file size on a free account?", "up to 2 GB each"),
    ("Why do I get error E429?", "300 requests per minute"),
    ("How are uploads throttled?", "300 requests per minute"),
    ("What does error E413 mean?", "rejected with error E413"),
    ("How big is each part of a resumable upload?", "split into 8 MB parts"),
    ("Can an interrupted upload resume?", "resumes from the last confirmed part"),
    ("When do shared links expire?", "expire after 30 days"),
    ("How do I revoke a link?", "revokes access immediately"),
    ("Can I password protect a share link?", "protect a link with a password"),
    ("How often are encryption keys rotated?", "rotated every 90 days"),
    ("Is my data encrypted?", "encrypted with AES-256"),
    ("How do I set up two-factor authentication?", "with an authenticator app"),
    ("When does an idle session time out?", "after 12 hours"),
    ("Can I get a refund on a monthly subscription?", "Monthly subscriptions are not refundable"),
    ("What happens if my payment fails?", "7 day grace period"),
    ("How much does the Pro plan cost per year?", "80 dollars per year"),
    ("What if two devices edit the same file?", "saved as a conflicted copy"),
    ("How fast can the desktop client upload?", "5 MB/s"),
    ("Can I keep some folders online only?", "stay online only"),
    ("How do I get my money back?", "can be refunded within 14 days"),  # 同义改写: 全文没有 money / back
]

Chunks = Dict[str, str]  # 块 id ("文档名#序号") → 块的文本
Ranker = Callable[[str], List[str]]  # 查询 → 按相关度从高到低排好的块 id


# ------------------------------------------------------------------ 切块
def fixed_chunks(size: int = 30, overlap: int = 10) -> Chunks:
    """每 size 个词切一刀, 相邻块重叠 overlap 个词。不看结构: 标题和它的段落可能被切开。"""
    chunks = {}
    for doc, text in DOCS.items():
        words = text.split()
        # 步长 = size - overlap (默认 20 个词)。起点只走到 len - overlap: 再往后的块全落在上一块的重叠区里
        for i, start in enumerate(range(0, max(1, len(words) - overlap), size - overlap)):
            chunks[f"{doc}#{i}"] = " ".join(words[start : start + size])
    return chunks


def structural_chunks() -> Chunks:
    """按 ## 小节切, 每块前面带 "文档 > 小节" 路径: 块被单独取出来时仍知道自己在讲什么。"""
    chunks = {}
    for doc, text in DOCS.items():
        title = text.splitlines()[0].lstrip("# ")  # 第一行 "# Uploads" → "Uploads"
        # 按 "\n## " 切开后, 第 0 段是文档标题行, 从第 1 段起才是小节
        for i, section in enumerate(text.split("\n## ")[1:]):
            heading, _, body = section.partition("\n")
            chunks[f"{doc}#{i}"] = f"{title} > {heading}\n{' '.join(body.split())}"
    return chunks


# ------------------------------------------------------------------ 稀疏: BM25
class BM25:
    """BM25 打分: score = Σ_词 idf × tf × (k1 + 1) / (tf + k1 × (1 − b + b × 块长 / 平均块长))。

    k1  tf 饱和得多快。越小, 同一个词重复出现带来的加分越早封顶。1.5 是常用的默认值
    b   长度归一化的力度。0 = 不管块长, 1 = 完全按长度打折。0.75 是常用的默认值
    """

    def __init__(self, chunks: Chunks, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = {cid: Counter(tokenize(text)) for cid, text in chunks.items()}  # 每块里每个词出现几次
        self.lens = {cid: sum(tf.values()) for cid, tf in self.tfs.items()}  # 每块有多少个词
        self.avgdl = sum(self.lens.values()) / len(self.lens)  # 平均块长 (average document length)
        df = Counter(w for tf in self.tfs.values() for w in tf)  # 每个词出现在几个块里
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query: str, cid: str) -> float:
        """查询和一个块的 BM25 分。只累加查询里出现、块里也出现的词。"""
        # norm 是公式分母里除 tf 以外的那一项: 块比平均长, norm 就大, 同样的 tf 得分就低
        tf, norm = self.tfs[cid], self.k1 * (1 - self.b + self.b * self.lens[cid] / self.avgdl)
        # tf 饱和: 第 1 次出现最值钱, 之后边际递减; 长块的 tf 按长度打折
        return sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in set(tokenize(query)) if w in tf)

    def rank(self, query: str) -> List[str]:
        """全部块按分数从高到低排。0 分的块 (一个词都没命中) 不返回, 所以词面路线可能召回不到。"""
        scored = [(self.score(query, cid), cid) for cid in self.tfs]
        return [cid for s, cid in sorted(scored, key=lambda x: -x[0]) if s > 0]


# ------------------------------------------------------------------ 稠密: 字符 n-gram 哈希 + 随机投影
BUCKETS, DIM = 4096, 128  # n-gram 先哈希进 4096 个桶, 再投影成 128 维的向量


def char_ngrams(text: str) -> List[str]:
    """把文本拆成字符 3-gram: "resume" → "<re" "res" "esu" "sum" "ume" "me>"。

    resume 和 resumes 的词面不同, 3-gram 却大半相同, 所以向量相近。
    """
    grams = []
    for word in tokenize(text):  # 中文 token 本来就是 bigram, 直接当一个 gram
        w = f"<{word}>"  # 两头加尖括号标出词的边界: 词首的 "re" 和词中的 "re" 算不同的 gram
        grams.extend(w[i : i + 3] for i in range(len(w) - 2)) if word.isascii() else grams.append(word)
    return grams


class DenseIndex:
    """简化: 这是"拼写相似度"的稠密化, 不是语义 embedding。

    换神经 embedding 要换两处: embed(), 以及 __init__ 里建 self.vecs 的那一行 (它直接调 _project, 不经过 embed)。
    rank() 的点积不用动。
    """

    def __init__(self, chunks: Chunks) -> None:
        rng = random.Random(0)  # 固定种子: 投影矩阵每次一样, 结果可复现
        # 投影矩阵 [BUCKETS, DIM] = [4096, 128], 元素是 ±1。每个桶对应一个固定的 128 维方向
        self.proj = [[rng.choice((-1.0, 1.0)) for _ in range(DIM)] for _ in range(BUCKETS)]
        grams = {cid: Counter(self._buckets(text)) for cid, text in chunks.items()}  # 每块里每个桶出现几次
        df = Counter(b for g in grams.values() for b in g)  # 每个桶出现在几个块里
        n = len(chunks)
        self.idf = {b: math.log(1 + n / c) for b, c in df.items()}  # "<th"、"the" 这种满地都是的 gram 降权
        self.vecs = {cid: self._project(g) for cid, g in grams.items()}  # 每块一个 [DIM] 向量, 已归一化

    @staticmethod
    def _buckets(text: str) -> List[int]:
        """文本 → 每个 3-gram 所在的桶号 (0 … 4095)。不同的 gram 可能撞进同一个桶。"""
        return [zlib.crc32(g.encode()) % BUCKETS for g in char_ngrams(text)]  # crc32 而非 hash(): 后者每个进程随机加盐

    def _project(self, grams: Counter) -> List[float]:
        """桶计数 → 归一化的 [DIM] 向量: 把各桶的方向按权重加起来。"""
        vec = [0.0] * DIM  # [DIM]
        for bucket, count in grams.items():
            weight = (1 + math.log(count)) * self.idf.get(bucket, 0.0)  # 语料外的 gram 没有方向
            row = self.proj[bucket]
            for i in range(DIM):
                vec[i] += weight * row[i]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0  # 全零向量的模是 0, 换成 1 防除零
        return [v / norm for v in vec]  # [DIM], 长度为 1

    def embed(self, text: str) -> List[float]:
        """查询文本 → [DIM] 向量。"""
        return self._project(Counter(self._buckets(text)))

    def rank(self, query: str) -> List[str]:
        """全部块按余弦相似度从高到低排。和 BM25 不同, 这里每个块都会返回, 不存在召回不到。"""
        q = self.embed(query)
        # 两边都已归一化, 点积就是余弦
        scored = [(sum(a * b for a, b in zip(q, vec)), cid) for cid, vec in self.vecs.items()]
        return [cid for _, cid in sorted(scored, key=lambda x: -x[0])]


# ------------------------------------------------------------------ 融合与重排
def rrf(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)。只用名次, 不用分数 —— BM25 没有上界, 余弦在 [-1, 1], 没法直接相加。"""
    # k=60 是 RRF 常用的默认值。k 越大, 第 1 名和第 2 名的分差越小, 各路的头名越难一锤定音
    scores: Counter = Counter()
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):  # 名次从 1 起算
            scores[cid] += 1 / (k + rank)
    return [cid for cid, _ in scores.most_common()]


# 停用词: rerank 只看实词, 这些词不参与覆盖率和短语匹配
STOP = {"what", "is", "the", "a", "an", "on", "of", "do", "does", "i", "my", "can", "how", "when", "why", "if", "get", "are", "each", "to"}


def _match(q: str, w: str) -> bool:
    return q == w or (min(len(q), len(w)) >= 5 and q[:5] == w[:5])  # 粗糙词干: 前 5 个字母相同算同词


def rerank_score(query: str, text: str) -> float:
    """逐对精读 query 与块: 查询实词覆盖率 + 相邻实词在块里按原顺序紧挨着出现 (短语匹配)。O(|q|·|块|), 只给 top-k 用。"""
    q = [w for w in tokenize(query) if w not in STOP]
    words = tokenize(text)
    if not q:
        return 0.0
    hits = [[i for i, w in enumerate(words) if _match(t, w)] for t in q]  # 每个查询词在块里出现的位置
    coverage = sum(bool(h) for h in hits) / len(q)  # 查询实词里有几成在块里出现过
    pairs = list(zip(hits, hits[1:]))  # 查询里相邻的两个实词
    # 0 < j - i <= 2: 后一个词在前一个词之后, 中间最多隔 1 个词
    phrase = sum(any(0 < j - i <= 2 for i in a for j in b) for a, b in pairs) / len(pairs) if pairs else 0.0
    return coverage + 0.5 * phrase  # 短语匹配的权重取覆盖率的一半; 满分 1.5


def reranked(first_stage: Ranker, chunks: Chunks, top: int = 10) -> Ranker:
    """给一阶段的排序器套上 rerank: 取它的前 top 名精排。排在 top 名之外的块会被丢掉。"""

    def rank(query: str) -> List[str]:
        head = first_stage(query)[:top]
        # 稳定排序: 精排分数打平时保留一阶段的名次
        return sorted(head, key=lambda cid: -rerank_score(query, chunks[cid]))

    return rank


# ------------------------------------------------------------------ 评测
def evaluate(rank: Ranker, chunks: Chunks) -> List[int]:
    """每道题第一个相关块的名次 (1 起; 没召回记 0)。"""
    ranks = []
    for question, answer in QA:
        # 两边都先把空白并成单个空格再比: 切块时换行变成了空格, 不这样做会对不上
        relevant = {cid for cid, text in chunks.items() if " ".join(answer.split()) in " ".join(text.split())}
        ranks.append(next((i for i, cid in enumerate(rank(question), 1) if cid in relevant), 0))
    return ranks


def metrics(ranks: List[int]) -> Tuple[float, float, float]:
    """recall@k: 相关块进了前 k 的题目比例; MRR: 名次倒数的平均 (没召回记 0)。"""
    n = len(ranks)
    # 依次是 recall@1, recall@3, MRR
    return sum(r == 1 for r in ranks) / n, sum(0 < r <= 3 for r in ranks) / n, sum(1 / r for r in ranks if r) / n


def routes(chunks: Chunks) -> Dict[str, Ranker]:
    """在同一批块上建好五条检索路线, 返回 {路线名: 排序器}。"""
    tfidf, bm25, dense = TfidfIndex(chunks), BM25(chunks), DenseIndex(chunks)
    hybrid = lambda q: rrf([bm25.rank(q), dense.rank(q)])  # noqa: E731
    return {
        "tfidf (m08)": lambda q: [t for s, t in tfidf.search(q, k=len(chunks)) if s > 0],
        "bm25": bm25.rank,
        "dense": dense.rank,
        "hybrid rrf": hybrid,
        "hybrid+rerank": reranked(hybrid, chunks),
    }


# ------------------------------------------------------------------ 接进 agent
class RagSearchTool(Tool):
    """与 m08 同名同 schema: 检索管线整个换掉, 模型侧看到的仍是 search_docs(query)。"""

    name = "search_docs"
    description = "Hybrid (BM25 + dense) search with rerank over chunked docs; returns top-k chunks with source."
    parameters = _obj(["query"], query={"type": "string"})
    # 语料是 prompt injection 的常见入口 (m12)。core/retrieval.py 的 VectorSearchTool 没有这个标记
    untrusted_output = True

    def __init__(self, chunks: Chunks, k: int = 2) -> None:
        self.chunks, self.k = chunks, k
        self.rank = routes(chunks)["hybrid+rerank"]

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        hits = self.rank(args["query"])[: self.k]
        return ToolResult(self.name, "\n".join(f"[{cid}] {self.chunks[cid]}" for cid in hits) or "no matches")


def main() -> None:
    banner("M16 - RAG: chunking / BM25 / dense / RRF / rerank")
    structural, fixed = structural_chunks(), fixed_chunks()

    print("\n[1] 切块: 定长+重叠 vs 按标题结构 (同一条 hybrid+rerank 管线, 同一套问答)")
    scores = {}
    for name, chunks in (("fixed 30w / overlap 10", fixed), ("structural (## 小节)", structural)):
        scores[name] = metrics(evaluate(routes(chunks)["hybrid+rerank"], chunks))
        kv(name, f"{len(chunks)} 块, recall@1={scores[name][0]:.2f}  recall@3={scores[name][1]:.2f}  MRR={scores[name][2]:.2f}")
    print("    定长块:", repr(fixed["uploads#1"][:92]))
    print("    结构块:", repr(structural["uploads#2"][:92]))
    f, s = scores.values()  # f = 定长切块的 (recall@1, recall@3, MRR), s = 结构切块的
    # 标题跟着段落走 + 不会把一句话切成两半
    assert s[2] > f[2] + 0.1, f"结构切块的 MRR 应比定长切块高出 0.1 以上, 实际: {s[2]:.2f} vs {f[2]:.2f}"
    assert s[1] >= f[1], f"结构切块的 recall@3 应不低于定长切块, 实际: {s[1]:.2f} vs {f[1]:.2f}"

    print(f"\n[2] 五条检索路线 × {len(QA)} 道标注问答 (结构切块), 名次 0 = 没召回")
    ranks = {name: evaluate(rank, structural) for name, rank in routes(structural).items()}
    for name, r in ranks.items():
        r1, r3, mrr = metrics(r)
        print(f"    {name:<14} recall@1={r1:.2f}  recall@3={r3:.2f}  MRR={mrr:.2f}  名次={r}")
    best = metrics(ranks["hybrid+rerank"])
    for single in ("tfidf (m08)", "bm25", "dense"):
        m = metrics(ranks[single])
        # 严格胜过每条单路; recall@3 见 [3] 的同义题
        assert best[0] > m[0], f"hybrid+rerank 的 recall@1 应高于 {single}, 实际: {best[0]:.2f} vs {m[0]:.2f}"
        assert best[2] > m[2], f"hybrid+rerank 的 MRR 应高于 {single}, 实际: {best[2]:.2f} vs {m[2]:.2f}"
        assert best[1] >= m[1], f"hybrid+rerank 的 recall@3 应不低于 {single}, 实际: {best[1]:.2f} vs {m[1]:.2f}"
    # rerank 的贡献: 把已召回的相关块往前挪
    assert best[2] > metrics(ranks["hybrid rrf"])[2], "加上 rerank 后 MRR 应高于只做 RRF 融合"

    print("\n[3] 两路各有盲区")
    for i in (5, 3, 17):  # QA 的下标: 5 = 断点续传那题, 3 = 错误码 E413, 17 = 上传速度
        row = "  ".join(f"{n}={ranks[n][i]}" for n in ("bm25", "dense", "hybrid rrf", "hybrid+rerank"))
        print(f"    {QA[i][0]:<42} {row}")
    # resume ≠ resumes / resumable: 词面 0 命中, 字符 n-gram 接得住
    assert ranks["bm25"][5] == 0, f"BM25 按词面匹配, 这题应召回不到 (名次 0), 实际: {ranks['bm25'][5]}"
    assert ranks["dense"][5] == 1, f"字符 n-gram 向量应把这题排第 1, 实际: {ranks['dense'][5]}"
    # 只有一路召回 → RRF 名次靠后 → rerank 捞回来
    assert ranks["hybrid rrf"][5] > 3, f"只有一路召回时 RRF 的名次应落在前 3 之外, 实际: {ranks['hybrid rrf'][5]}"
    assert ranks["hybrid+rerank"][5] == 1, f"rerank 应把它捞回第 1, 实际: {ranks['hybrid+rerank'][5]}"
    # 错误码 E413: 罕见精确词在 BM25 里 idf 极高, 在 n-gram 向量里被稀释
    assert ranks["bm25"][3] == 1 < ranks["dense"][3], (
        f"错误码这题 BM25 应排第 1, dense 应排得更靠后, 实际: {ranks['bm25'][3]} vs {ranks['dense'][3]}"
    )
    syn = len(QA) - 1  # 最后一题是同义改写
    kv("同义改写", f"{QA[syn][0]!r} → 各路名次 {[r[syn] for r in ranks.values()]}")
    # 没有一路能把它排第一: 字符级"稠密"不懂语义
    assert all(r[syn] != 1 for r in ranks.values()), "同义改写这题, 五条路线都不应排到第 1"

    print("\n[4] 接进 agent loop: 工具名仍是 search_docs")
    agent = Agent(RuleBasedLLM(), ToolRegistry([RagSearchTool(structural)]), PermissionGate("auto"), max_turns=3, name="m16")
    final = agent.run("检索: Can an interrupted upload resume?")
    assert final.splitlines()[1].startswith("search_docs: [uploads#1]"), (
        f"工具名应仍是 search_docs, 排第一的应是 uploads 的断点续传小节, 实际: {final}"
    )
    assert "last confirmed part" in final, "最终回答应带上答案所在的原文"

    print("\n  OK: 切块决定检索的上限, 混合召回补盲区, rerank 定名次, 评测集告诉你到底有没有变好。")


if __name__ == "__main__":
    main()
