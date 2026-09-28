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

Chunks = Dict[str, str]
Ranker = Callable[[str], List[str]]


# ------------------------------------------------------------------ 切块
def fixed_chunks(size: int = 30, overlap: int = 10) -> Chunks:
    """每 size 个词切一刀, 相邻块重叠 overlap 个词。不看结构: 标题和它的段落可能被切开。"""
    chunks = {}
    for doc, text in DOCS.items():
        words = text.split()
        for i, start in enumerate(range(0, max(1, len(words) - overlap), size - overlap)):
            chunks[f"{doc}#{i}"] = " ".join(words[start : start + size])
    return chunks


def structural_chunks() -> Chunks:
    """按 ## 小节切, 每块前面带 "文档 > 小节" 路径: 块被单独取出来时仍知道自己在讲什么。"""
    chunks = {}
    for doc, text in DOCS.items():
        title = text.splitlines()[0].lstrip("# ")
        for i, section in enumerate(text.split("\n## ")[1:]):
            heading, _, body = section.partition("\n")
            chunks[f"{doc}#{i}"] = f"{title} > {heading}\n{' '.join(body.split())}"
    return chunks


# ------------------------------------------------------------------ 稀疏: BM25
class BM25:
    def __init__(self, chunks: Chunks, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = {cid: Counter(tokenize(text)) for cid, text in chunks.items()}
        self.lens = {cid: sum(tf.values()) for cid, tf in self.tfs.items()}
        self.avgdl = sum(self.lens.values()) / len(self.lens)
        df = Counter(w for tf in self.tfs.values() for w in tf)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query: str, cid: str) -> float:
        tf, norm = self.tfs[cid], self.k1 * (1 - self.b + self.b * self.lens[cid] / self.avgdl)
        # tf 饱和: 第 1 次出现最值钱, 之后边际递减; 长块的 tf 按长度打折
        return sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in set(tokenize(query)) if w in tf)

    def rank(self, query: str) -> List[str]:
        scored = [(self.score(query, cid), cid) for cid in self.tfs]
        return [cid for s, cid in sorted(scored, key=lambda x: -x[0]) if s > 0]


# ------------------------------------------------------------------ 稠密: 字符 n-gram 哈希 + 随机投影
BUCKETS, DIM = 4096, 128


def char_ngrams(text: str) -> List[str]:
    grams = []
    for word in tokenize(text):  # 中文 token 本来就是 bigram, 直接当一个 gram
        w = f"<{word}>"
        grams.extend(w[i : i + 3] for i in range(len(w) - 2)) if word.isascii() else grams.append(word)
    return grams


class DenseIndex:
    """ponytail: 这是"拼写相似度"的稠密化, 不是语义 embedding; 换神经 embedding 只需替换 embed()。"""

    def __init__(self, chunks: Chunks) -> None:
        rng = random.Random(0)  # 固定种子: 投影矩阵每次一样, 结果可复现
        self.proj = [[rng.choice((-1.0, 1.0)) for _ in range(DIM)] for _ in range(BUCKETS)]
        grams = {cid: Counter(self._buckets(text)) for cid, text in chunks.items()}
        df = Counter(b for g in grams.values() for b in g)
        n = len(chunks)
        self.idf = {b: math.log(1 + n / c) for b, c in df.items()}  # "<th"、"the" 这种满地都是的 gram 降权
        self.vecs = {cid: self._project(g) for cid, g in grams.items()}

    @staticmethod
    def _buckets(text: str) -> List[int]:
        return [zlib.crc32(g.encode()) % BUCKETS for g in char_ngrams(text)]  # crc32 而非 hash(): 后者每个进程随机加盐

    def _project(self, grams: Counter) -> List[float]:
        vec = [0.0] * DIM
        for bucket, count in grams.items():
            weight = (1 + math.log(count)) * self.idf.get(bucket, 0.0)  # 语料外的 gram 没有方向
            row = self.proj[bucket]
            for i in range(DIM):
                vec[i] += weight * row[i]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, text: str) -> List[float]:
        return self._project(Counter(self._buckets(text)))

    def rank(self, query: str) -> List[str]:
        q = self.embed(query)
        scored = [(sum(a * b for a, b in zip(q, vec)), cid) for cid, vec in self.vecs.items()]
        return [cid for _, cid in sorted(scored, key=lambda x: -x[0])]


# ------------------------------------------------------------------ 融合与重排
def rrf(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)。只用名次, 不用分数 —— BM25 没有上界, 余弦在 [-1, 1], 没法直接相加。"""
    scores: Counter = Counter()
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] += 1 / (k + rank)
    return [cid for cid, _ in scores.most_common()]


STOP = {"what", "is", "the", "a", "an", "on", "of", "do", "does", "i", "my", "can", "how", "when", "why", "if", "get", "are", "each", "to"}


def _match(q: str, w: str) -> bool:
    return q == w or (min(len(q), len(w)) >= 5 and q[:5] == w[:5])  # 粗糙词干: 前 5 个字母相同算同词


def rerank_score(query: str, text: str) -> float:
    """逐对精读 query 与块: 查询实词覆盖率 + 相邻实词在块里按原顺序紧挨着出现 (短语匹配)。O(|q|·|块|), 只给 top-k 用。"""
    q = [w for w in tokenize(query) if w not in STOP]
    words = tokenize(text)
    if not q:
        return 0.0
    hits = [[i for i, w in enumerate(words) if _match(t, w)] for t in q]
    coverage = sum(bool(h) for h in hits) / len(q)
    pairs = list(zip(hits, hits[1:]))
    phrase = sum(any(0 < j - i <= 2 for i in a for j in b) for a, b in pairs) / len(pairs) if pairs else 0.0
    return coverage + 0.5 * phrase


def reranked(first_stage: Ranker, chunks: Chunks, top: int = 10) -> Ranker:
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
        relevant = {cid for cid, text in chunks.items() if " ".join(answer.split()) in " ".join(text.split())}
        ranks.append(next((i for i, cid in enumerate(rank(question), 1) if cid in relevant), 0))
    return ranks


def metrics(ranks: List[int]) -> Tuple[float, float, float]:
    """recall@k: 相关块进了前 k 的题目比例; MRR: 名次倒数的平均 (没召回记 0)。"""
    n = len(ranks)
    return sum(r == 1 for r in ranks) / n, sum(0 < r <= 3 for r in ranks) / n, sum(1 / r for r in ranks if r) / n


def routes(chunks: Chunks) -> Dict[str, Ranker]:
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
    untrusted_output = True  # 语料是 prompt injection 的常见入口 (m12), m08 没标这一点

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
    f, s = scores.values()
    assert s[2] > f[2] + 0.1 and s[1] >= f[1]  # 标题跟着段落走 + 不会把一句话切成两半

    print(f"\n[2] 五条检索路线 × {len(QA)} 道标注问答 (结构切块), 名次 0 = 没召回")
    ranks = {name: evaluate(rank, structural) for name, rank in routes(structural).items()}
    for name, r in ranks.items():
        r1, r3, mrr = metrics(r)
        print(f"    {name:<14} recall@1={r1:.2f}  recall@3={r3:.2f}  MRR={mrr:.2f}  名次={r}")
    best = metrics(ranks["hybrid+rerank"])
    for single in ("tfidf (m08)", "bm25", "dense"):
        m = metrics(ranks[single])
        assert best[0] > m[0] and best[2] > m[2] and best[1] >= m[1], single  # 严格胜过每条单路; recall@3 见 [3] 的同义题
    assert best[2] > metrics(ranks["hybrid rrf"])[2]  # rerank 的贡献: 把已召回的相关块往前挪

    print("\n[3] 两路各有盲区")
    for i in (5, 3, 17):
        row = "  ".join(f"{n}={ranks[n][i]}" for n in ("bm25", "dense", "hybrid rrf", "hybrid+rerank"))
        print(f"    {QA[i][0]:<42} {row}")
    assert ranks["bm25"][5] == 0 and ranks["dense"][5] == 1  # resume ≠ resumes / resumable: 词面 0 命中, 字符 n-gram 接得住
    assert ranks["hybrid rrf"][5] > 3 and ranks["hybrid+rerank"][5] == 1  # 只有一路召回 → RRF 名次靠后 → rerank 捞回来
    assert ranks["bm25"][3] == 1 < ranks["dense"][3]  # 错误码 E413: 罕见精确词在 BM25 里 idf 极高, 在 n-gram 向量里被稀释
    syn = len(QA) - 1
    kv("同义改写", f"{QA[syn][0]!r} → 各路名次 {[r[syn] for r in ranks.values()]}")
    assert all(r[syn] != 1 for r in ranks.values())  # 没有一路能把它排第一: 字符级"稠密"不懂语义

    print("\n[4] 接进 agent loop: 工具名仍是 search_docs")
    agent = Agent(RuleBasedLLM(), ToolRegistry([RagSearchTool(structural)]), PermissionGate("auto"), max_turns=3, name="m16")
    final = agent.run("检索: Can an interrupted upload resume?")
    assert final.splitlines()[1].startswith("search_docs: [uploads#1]") and "last confirmed part" in final

    print("\n  OK: 切块决定检索的上限, 混合召回补盲区, rerank 定名次, 评测集告诉你到底有没有变好。")


if __name__ == "__main__":
    main()
