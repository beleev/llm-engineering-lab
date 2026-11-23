"""
draftless.py — 不用草稿模型的投机解码: 从已有文本里查 n-gram 当草稿

是什么: m07 的投机解码需要一个 draft 模型, 它本身要显存、要算、要和 target 对齐。可很多负载的输出大段照抄输入
    (代码编辑、RAG 摘录、多轮对话复述), 或者和以前的输出高度相似 (Agent 反复发同一类工具调用)。
    这时"下一段大概率是什么"直接查文本就知道, 草稿的成本是一次 CPU 上的字符串匹配。
两种查法 (验证和接受规则完全复用 m07, target 一次 forward 验 K+1 个位置, greedy 下输出逐 token 不变):
    PromptLookupDrafter : 取已生成文本最后 n 个 token (n 从 max_n 降到 min_n), 在 prompt + 已生成里找最近一次出现,
                          把它后面的 K 个 token 当草稿 (prompt lookup decoding; vLLM 的 method="ngram" 同理)
    SuffixCacheDrafter  : 把以前请求的输出也存进 n-gram 索引, 沿"出现次数最多的下一个 token"往下走,
                          草稿长度上限随匹配长度 p 伸缩 (SuffixDecoding 的 MAX_SPEC = α·p)
关键: 草稿猜错只浪费 target 一次 forward 里多算的几个位置; decode 带宽受限时这几乎不花时间 (m07 / m19)。
    查不到匹配就不提草稿, 退化成普通 decode, 不会比逐 token 更慢。
简化: SuffixDecoding 用后缀树并按频率长出一棵草稿树; 这里用 n ≤ max_n 的 n-gram 计数表, 只出一条链。
简化: REST 的检索库 (后缀数组) 与 Lookahead decoding 的 Jacobi n-gram 池没有实现, 只在 README 里讲。
简化: OracleTarget 是合成的 target: 它的 greedy 输出是事先写好的参考答案, 只用来量接受率;
      "无损"另在真实 TinyLM 上验证。
读代码盯住: PromptLookupDrafter.propose 里"最近一次出现"的倒序搜索, 和 SuffixCacheDrafter 的 max_len = α·p。
对应真实系统: HF transformers 的 prompt_lookup_num_tokens; vLLM speculative_config 的 method="ngram"
    (prompt_lookup_max / prompt_lookup_min / num_speculative_tokens)。
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Sequence

import numpy as np


class PromptLookupDrafter:
    """在 out (prompt + 已生成) 里找末尾 n-gram 的最近一次出现, 把它后面的 token 抄过来当草稿。"""

    def __init__(self, max_n: int = 3, min_n: int = 1):
        self.max_n, self.min_n = max_n, min_n
        self.calls = 0                     # draft 模型 forward 次数: 恒为 0, 与 m07 的 ModelDrafter 对照
        self.proposed = 0                  # 一共提了多少个草稿 token

    def propose(self, out: List[int], K: int, temperature: float = 0.0, rng=None, hidden=None):
        """接口与 m07 的 ModelDrafter 相同 → (草稿 token, None)。查不到时返回空草稿。"""
        for n in range(min(self.max_n, len(out) - 1), self.min_n - 1, -1):
            tail = out[-n:]
            for s in range(len(out) - n - 1, -1, -1):                       # ★ 从后往前: 最近一次出现
                if out[s:s + n] == tail:
                    draft = out[s + n:s + n + K]
                    self.proposed += len(draft)
                    return draft, None
        return [], None


class SuffixCacheDrafter:
    """n-gram → 下一个 token 的计数表, 内容 = 以前请求的输出 + 当前请求的全部文本。"""

    def __init__(self, max_n: int = 6, alpha: float = 2.0, min_prob: float = 0.3):
        self.max_n, self.alpha, self.min_prob = max_n, alpha, min_prob
        self.history = defaultdict(Counter)      # 跨请求累积 (SuffixDecoding 的全局树)
        self.calls, self.proposed = 0, 0
        self._req, self._seen = None, 0          # 当前请求的计数表, 以及已经记进去的 token 数

    @staticmethod
    def _add(table, tokens: Sequence[int], max_n: int, start: int = 0) -> None:
        """把 tokens 里以位置 i (i ≥ start) 结尾的每个 n-gram 记一次 "后面跟着 tokens[i+1]"。"""
        for i in range(max(start, 0), len(tokens) - 1):
            for n in range(1, max_n + 1):
                if i - n + 1 < 0:
                    break
                table[tuple(tokens[i - n + 1:i + 1])][tokens[i + 1]] += 1

    def start_request(self) -> None:
        """新请求: 清空当前请求的计数表 (全局表保留)。"""
        self._req, self._seen = defaultdict(Counter), 0

    def finish_request(self, output: Sequence[int]) -> None:
        """请求结束: 把它的输出记进全局表, 供以后的请求查。"""
        self._add(self.history, list(output), self.max_n)

    def _next(self, ctx: List[int]):
        """最长匹配的后缀长度 p, 以及在该后缀下出现最多的下一个 token 与它的经验概率。"""
        for n in range(min(self.max_n, len(ctx)), 0, -1):
            key = tuple(ctx[-n:])
            c = self._req.get(key, Counter()) + self.history.get(key, Counter())
            if c:
                tok, cnt = c.most_common(1)[0]
                return n, tok, cnt / sum(c.values())
        return 0, None, 0.0

    def propose(self, out: List[int], K: int, temperature: float = 0.0, rng=None, hidden=None):
        if self._req is None:
            self.start_request()
        self._add(self._req, out, self.max_n, start=self._seen - 1)        # 增量记入当前请求的新 token
        self._seen = len(out)
        p, _, _ = self._next(out)
        max_len = min(K, int(self.alpha * p))                               # ★ 匹配越长, 敢猜得越远
        ctx, draft = list(out), []
        while len(draft) < max_len:
            _, tok, prob = self._next(ctx)
            if tok is None or prob < self.min_prob:                         # 下一步分歧太大就停
                break
            draft.append(tok)
            ctx.append(tok)
        self.proposed += len(draft)
        return draft, None


class OracleTarget:
    """合成的"会复制"的 target: 上下文与 prompt + ref[:j] 一致时, greedy 下一个 token 就是 ref[j]。

    接口对齐 core.TinyLM 里 m07 用到的部分: forward(ids, kv, return_hidden) 和 truncate_kv 可以处理的 kv。
    kv 只存已喂过的 token id (T, 1), 回滚时 m07 的 truncate_kv 照样切片。
    偏离参考答案的上下文 (验证错误草稿时才会出现) 给一个由上下文哈希决定的 token, 保证它是确定性的函数。
    """

    def __init__(self, prompt: Sequence[int], ref: Sequence[int], vocab: int):
        self.prompt, self.ref, self.V = list(prompt), list(ref), vocab
        self.forwards = 0

    def _next(self, ctx: List[int]) -> int:
        j = len(ctx) - len(self.prompt)
        if 0 <= j < len(self.ref) and ctx[:len(self.prompt)] == self.prompt and ctx[len(self.prompt):] == self.ref[:j]:
            return self.ref[j]
        return hash(tuple(ctx[-4:])) % self.V                               # 偏离路径: 任意但确定

    def forward(self, ids, kv=None, positions=None, mask=None, return_hidden=False):
        self.forwards += 1
        past = [] if kv is None else kv[0][0][:, 0].tolist()
        ctx = past + [int(t) for t in ids]
        logits = np.zeros((len(ids), self.V), dtype=np.float32)
        for i in range(len(ids)):
            logits[i, self._next(ctx[:len(past) + i + 1])] = 1.0
        arr = np.array(ctx, dtype=np.int64)[:, None]
        new_kv = [(arr, arr)]
        return (logits, new_kv, np.zeros((len(ids), 1), np.float32)) if return_hidden else (logits, new_kv)
