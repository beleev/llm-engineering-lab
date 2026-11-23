"""
diffusion_decode.py — 扩散语言模型的解码: 一步揭开几个位置, 以及没有精确 KV cache 怎么办

是什么: 掩码扩散语言模型 (LLaDA 一类) 从全是 [MASK] 的回答开始, 每一步对所有遮住的位置各给一个分布,
      选一部分位置揭开 (unmask), 直到全部揭开。一步揭开得越多, 步数越少, 也就越快。
瓶颈:   1. 模型每步给的是每个位置各自的边缘分布 p(x_i | 已揭开的部分), 同一步揭开 i 和 j 等于从
         p(x_i|·)·p(x_j|·) 里取值, 而正确的是联合分布 p(x_i|·)·p(x_j|·, x_i)。两个位置强相关时
         (短语 "high card" / "full house"), 各取各的最大值会拼出 "high house"。
      2. 注意力是双向的, 任何一个位置被揭开, 所有位置在第 2 层及以后的 K/V 都会变, 没有自回归那种
         「算一次永远有效」的 KV cache。
做法:   - 固定个数: 每步揭开置信度最高的 k 个 (LLaDA 的 low-confidence remasking), 步数 = 长度 / k。
      - 置信度阈值: 每步揭开所有置信度 > τ 的位置, 一个都没有就只揭开最高的那个 (Fast-dLLM, arXiv 2505.22618)。
      - 近似缓存: 每 r 步完整算一遍, 存下所有位置的 K/V; 中间的步只重算当前块, 块外 (前缀 + 后缀) 用旧 K/V
        (Fast-dLLM 的 DualCache; dKV-Cache arXiv 2505.15781 也是「隔几步刷新」)。
      - 块扩散 (BD3-LM, arXiv 2503.09573): 块之间因果、块内双向, 写完的块 K/V 不再变, 缓存是精确的。
简化:   Oracle 是合成的「完美模型」: 序列由互相独立的组拼成, 组内是一张短语表, 它返回精确的条件边缘分布。
      所以这里的错误全部来自解码策略本身, 不来自模型没学好。真实模型的置信度还会受训练质量影响。
简化:   近似缓存的误差在一个随机权重的 3 层双向小模型上测, 只说明误差随刷新间隔怎么变, 不说明真实模型的精度损失。
盯住:   decode() 里 fixed / threshold 两个分支选位置的那一行; CachedDecoder.step 里哪些 K/V 是旧的。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from llm_infer.core.utils import rms_norm, softmax

MASK = -1


# --------------------------------------------------------------------- #
# Oracle: 带依赖结构的合成分布, 返回精确的条件边缘分布                    #
# --------------------------------------------------------------------- #

@dataclass
class Group:
    start: int                   # 组占 [start, start + 长度) 这几个连续位置
    phrases: np.ndarray          # (m, 长度) 每行一条合法短语
    probs: np.ndarray            # (m,)


class Oracle:
    """组与组独立; 组内按短语表取值。marginals 给出每个位置在「已揭开的位置」条件下的精确边缘分布。"""

    def __init__(self, groups: List[Group], vocab: int):
        self.groups, self.V = groups, vocab
        self.L = sum(g.phrases.shape[1] for g in groups)

    def marginals(self, x: np.ndarray) -> np.ndarray:
        P = np.zeros((self.L, self.V))
        for g in self.groups:
            seg = x[g.start:g.start + g.phrases.shape[1]]
            seen = seg != MASK
            ok = (g.phrases[:, seen] == seg[seen]).all(axis=1)       # 和已揭开的位置一致的短语
            w = g.probs * ok
            # 条件分布: 只在一致的短语上重新归一。已经拼出了表里没有的组合时, 剩下的位置按无条件分布填
            w = w / w.sum() if w.sum() > 0 else g.probs
            for j in range(g.phrases.shape[1]):
                np.add.at(P[g.start + j], g.phrases[:, j], w)
        return P

    def invalid_groups(self, x: np.ndarray) -> Tuple[int, int]:
        """→ (不是合法短语的多 token 组数, 多 token 组总数)。"""
        bad = total = 0
        for g in self.groups:
            if g.phrases.shape[1] < 2:
                continue
            total += 1
            seg = x[g.start:g.start + g.phrases.shape[1]]
            bad += not (g.phrases == seg).all(axis=1).any()
        return bad, total


def make_task(rng: np.random.RandomState, L: int = 32, vocab: int = 64, p_group: float = 0.4) -> Oracle:
    """拼一条长 L 的序列: 一部分位置是独立的高置信位置, 其余位置落在 2–4 个 token 的相关短语里。
    p_group 控制相关短语的多少 (0 = 全部独立)。"""
    groups, pos = [], 0
    while pos < L:
        if rng.rand() < p_group and L - pos >= 3 and rng.rand() < 0.5:
            # 「三角」短语表: (a,b,x) (a,y,c) (z,b,c) 各约 1/3。每个位置的最大值都约 2/3,
            # 两两之间也都在表里出现过, 但 (a,b,c) 三个一起不在表里: 置信度过半还不够安全
            a_, b_, c_, x_, y_, z_ = rng.choice(vocab, 6, replace=False)
            phrases = np.array([[a_, b_, x_], [a_, y_, c_], [z_, b_, c_]])
            groups.append(Group(pos, phrases, rng.dirichlet([20, 20, 20])))
            n = 3
        elif rng.rand() < p_group and L - pos >= 2:
            n = min(rng.randint(2, 5), L - pos)
            # 3 条短语, 仿 "high card / full house / low house" 这类结构: 短语 0 自成一套 token,
            # 短语 1 和 2 只在位置 d 上不同。短语 0 概率最大但不过半时, 位置 d 的最大值来自短语 0,
            # 其余位置的最大值来自短语 1/2 共用的 token, 各位置的最大值拼起来不是表里任何一条短语
            p0, shared = rng.choice(vocab, size=(2, n), replace=False)          # 两套 token 逐位不同
            d = rng.randint(n)
            p2 = shared.copy()
            p2[d] = (shared[d] + 1 + rng.randint(vocab - 1)) % vocab
            while p2[d] == p0[d]:
                p2[d] = (p2[d] + 1) % vocab
            phrases = np.stack([p0, shared, p2])
            groups.append(Group(pos, phrases, rng.dirichlet([5, 4, 4])))
        else:
            top = rng.uniform(0.92, 0.995)                           # 独立位置: 置信度高, 和别的位置无关
            toks = rng.choice(vocab, 4, replace=False)[:, None]
            groups.append(Group(pos, toks, np.array([top] + [(1 - top) / 3] * 3)))
            n = 1
        pos += n
    return Oracle(groups, vocab)


def decode(oracle: Oracle, mode: str, k: int = 1, tau: float = 0.9) -> Tuple[np.ndarray, List[int]]:
    """贪心解码 → (结果, 每步揭开几个)。mode="fixed": 每步揭开置信度最高的 k 个; mode="threshold": 置信度 > τ 的全揭开。"""
    x = np.full(oracle.L, MASK)
    per_step = []
    while (x == MASK).any():
        P = oracle.marginals(x)
        conf = np.where(x == MASK, P.max(1), -1.0)                   # 已揭开的位置不参与
        order = np.argsort(-conf, kind="stable")
        n_masked = int((x == MASK).sum())
        if mode == "fixed":
            chosen = order[:min(k, n_masked)]
        else:
            chosen = order[:max(1, int((conf > tau).sum()))]         # ★ 一个都不过阈值时只揭开最高的那个
        x[chosen] = P[chosen].argmax(1)                              # 同一步揭开的位置各取各的最大值
        per_step.append(len(chosen))
    return x, per_step


# --------------------------------------------------------------------- #
# 双向小模型 + 近似 KV cache                                             #
# --------------------------------------------------------------------- #

class BiLM:
    """随机权重的 3 层单头 Transformer。block_causal=False 是全双向 (LLaDA), True 是块间因果、块内双向 (块扩散)。"""

    def __init__(self, vocab=64, d=32, n_layer=3, max_len=64, seed=0):
        rs = np.random.RandomState(seed)
        rand = lambda *s: (rs.randn(*s) * s[0] ** -0.5).astype(np.float32)
        self.V, self.d = vocab, d
        self.emb = rand(vocab + 1, d) * np.float32(d ** -0.5 * (vocab + 1) ** 0.5)   # 最后一行是 [MASK]
        self.pos = rand(max_len, d) * np.float32(max_len ** 0.5 / d ** 0.5)
        self.layers = [dict(g=np.ones(d, np.float32), wq=rand(d, d), wk=rand(d, d), wv=rand(d, d), wo=rand(d, d))
                       for _ in range(n_layer)]
        self.head = rand(d, vocab)

    def embed(self, x, idx):
        tok = np.where(x[idx] == MASK, self.V, x[idx])
        return self.emb[tok] + self.pos[idx]

    def run(self, x: np.ndarray, idx: np.ndarray, cache=None, allow=None):
        """只为 idx 这些位置算前向。cache[l] = (K, V) 全长, idx 以外的位置用 cache 里的 (可能是旧的);
        cache=None 时 idx 必须是全部位置。allow (len(idx), L) 为 False 的键屏蔽掉。→ (logits, 每层新算的全长 K/V)。"""
        L = len(x)
        h = self.embed(x, idx)
        new = []
        for li, Ly in enumerate(self.layers):
            u = rms_norm(h, Ly["g"])
            q, k, v = u @ Ly["wq"], u @ Ly["wk"], u @ Ly["wv"]
            if cache is None:
                K, V = k, v
            else:
                K, V = cache[li][0].copy(), cache[li][1].copy()
                K[idx], V[idx] = k, v                                # ★ 只有 idx 的 K/V 是新的, 其余是缓存里的
            s = q @ K.T / self.d ** 0.5
            if allow is not None:
                s = np.where(allow, s, -np.inf)
            h = h + softmax(s) @ V @ Ly["wo"]
            new.append((K, V))
        return rms_norm(h, 1.0) @ self.head, new


def block_allow(L: int, block: int, idx=None) -> np.ndarray:
    """块扩散的掩码: 位置 i 能看位置 j 当且仅当 j 所在的块不晚于 i 所在的块。"""
    b = np.arange(L) // block
    rows = b if idx is None else b[idx]
    return rows[:, None] >= b[None, :]


def trajectory(model: BiLM, prompt: np.ndarray, gen: int, block: int, block_causal: bool = False):
    """精确解码 (每步完整前向), 半自回归按块推进, 每步揭开当前块里置信度最高的 1 个位置。
    → 每一步开始时的序列列表 (近似缓存沿这条轨迹对比, 把缓存误差和轨迹分叉分开)。"""
    x = np.concatenate([prompt, np.full(gen, MASK)])
    L, P = len(x), len(prompt)
    allow = block_allow(L, block) if block_causal else None
    states = []
    for b0 in range(P, L, block):
        for _ in range(block):
            states.append(x.copy())
            logits, _ = model.run(x, np.arange(L), allow=allow)
            p = softmax(logits)
            act = np.arange(b0, b0 + block)
            act = act[x[act] == MASK]
            best = act[np.argmax(p[act].max(1))]
            x[best] = int(p[best].argmax())
    return states, x


def _choice(p: np.ndarray) -> Tuple[int, int]:
    """每步揭开哪个位置、填哪个 token: 置信度最高的那一行和它的 argmax。"""
    i = int(p.max(1).argmax())
    return i, int(p[i].argmax())


def dual_cache_error(model: BiLM, states, P: int, block: int, refresh: int) -> Tuple[float, float]:
    """沿精确轨迹, 每 refresh 步完整算一次并缓存 K/V, 其余步只重算当前块。
    → (当前块 logits 的平均 |Δ|, 揭开决策 (位置, token) 与精确解码不一致的步数占比)。"""
    errs, diff, cache = [], 0, None
    for t, x in enumerate(states):
        L = len(x)
        exact, full_kv = model.run(x, np.arange(L))
        if t % refresh == 0:
            cache = full_kv                                          # 刷新: 所有位置的 K/V 都是新的
        b0 = P + (t // block) * block
        act = np.arange(b0, b0 + block)
        approx, _ = model.run(x, act, cache=cache)
        errs.append(float(np.abs(approx - exact[act]).mean()))
        m = x[act] == MASK
        diff += _choice(softmax(exact[act])[m]) != _choice(softmax(approx)[m])
    return float(np.mean(errs)), diff / len(states)


def block_cache_error(model: BiLM, prompt: np.ndarray, gen: int, block: int) -> float:
    """块扩散: 写完的块算一次 K/V 存起来, 之后只算当前块。→ 与每步完整前向 (同一块因果掩码) 的最大 |Δlogit|。"""
    states, _ = trajectory(model, prompt, gen, block, block_causal=True)
    worst = 0.0
    P = len(prompt)
    cache = None
    for t, x in enumerate(states):
        L = len(x)
        allow = block_allow(L, block)
        exact, full_kv = model.run(x, np.arange(L), allow=allow)
        b0 = P + (t // block) * block
        if t % block == 0:
            # 新块开始: 前面的块都已写完。这里为了少写代码直接拿完整前向的 K/V 当缓存,
            # 块因果掩码下前面块的 K/V 与后面的块无关, 等价于「写完那块时算一次」
            cache = [(k.copy(), v.copy()) for k, v in full_kv]
        act = np.arange(b0, b0 + block)
        approx, _ = model.run(x, act, cache=cache, allow=allow[act])
        worst = max(worst, float(np.abs(approx - exact[act]).max()))
    return worst
